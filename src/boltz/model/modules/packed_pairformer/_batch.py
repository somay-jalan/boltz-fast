"""Packed Pairformer: tiled layouts, native-shaped GEMMs and first-order autograd."""

import torch
from boltz.model.layers.triangular_mult import TriangleMultiplicationOutgoing
from ._layout import Layout
from . import _normalization as native
from . import _math_kernels as math_kernels
from . import _layout_kernels as kernels
from ._autograd import Prepare, PackQKV
from ._ops import norm, silu, exact_attention, block
from ._pointwise_kernels import transpose_add


class PackedPairformer(torch.nn.Module):
    """Explicit fixed-length adapter; inputs/outputs use padded batch shapes.

    Lengths include internal masked positions and exclude trailing padding.
    Requires FP32 states and weights, CUDA BF16 autocast, and dropout zero in
    training. Layers operate on all records together. Matrix products remain segmented rather than concatenated GEMMs.
    """

    def __init__(self, model, lengths, width=None, *, concurrent=None):
        super().__init__()
        parameters = list(model.parameters())
        if not parameters or any(
            not p.is_cuda
            or p.dtype != torch.float32
            or p.device != parameters[0].device
            for p in parameters
        ):
            raise ValueError("Expected FP32 model parameters on one CUDA device")
        if not lengths:
            raise ValueError("At least one record length is required")
        self.model = model
        self.lengths = tuple(int(n) for n in lengths)
        self.width = max(self.lengths) if width is None else int(width)
        self.layout = Layout(self.lengths, self.width, next(model.parameters()).device)
        if concurrent is None:
            concurrent = len(self.lengths) > 1 and max(self.lengths) <= 128
        self.streams = (
            [
                torch.cuda.Stream(device=next(model.parameters()).device)
                for _ in self.lengths
            ]
            if concurrent
            else []
        )
        indices = []
        for start, n in zip(self.layout.pair_offsets, self.lengths):
            r = torch.arange(n * n, device=next(model.parameters()).device)
            indices.append(start + (r % n) * n + r // n)
        self.transpose_indices = torch.cat(indices)
        if any(layer.dropout for layer in model.layers) and model.training:
            raise ValueError("PackedPairformer training requires dropout=0")
        if any(
            att.mha.no_heads != 4 or att.mha.c_hidden != 32
            for layer in model.layers
            for att in (layer.tri_att_start, layer.tri_att_end)
        ):
            raise ValueError("Requires Boltz-2 triangle dimensions")

    def parts(self, x):
        # A single split node concatenates disjoint gradients once. Independent
        # slices each allocate/zero a full packed gradient and then add them.
        pieces = x.split(tuple(n * n for n in self.lengths), dim=0)
        return [p.reshape(1, n, n, *x.shape[1:]) for p, n in zip(pieces, self.lengths)]

    def single_parts(self, x):
        pieces = x.split(self.lengths, dim=0)
        return [p.reshape(1, n, *x.shape[1:]) for p, n in zip(pieces, self.lengths)]

    def map_records(self, fn, args):
        if not self.streams or torch.is_grad_enabled():
            return [fn(*values) for values in args]
        current = torch.cuda.current_stream(self.layout.lengths_gpu.device)
        outputs = []
        for stream, values in zip(self.streams, args):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for value in values:
                    value.record_stream(stream)
                result = fn(*values)
                result.record_stream(current)
                outputs.append(result)
        for stream in self.streams:
            current.wait_stream(stream)
        return outputs

    @staticmethod
    def join(xs):
        return torch.cat([x.reshape(-1, x.shape[-1]) for x in xs])

    def project(self, module, x):
        # Share the cast across records; keep each F.linear GEMM's shape.
        w = module.weight.to(x.dtype)
        b = None if module.bias is None else module.bias.to(x.dtype)
        if not torch.is_grad_enabled() and b is None:
            out = x.new_empty((x.shape[0], w.shape[0]))
            for start, end in zip(
                self.layout.pair_offsets[:-1], self.layout.pair_offsets[1:]
            ):
                torch.mm(x[start:end], w.t(), out=out[start:end])
            return out
        return self.join(
            [torch.nn.functional.linear(part, w, b) for part in self.parts(x)]
        )

    @staticmethod
    def contract_operator(p, gate, mask, outgoing):
        fn = Prepare.apply if torch.is_grad_enabled() else kernels.prepare
        a, b = fn(p, gate, mask, outgoing)
        return torch.bmm(a, b).permute(1, 2, 0).unsqueeze(0)

    def multiply(self, module, z, masks):
        x = norm(module.norm_in, z)
        p, g = self.project(module.p_in, x), self.project(module.g_in, x)
        fn = self.contract_operator
        ys = self.map_records(
            lambda pi, gi, mi: fn(
                pi, gi, mi, isinstance(module, TriangleMultiplicationOutgoing)
            ),
            zip(self.parts(p), self.parts(g), masks),
        )
        y = norm(module.norm_out, self.join(ys))
        return self.project(module.p_out, y) * self.project(module.g_out, x).sigmoid()

    def attention(self, module, z, masks):
        zs = self.parts(z)
        if not module.starting and not torch.is_grad_enabled():
            ln = module.layer_norm
            normalized = math_kernels.norm(
                z,
                ln.weight,
                ln.bias,
                ln.eps,
                output_dtype=torch.bfloat16,
                row_indices=self.transpose_indices,
            )
            updates = self.map_records(
                lambda zi, mi, xi: self.attention_operator(
                    module, zi, mi, None, normalized=xi, transpose_output=False
                ),
                zip(zs, masks, self.parts(normalized)),
            )
            return self.join(updates)
        values = z if module.starting else self.join([p.transpose(1, 2) for p in zs])
        xs = self.parts(norm(module.layer_norm, values))
        return self.join(
            self.map_records(
                lambda zi, mi, xi: self.attention_operator(
                    module, zi, mi, None, normalized=xi
                ),
                zip(zs, masks, xs),
            )
        )

    def layer(self, layer, s, z, masks, pair_masks):
        z = z + self.multiply(layer.tri_mul_out, z, pair_masks)
        z = z + self.multiply(layer.tri_mul_in, z, pair_masks)
        z = z + self.attention(layer.tri_att_start, z, pair_masks)
        update = self.attention(layer.tri_att_end, z, pair_masks)
        z = (
            z + update
            if torch.is_grad_enabled()
            else transpose_add(z, update, self.transpose_indices)
        )
        t = layer.transition_z
        x = norm(t.norm, z)
        z = z + self.project(
            t.fc3, silu(self.project(t.fc1, x), self.project(t.fc2, x))
        )
        singles = []
        with torch.autocast("cuda", enabled=False):
            for si, zi, mi in zip(self.single_parts(s), self.parts(z), masks):
                sn = layer.pre_norm_s(si.float())
                si = si.float() + native._token_attention(
                    layer.attention, sn, zi.float(), mi
                )
                singles.append(layer.s_post_norm(si + layer.transition_s(si)))
        return self.join(singles), z

    @torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
    def forward(self, s, z, mask, pair_mask):
        if (
            s.shape != (len(self.lengths), self.width, 384)
            or z.shape != (len(self.lengths), self.width, self.width, 128)
            or mask.shape != s.shape[:2]
            or pair_mask.shape != z.shape[:3]
        ):
            raise ValueError("Inputs must match the adapter's fixed batch layout")
        if s.dtype != torch.float32 or z.dtype != torch.float32 or not s.is_cuda:
            raise ValueError("Expected FP32 CUDA states")
        if any(
            t.device != self.layout.lengths_gpu.device or t.dtype != torch.float32
            for t in (s, z, mask, pair_mask)
        ):
            raise ValueError(
                "All four inputs must be FP32 on the adapter's CUDA device"
            )
        if mask.requires_grad or pair_mask.requires_grad:
            raise ValueError("Masks are nondifferentiable metadata")
        if self.model.training and any(l.dropout for l in self.model.layers):
            raise ValueError("PackedPairformer training requires dropout=0")
        masks = list(self.layout.parts(self.layout.pack(mask)))
        pair_masks = list(self.layout.parts(self.layout.pack(pair_mask, True), True))
        s, z = self.layout.pack(s), self.layout.pack(z, True)
        for layer in self.model.layers:
            s, z = block(
                self.model,
                lambda si, zi, l=layer: self.layer(l, si, zi, masks, pair_masks),
                s,
                z,
            )
        return self.layout.unpack(s), self.layout.unpack(z, True)

    @staticmethod
    def attention_operator(module, z, mask, chunk_size, **kwargs):
        pack = PackQKV.apply if torch.is_grad_enabled() else kernels.pack_qkv
        return exact_attention(module, z, mask, chunk_size, qkv_pack=pack, **kwargs)
