"""Inference-only packed execution for variable-size independent records.

Token rows and separate pair squares stay packed across a complete stack.
Dense tensors exist only at compatibility boundaries. Biological chain IDs
are never used as packing boundaries. Atom windows retain their original
neighbourhoods and only their final partial windows require padding.
"""

from contextvars import ContextVar
from functools import wraps
import hashlib

import torch

from boltz.data import const


_CONTEXT = ContextVar("boltz_packed_context", default=None)


def _lengths(mask):
    # Preserve holes in a mask; only discard trailing padding.
    indices = torch.arange(mask.shape[-1], device=mask.device) + 1
    return tuple((mask.bool() * indices).amax(-1).clamp_min(1).cpu().tolist())


class Layout:
    """Offsets for token rows and flattened per-record pair squares."""

    def __init__(self, lengths, width, device):
        self.lengths = tuple(int(n) for n in lengths)
        self.width = width
        self.batch_size = len(lengths)
        if not lengths or min(lengths) < 1 or max(lengths) > width:
            raise ValueError("Invalid packed record lengths")
        self.offsets = [0]
        self.pair_offsets = [0]
        rows, pairs = [], []
        for i, n in enumerate(lengths):
            self.offsets.append(self.offsets[-1] + n)
            self.pair_offsets.append(self.pair_offsets[-1] + n * n)
            r = torch.arange(n, device=device)
            rows.append(i * width + r)
            pairs.append((i * width * width + r[:, None] * width + r).flatten())
        self.row_indices = torch.cat(rows)
        self.pair_indices = torch.cat(pairs)
        self.offsets_gpu = torch.tensor(self.offsets, device=device, dtype=torch.int32)
        self.pair_offsets_gpu = torch.tensor(self.pair_offsets, device=device, dtype=torch.int32)
        self.lengths_gpu = torch.tensor(lengths, device=device, dtype=torch.int32)
        self.is_dense = all(n == width for n in lengths)
        self.kernel_tasks = {}

    def pack(self, x, pair=False):
        rank = 3 if pair else 2
        flat = x.reshape(-1, *x.shape[rank:])
        if self.is_dense:
            return flat
        return flat.index_select(0, self.pair_indices if pair else self.row_indices)

    def unpack(self, x, pair=False):
        shape = (self.batch_size, self.width)
        if pair:
            shape += (self.width,)
        shape += x.shape[1:]
        if self.is_dense:
            return x.reshape(shape)
        out = x.new_zeros(shape)
        out.reshape(-1, *x.shape[1:]).index_copy_(
            0, self.pair_indices if pair else self.row_indices, x
        )
        return out

    def parts(self, x, pair=False):
        offsets = self.pair_offsets if pair else self.offsets
        for i, n in enumerate(self.lengths):
            shape = (1, n, n) if pair else (1, n)
            yield x[offsets[i]:offsets[i + 1]].reshape(*shape, *x.shape[1:])


class Context:
    def __init__(self, feats):
        self.tokens = _lengths(feats["token_pad_mask"])
        self.atoms = _lengths(feats["atom_pad_mask"])
        self.layouts = {}
        self.windows = {}

    def layout(self, mask, kind="tokens"):
        base = getattr(self, kind)
        batch, width = mask.shape[:2]
        if batch % len(base):
            raise ValueError("Packed record/replica dimensions are inconsistent")
        mult = batch // len(base)
        lengths = tuple(n for n in base for _ in range(mult))
        key = (kind, batch, width, mask.device)
        if key not in self.layouts:
            self.layouts[key] = Layout(lengths, width, mask.device)
        return self.layouts[key]


def layout_for(mask, pair=False):
    ctx = _CONTEXT.get()
    if ctx is not None:
        return ctx.layout(mask)
    if pair:
        mask = mask.bool().any(-1) | mask.bool().any(-2)
    return Layout(_lengths(mask), mask.shape[1], mask.device)


def packed_context(forward):
    @wraps(forward)
    def wrapped(self, feats, *args, **kwargs):
        if not getattr(self, "packed_inference", False):
            return forward(self, feats, *args, **kwargs)
        if self.training:
            raise RuntimeError("Packed execution currently supports inference only")
        token = _CONTEXT.set(Context(feats))
        try:
            return forward(self, feats, *args, **kwargs)
        finally:
            _CONTEXT.reset(token)
    return wrapped


def enable_packed(model, pair_backend="sequential"):
    """Enable the optional implementation without changing checkpoint weights."""
    if model.training:
        raise ValueError("Call eval() before enabling packed inference")
    if pair_backend not in {"sequential", "triton"}:
        raise ValueError(f"Unknown packed pair backend: {pair_backend}")
    for module in model.modules():
        module.packed_inference = True
        module.packed_pair_backend = pair_backend


def segmented_attention(q, k, v, bias, mask, layout, bias_layout, multiplicity, inf):
    """Reference segmented attention; CUDA uses the fused packed implementation."""
    if q.is_cuda:
        from boltz.model.modules.packed_kernels import attention
        return attention(q, k, v, bias, mask, layout, bias_layout, multiplicity, inf)
    out = []
    for i, n in enumerate(layout.lengths):
        a, b = layout.offsets[i:i + 2]
        j = i // multiplicity
        u, w = bias_layout.pair_offsets[j:j + 2]
        qi, ki, vi = q[a:b], k[a:b], v[a:b]
        bi = bias[u:w].reshape(n, n, -1).permute(2, 0, 1)
        logits = torch.einsum("ihd,jhd->hij", qi.float(), ki.float())
        logits = logits / q.shape[-1] ** 0.5 + bi.float()
        logits = logits + (1 - mask[a:b].float())[None, None] * -inf
        out.append(torch.einsum("hij,jhd->ihd", logits.softmax(-1), vi.float()).to(v.dtype))
    return torch.cat(out)


def token_attention(module, s, z, mask, layout, bias_layout=None, multiplicity=1):
    bias_layout = layout if bias_layout is None else bias_layout
    h, d = module.num_heads, module.head_dim
    q = module.proj_q(s).reshape(-1, h, d)
    k = module.proj_k(s).reshape(-1, h, d)
    v = module.proj_v(s).reshape(-1, h, d)
    if module.compute_pair_bias:
        bias = module.proj_z[1](module.proj_z[0](z))
    else:
        bias = z
    g = module.proj_g(s).sigmoid()
    with torch.autocast(s.device.type, enabled=False):
        out = segmented_attention(q, k, v, bias, mask, layout, bias_layout, multiplicity, module.inf)
    return module.proj_o(g * out.reshape(-1, module.c_s))


def pair_operation(module, values, masks, layout, use_kernels, attention=False):
    """Dispatch record-local pair operations without a padded global square."""
    if getattr(module, "packed_pair_backend", "sequential") == "triton":
        from boltz.model.modules.packed_triangles import triangle_attention, triangle_multiply
        operation = triangle_attention if attention else triangle_multiply
        return operation(module, values, masks, layout)
    results = []
    for n, value, mask in zip(layout.lengths, layout.parts(values, True), layout.parts(masks, True)):
        args = {"mask": mask, "use_kernels": use_kernels}
        if attention:
            args["chunk_size"] = 128 if n > const.chunk_size_threshold else 512
        result = module(value, **args)
        results.append(result.reshape(-1, result.shape[-1]))
    return torch.cat(results)


def pair_layer(layer, z, pair_mask, layout, use_kernels):
    z = z + pair_operation(layer.tri_mul_out, z, pair_mask, layout, use_kernels)
    z = z + pair_operation(layer.tri_mul_in, z, pair_mask, layout, use_kernels)
    z = z + pair_operation(layer.tri_att_start, z, pair_mask, layout, use_kernels, True)
    z = z + pair_operation(layer.tri_att_end, z, pair_mask, layout, use_kernels, True)
    return z + layer.transition_z(z)


def pairformer(module, s, z, mask, pair_mask, use_kernels):
    layout = layout_for(mask if s is not None else pair_mask, pair=s is None)
    z = layout.pack(z, True)
    pair_mask = layout.pack(pair_mask, True)
    if s is not None:
        s = layout.pack(s)
        mask = layout.pack(mask)
    for layer in module.layers:
        z = pair_layer(layer, z, pair_mask, layout, use_kernels)
        if s is not None:
            with torch.autocast(s.device.type, enabled=False):
                normed = layer.pre_norm_s(s.float())
                s = s.float() + token_attention(layer.attention, normed, z.float(), mask.float(), layout)
                s = layer.s_post_norm(s + layer.transition_s(s))
    if s is None:
        return layout.unpack(z, True)
    return layout.unpack(s), layout.unpack(z, True)


def diffusion_transformer(module, a, s, bias, mask, multiplicity):
    layout = layout_for(mask)
    # Biases are shared by replicas, never by independent records.
    base_lengths = layout.lengths[::multiplicity]
    ctx = _CONTEXT.get()
    key = ("diffusion_bias", base_lengths, bias.shape[1], bias.device)
    if ctx is not None and key in ctx.layouts:
        bias_layout = ctx.layouts[key]
    else:
        bias_layout = Layout(base_lengths, bias.shape[1], bias.device)
        if ctx is not None:
            ctx.layouts[key] = bias_layout
    a, s, mask = layout.pack(a), layout.pack(s), layout.pack(mask)
    bias = bias_layout.pack(bias, True)
    bias = bias.reshape(bias.shape[0], len(module.layers), -1)
    for i, layer in enumerate(module.layers):
        b = layer.adaln(a, s)
        b = token_attention(layer.pair_bias_attn, b, bias[:, i], mask, layout, bias_layout, multiplicity)
        a = a + layer.output_projection(s) * b
        a = layer.post_lnorm(a + layer.transition(a, s))
    return layout.unpack(a)


def atom_transformer(module, q, c, bias, mask, multiplicity):
    """Concatenate record-local windows and use a bounded indexed key lookup."""
    batch, atoms, channels = q.shape
    W, H = module.attn_window_queries, module.attn_window_keys
    if atoms % W or W % 2 or H % W:
        raise ValueError("Packed atom windows require the model's aligned window sizes")
    ctx = _CONTEXT.get()
    if ctx is None:
        lengths = _lengths(mask)
    else:
        lengths = tuple(n for n in ctx.atoms for _ in range(batch // len(ctx.atoms)))
    key = (batch, atoms, W, H, mask.device)
    cached = ctx.windows.get(key) if ctx is not None else None
    if cached is None:
        window_counts = [(n + W - 1) // W for n in lengths]
        old_windows = atoms // W
        take, key_indices, valid_keys = [], [], []
        offset = 0
        for i, count in enumerate(window_counts):
            take.extend(i * old_windows + j for j in range(count))
            # This is the same half-window offset as get_indexing_matrix.
            start = torch.arange(count, device=q.device)[:, None] * W - (H - W) // 2
            local = start + torch.arange(H, device=q.device)
            valid = (local >= 0) & (local < count * W)
            key_indices.append((offset + local.clamp(0, count * W - 1)).flatten())
            valid_keys.append(valid.flatten())
            offset += count * W
        cached = (torch.tensor(take, device=q.device), torch.cat(key_indices), torch.cat(valid_keys))
        if ctx is not None:
            ctx.windows[key] = cached
    take, key_indices, valid_keys = cached
    q = q.reshape(-1, W, channels).index_select(0, take)
    c = c.reshape(-1, W, c.shape[-1]).index_select(0, take)
    mask = mask.reshape(-1, W).index_select(0, take)
    bias = bias.repeat_interleave(multiplicity, 0).reshape(-1, W, H, bias.shape[-1]).index_select(0, take)

    def to_keys(x):
        values = x.reshape(-1, x.shape[-1]).index_select(0, key_indices)
        values = values * valid_keys[:, None].to(values.dtype)
        return values.reshape(-1, H, x.shape[-1])

    # to_keys makes this fixed-window attention, not token attention.
    q = module.diffusion_transformer(q, c, bias=bias, mask=mask.float(), to_keys=to_keys, multiplicity=1)
    out = q.new_zeros(batch * (atoms // W), W, channels)
    out.index_copy_(0, take, q)
    return out.reshape(batch, atoms, channels)


def msa_module(module, z, emb, feats, use_kernels):
    """Pack selected MSA rows before one-hot expansion and keep layers packed."""
    layout = layout_for(feats["token_pad_mask"])
    msa_parts, mask_parts, shapes = [], [], []
    for i, n in enumerate(layout.lengths):
        raw_mask = feats["msa_mask"][i, :, :n] if isinstance(feats["msa_mask"], torch.Tensor) else feats["msa_mask"][i][:, :n]
        indices = torch.nonzero(raw_mask.bool().any(-1), as_tuple=False).squeeze(-1)
        if module.subsample_msa:
            count = min(indices.numel(), module.num_subsampled_msa)
            generator = None
            if "record" in feats:
                record_id = feats["record"][i].id
                occurrence = module._inference_msa_calls.get(record_id, 0)
                module._inference_msa_calls[record_id] = occurrence + 1
                record_seed = int.from_bytes(hashlib.blake2b(record_id.encode(), digest_size=8).digest(), "little")
                seed = (torch.initial_seed() + record_seed + occurrence) % (2**63 - 1)
                generator = torch.Generator(device=indices.device).manual_seed(seed)
            order = torch.randperm(indices.numel(), device=indices.device, generator=generator)[:count]
            indices = indices[order]
        if indices.numel() == 0:
            raise ValueError("A packed MSA record has no valid rows")
        raw = feats["msa"][i][indices, :n]
        parts = [torch.nn.functional.one_hot(raw, num_classes=const.num_tokens)]
        for name in ["has_deletion", "deletion_value"] + (["msa_paired"] if module.use_paired_feature else []):
            parts.append(feats[name][i][indices, :n].unsqueeze(-1))
        m = torch.cat(parts, -1)
        msa_parts.append(m.reshape(-1, m.shape[-1]))
        mask_parts.append(raw_mask[indices].unsqueeze(0))
        shapes.append((indices.numel(), n))
    sizes = [m * n for m, n in shapes]
    m = module.msa_proj(torch.cat(msa_parts))
    projected_s = module.s_proj(layout.pack(emb))
    broadcasts = [s.expand(rows, n, -1).reshape(rows * n, -1) for s, (rows, n) in zip(layout.parts(projected_s), shapes)]
    m = m + torch.cat(broadcasts)
    z = layout.pack(z, True)
    pair_mask = layout.pack(feats["token_pad_mask"][:, :, None].float() * feats["token_pad_mask"][:, None, :].float(), True)
    grouped = getattr(module, "packed_pair_backend", "sequential") == "triton"
    if grouped:
        from boltz.model.modules.grouped_msa import (
            MSAPlan, pair_weighted_averaging, outer_product_mean,
        )
        plan_key = ("msa", tuple(shapes))
        if plan_key not in layout.kernel_tasks:
            layout.kernel_tasks[plan_key] = MSAPlan(shapes, m.device)
        plan = layout.kernel_tasks[plan_key]
        msa_mask = torch.cat([mask.flatten() for mask in mask_parts])
    for layer in module.layers:
        if grouped:
            m = m + pair_weighted_averaging(layer.pair_weighted_averaging, m, z, pair_mask, plan)
            m = m + layer.msa_transition(m)
            z = z + outer_product_mean(layer.outer_product_mean, m, msa_mask, plan)
        else:
            updates = []
            for mi, zi, mask_i, (rows, n) in zip(m.split(sizes), layout.parts(z, True), layout.parts(pair_mask, True), shapes):
                update = layer.pair_weighted_averaging(mi.reshape(1, rows, n, -1), zi, mask_i, n > const.chunk_size_threshold)
                updates.append(update.reshape(rows * n, -1))
            m = m + torch.cat(updates)
            m = m + layer.msa_transition(m)
            updates = []
            for mi, mask_i, (rows, n) in zip(m.split(sizes), mask_parts, shapes):
                update = layer.outer_product_mean(mi.reshape(1, rows, n, -1), mask_i, 4 if n > const.chunk_size_threshold else None)
                updates.append(update.reshape(n * n, -1))
            z = z + torch.cat(updates)
        z = pair_layer(layer.pairformer_layer, z, pair_mask, layout, use_kernels)
    return layout.unpack(z, True)
