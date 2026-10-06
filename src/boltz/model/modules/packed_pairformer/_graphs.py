"""Fixed-layout CUDA graphs for inference and first-order training."""

import torch
from boltz.model.layers.pairformer import PairformerModule
from ._batch import PackedPairformer
from ._layout import _lengths


class PackedInferenceGraph:
    """Capture the packed adapter for fixed-layout inference replay.

    Call under no_grad. Current input and mask values, and in-place weight
    updates, are read on every replay. Lengths/width/device are fixed; rebuild
    after replacing parameter storage, changing the model, or moving devices.
    Do not call concurrently from multiple host threads or CUDA streams.
    """

    adapter_type = PackedPairformer

    def __init__(self, model, sample_inputs, lengths, *, concurrent=None):
        if model.training:
            raise ValueError("Inference capture requires model.eval()")
        self.model = model
        self.adapter = self.adapter_type(
            model, lengths, sample_inputs[0].shape[1], concurrent=concurrent
        )
        self.signature = tuple((x.shape, x.dtype, x.device) for x in sample_inputs)
        with torch.no_grad():
            self.static = tuple(x.detach().clone() for x in sample_inputs)
            stream = torch.cuda.Stream(device=sample_inputs[0].device)
            current = torch.cuda.current_stream(sample_inputs[0].device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.outputs = self.adapter(*self.static)
            current.wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.outputs = self.adapter(*self.static)

    def __call__(self, *inputs):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Inference graph requires torch.no_grad(); use the adapter for autograd"
            )
        if self.model.training:
            raise ValueError("Inference graph requires model.eval()")
        if tuple((x.shape, x.dtype, x.device) for x in inputs) != self.signature:
            raise ValueError("Inputs differ from the captured layout")
        for dst, src in zip(self.static, inputs):
            dst.copy_(src)
        self.graph.replay()
        return tuple(x.clone() for x in self.outputs)


def _signature(tensors):
    return tuple((tuple(x.shape), x.dtype, x.device, x.requires_grad) for x in tensors)


class PackedTrainingGraph:
    """First-order, one-forward/one-backward callable with shared model weights.

    Rebuild for a new shape, token-length layout, training configuration, or
    parameter storage. In-place optimizer updates and changing input/mask values
    are supported. Cached inputs need not require gradients. Gradient accumulation
    across completed forward/backward steps is supported; overlapping forwards
    and retained-graph backward are not. This initial interface requires dropout
    zero and FP32 inputs/parameters, with BF16 pair projections internally.

    Construct with ``PackedTrainingGraph(model, (s, z, mask, pair_mask))``.
    Call the returned object in place of the standalone Pairformer. Continue to
    use the original model's parameters, optimizer, zero_grad, and state_dict.
    """

    def __init__(self, model, sample_inputs):
        if not isinstance(model, PairformerModule) or not model.training:
            raise ValueError("A standalone PairformerModule in train mode is required")
        if len(sample_inputs) != 4:
            raise ValueError("Expected (s, z, mask, pair_mask)")
        if any(layer.dropout for layer in model.layers):
            raise ValueError("Training graph currently requires dropout=0")
        if any(
            layer.tri_att_start.mha.no_heads != 4
            or layer.tri_att_start.mha.c_hidden != 32
            for layer in model.layers
        ):
            raise ValueError("Training graph requires Boltz-2 triangle dimensions")
        s, z, mask, pair_mask = sample_inputs
        if (
            s.ndim != 3
            or z.ndim != 4
            or z.shape[:3] != (s.shape[0], s.shape[1], s.shape[1])
            or mask.shape != s.shape[:2]
            or pair_mask.shape != z.shape[:3]
        ):
            raise ValueError("Input batch, token, and mask shapes are inconsistent")
        if any(x.device != s.device or not x.is_cuda for x in sample_inputs):
            raise ValueError("All inputs must be on the same CUDA device")
        if s.dtype != torch.float32 or z.dtype != torch.float32:
            raise ValueError("Training graph currently requires FP32 s and z")
        if s.shape[-1] != 384 or z.shape[-1] != 128:
            raise ValueError("Expected 384 single channels and 128 pair channels")
        if mask.requires_grad or pair_mask.requires_grad:
            raise ValueError("Masks are nondifferentiable metadata")
        if any(
            p.dtype != torch.float32 or p.device != s.device for p in model.parameters()
        ):
            raise ValueError("Model parameters must be FP32 on the input CUDA device")
        self.model = model
        self.lengths = _lengths(mask)
        self.signature = _signature(sample_inputs)
        self.storage = self._storage()
        self.configuration = self._configuration()
        self.pending = False
        self.generation = 0
        samples = tuple(
            x.detach().clone().requires_grad_(x.requires_grad) for x in sample_inputs
        )
        batch = PackedPairformer(model, self.lengths, width=s.shape[1])
        # CUDA replay keeps raw addresses, not Python references to layout
        # tensors allocated before capture. Retain the adapter (and its index
        # buffers) for the complete graph lifetime.
        self.batch = batch
        names, parameters = zip(*batch.named_parameters())
        # Fresh autograd leaves sharing the same weight storage avoid stale
        # AccumulateGrad stream references from an earlier eager training step.
        # Replays take the ORIGINAL parameters as explicit autograd inputs, so
        # optimizer ownership and gradient accumulation remain unchanged. No
        # process-global stream-override setting is needed.
        parameter_samples = tuple(
            p.detach().requires_grad_(p.requires_grad) for p in parameters
        )

        def forward(s, z, mask, pair_mask, *weights):
            return torch.func.functional_call(
                batch, dict(zip(names, weights)), (s, z, mask, pair_mask), strict=True
            )

        # Crucial: capture backward OUTSIDE autocast. PackedPairformer enables it
        # only around forward; capturing backward under BF16 changes FP32 token
        # GEMMs and silently changes gradients despite bitwise forward equality.
        with torch.autocast("cuda", enabled=False):
            self.graph = torch.cuda.make_graphed_callables(
                forward, (*samples, *parameter_samples), num_warmup_iters=2
            )

    def _storage(self):
        return tuple(
            (id(p), p.data_ptr(), p.dtype, p.device, p.requires_grad)
            for p in self.model.parameters()
        )

    def _configuration(self):
        return (
            self.model.activation_checkpointing,
            tuple((id(m), m.training) for m in self.model.modules()),
            tuple(layer.dropout for layer in self.model.layers),
        )

    def __call__(self, s, z, mask, pair_mask):
        if not torch.is_grad_enabled():
            raise RuntimeError("This callable is for training with gradients enabled")
        if self.pending:
            raise RuntimeError(
                "Complete backward before the next training-graph forward"
            )
        if _signature((s, z, mask, pair_mask)) != self.signature:
            raise ValueError(
                "Input shape/dtype/device/requires_grad changed; rebuild graph"
            )
        if _lengths(mask) != self.lengths:
            raise ValueError("Record lengths changed; rebuild training graph")
        if (
            self._storage() != self.storage
            or self._configuration() != self.configuration
        ):
            raise ValueError(
                "Model storage or training configuration changed; rebuild graph"
            )
        with torch.autocast("cuda", enabled=False):
            out = self.graph(s, z, mask, pair_mask, *self.model.parameters())
        self.pending = True
        self.generation += 1
        generation = self.generation

        def before_backward(*unused):
            if not self.pending or self.generation != generation:
                raise RuntimeError(
                    "A training-graph forward supports exactly one backward"
                )

        def completed(*unused):
            self.pending = False

        out[0].grad_fn.register_prehook(before_backward)
        out[0].grad_fn.register_hook(completed)
        # Retained user outputs must not alias storage overwritten by replay.
        return tuple(x.clone() for x in out)
