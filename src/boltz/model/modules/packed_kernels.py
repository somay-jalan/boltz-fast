"""Forward-only segmented pair-biased attention for packed Boltz inference.

Each grid slice addresses one independent record/replica. Pair biases use
separate square offsets, so no cross-record attention matrix is allocated.
FP32 dot products explicitly request IEEE precision (not default TF32).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attention(
    Q, K, V, Bias, Mask, Offsets, PairOffsets, Lengths, Out,
    HEADS: tl.constexpr, DIM: tl.constexpr, MULT: tl.constexpr,
    BIAS_ROW_STRIDE: tl.constexpr, BIAS_HEAD_STRIDE: tl.constexpr,
    INF: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
):
    block, head, sequence = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(Lengths + sequence)
    if block * BM < length:
        offset = tl.load(Offsets + sequence)
        pair_offset = tl.load(PairOffsets + sequence // MULT)
        rows = block * BM + tl.arange(0, BM)
        dims = tl.arange(0, BD)
        cols = tl.arange(0, BN)
        q = tl.load(Q + ((offset + rows[:, None]) * HEADS + head) * DIM + dims[None, :],
                    mask=(rows[:, None] < length) & (dims[None, :] < DIM), other=0).to(tl.float32)
        maximum = tl.full((BM,), float('-inf'), tl.float32)
        denominator = tl.zeros((BM,), tl.float32)
        accumulator = tl.zeros((BM, BD), tl.float32)
        for start in range(tl.cdiv(length, BN)):
            keys = start * BN + cols
            k = tl.load(K + ((offset + keys[None, :]) * HEADS + head) * DIM + dims[:, None],
                        mask=(keys[None, :] < length) & (dims[:, None] < DIM), other=0).to(tl.float32)
            scores = tl.dot(q, k, input_precision="ieee") * (DIM ** -0.5)
            bias = tl.load(Bias + (pair_offset + rows[:, None] * length + keys[None, :]) * BIAS_ROW_STRIDE + head * BIAS_HEAD_STRIDE,
                           mask=(rows[:, None] < length) & (keys[None, :] < length), other=0).to(tl.float32)
            key_mask = tl.load(Mask + offset + keys, mask=keys < length, other=0).to(tl.float32)
            scores = scores + bias + (1 - key_mask[None, :]) * -INF
            scores = tl.where(keys[None, :] < length, scores, float('-inf'))
            next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
            correction = tl.exp(maximum - next_maximum)
            weights = tl.exp(scores - next_maximum[:, None])
            denominator = denominator * correction + tl.sum(weights, axis=1)
            accumulator = accumulator * correction[:, None]
            v = tl.load(V + ((offset + keys[:, None]) * HEADS + head) * DIM + dims[None, :],
                        mask=(keys[:, None] < length) & (dims[None, :] < DIM), other=0).to(tl.float32)
            accumulator += tl.dot(weights, v, input_precision="ieee")
            maximum = next_maximum
        result = accumulator / denominator[:, None]
        tl.store(Out + ((offset + rows[:, None]) * HEADS + head) * DIM + dims[None, :], result,
                 mask=(rows[:, None] < length) & (dims[None, :] < DIM))


def attention(q, k, v, bias, mask, layout, bias_layout, multiplicity, inf):
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v, bias)):
        raise RuntimeError("Packed attention is inference-only; backward is unsupported")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("Packed Q/K/V buffers must be contiguous")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Packed token self-attention requires matching Q/K/V layouts")
    if layout.lengths != tuple(n for n in bias_layout.lengths for _ in range(multiplicity)):
        raise ValueError("Pair biases do not match the record-major replica layout")
    out = torch.empty_like(v)
    heads, dim = q.shape[1:]
    _attention[(triton.cdiv(max(layout.lengths), 16), heads, len(layout.lengths))](
        q, k, v, bias, mask, layout.offsets_gpu, bias_layout.pair_offsets_gpu,
        layout.lengths_gpu, out, heads, dim, multiplicity, bias.stride(0), bias.stride(1),
        inf, 16, 32, max(16, triton.next_power_of_2(dim)), num_warps=4,
    )
    return out
