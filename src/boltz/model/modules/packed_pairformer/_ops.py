"""Shared mathematical operations for the sole packed backend."""

import torch
from torch.utils.checkpoint import checkpoint
from . import _autograd as grad_kernels
from . import _normalization as native
from ._pointwise_kernels import silu_mul
from ._layout_kernels import pack_qkv


def norm(module, x, projection=True):
    # Each projection receives the same BF16 values as inference. Keep the
    # shared norm edge in FP32 so separate projection gradients accumulate
    # before rounding, as in native autograd.
    if projection and torch.is_grad_enabled():
        return grad_kernels.RoundedNorm.apply(x, module.weight, module.bias, module.eps)
    return native._Norm.apply(
        x,
        module.weight,
        module.bias,
        module.eps,
        torch.bfloat16 if projection else torch.float32,
    )


def silu(x, y):
    return (
        grad_kernels.Silu.apply(x, y) if grad_kernels.needed(x, y) else silu_mul(x, y)
    )


def block(module, fn, *args):
    if module.training and module.activation_checkpointing and torch.is_grad_enabled():
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def bounded_inference_probabilities(scores, mask_bias, triangle_bias):
    """Reuse inference scores for probabilities with bounded bias temporaries.

    Slice only independent softmax rows, after the full-shaped QK GEMM.
    Preserve both bias-add rounding boundaries and ATen's softmax reduction.
    This destructive helper must never be used with autograd enabled.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("In-place probability workspace requires no_grad")
    with torch.autocast("cuda", enabled=False):
        for start in range(0, scores.shape[0], 16):
            part = scores[start : start + 16]
            mb = mask_bias[start : start + 16]
            biased = (part + mb).to(scores.dtype)
            biased = (biased + triangle_bias).to(scores.dtype)
            part.copy_(biased.softmax(-1))
    return scores


def exact_attention(
    module,
    z,
    mask,
    chunk_size,
    *,
    normalized=None,
    transpose_output=True,
    qkv_pack=None
):
    if module.mha.no_heads != 4 or module.mha.c_hidden != 32:
        raise ValueError("Requires Boltz-2 triangle dimensions")
    if not module.starting:
        z, mask = z.transpose(1, 2), mask.transpose(1, 2)
    x = norm(module.layer_norm, z) if normalized is None else normalized
    bias = module.linear(x).permute(0, 3, 1, 2)
    mask_bias = module.inf * (mask - 1)
    n = x.shape[1]
    chunk_size = n if chunk_size is None else chunk_size
    outputs = []
    for start in range(0, n, chunk_size):
        # Preserve native chunk_layer's projection GEMM shape exactly.
        xx = x[0, start : start + chunk_size]
        mha = module.mha
        q0, k0, v0 = mha.linear_q(xx), mha.linear_k(xx), mha.linear_v(xx)
        pack = qkv_pack or (
            grad_kernels.PackQKV.apply if grad_kernels.needed(q0, k0, v0) else pack_qkv
        )
        q, kt, v = pack(q0, k0, v0)
        scores = torch.matmul(q, kt)
        mb = mask_bias[0, start : start + chunk_size, None, None, :]
        if n <= 1024 and scores.numel() < 2**31:
            p = native._Probabilities.apply(scores, mb, bias)
        elif not torch.is_grad_enabled():
            p = bounded_inference_probabilities(scores, mb, bias)
        else:
            # ATen uses a different reduction beyond the persistent-softmax
            # range. Also avoid exceeding the fused kernel's int32 index range. Retain that algorithm instead of extrapolating our kernel.
            scores = (scores + mb).to(scores.dtype)
            scores = (scores + bias).to(scores.dtype)
            with torch.autocast("cuda", enabled=False):
                p = scores.softmax(-1)
        out = torch.matmul(p, v).transpose(-2, -3)
        outputs.append(module.mha._wrap_up(out, xx))
    out = torch.cat(outputs).unsqueeze(0)
    return out if module.starting or not transpose_output else out.transpose(1, 2)
