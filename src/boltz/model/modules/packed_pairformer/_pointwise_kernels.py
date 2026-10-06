"""SiLU product and transposed residual update kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def silu_mul_kernel(X, Y, O, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, 0)
    y = tl.load(Y + i, i < N, 0)
    a = tl.div_rn(
        x.to(tl.float32), 1 + tl.extra.cuda.libdevice.exp(-x.to(tl.float32))
    ).to(x.dtype)
    tl.store(O + i, a.to(tl.float32) * y.to(tl.float32), i < N)


def silu_mul(x, y):
    out = torch.empty_like(x)
    silu_mul_kernel[(triton.cdiv(x.numel(), 1024),)](
        x, y, out, x.numel(), 1024, num_warps=4, enable_fp_fusion=False
    )
    return out


@triton.jit
def _transpose_add(Z, U, INDEX, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(INDEX + i // 128, i < N, 0)
    z = tl.load(Z + i, i < N, 0).to(tl.float32)
    u = tl.load(U + row * 128 + i % 128, i < N, 0).to(tl.float32)
    tl.store(OUT + i, z + u, i < N)


def transpose_add(z, update, index):
    out = torch.empty_like(z)
    _transpose_add[(triton.cdiv(z.numel(), 1024),)](
        z, update, index, out, z.numel(), 1024, enable_fp_fusion=False
    )
    return out
