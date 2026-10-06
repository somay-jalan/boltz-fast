"""Rounding-preserving Pairformer kernels for CUDA (PyTorch 2.8 arithmetic).

Layer-norm Welford order follows ATen's vectorized 128-channel kernel;
softmax follows PersistentSoftmax.cuh's lane accumulation and XOR reduction.
Reference: https://github.com/pytorch/pytorch/tree/v2.8.0/aten/src/ATen/native/cuda
Do not substitute approximate exp/division or enable FP fusion globally.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def norm128(
    X,
    W,
    B,
    Y,
    ROWS: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
    EPS: tl.constexpr,
    M: tl.constexpr,
    ORDER: tl.constexpr,
    INDEX=None,
):
    r = tl.program_id(0) * M + tl.arange(0, M)
    lane = tl.arange(0, 32)
    base = (r // (N1 * N2)) * S0 + (r // N2 % N1) * S1 + (r % N2) * S2
    if INDEX is not None:
        base = tl.load(INDEX + r, r < ROWS, 0) * 128
    mean = tl.full((M, 32), 0, tl.float32)
    var = tl.full((M, 32), 0, tl.float32)
    for j in tl.static_range(4):
        x = tl.load(
            X + base[:, None] + (lane[None, :] * 4 + j) * S3, r[:, None] < ROWS, 0
        ).to(tl.float32)
        d = x - mean
        mean = tl.fma(d, 1.0 / (j + 1), mean)
        var = tl.fma(d, x - mean, var)
    count = tl.full((M, 32), 4, tl.float32)
    for shift in tl.static_range(5):
        idx = tl.broadcast_to(tl.minimum(lane + (16 >> shift), 31)[None, :], (M, 32))
        ma = tl.gather(mean, idx, 1)
        va = tl.gather(var, idx, 1)
        ca = tl.gather(count, idx, 1)
        cc = count + ca
        coef = tl.div_rn(1.0, cc)
        na = ca * coef
        nb = count * coef
        d = mean - ma
        if ORDER == 0:
            mean = tl.fma(na, ma, nb * mean)
        else:
            mean = tl.fma(nb, mean, na * ma)
        var = tl.fma((d * d) * ca, nb, va + var)
        count = cc
    mean = tl.sum(tl.where(lane[None, :] == 0, mean, 0), 1)
    var = tl.sum(tl.where(lane[None, :] == 0, var, 0), 1) / 128.0
    rs = tl.rsqrt(var + EPS)
    for j in tl.static_range(4):
        col = lane * 4 + j
        x = tl.load(X + base[:, None] + col[None, :] * S3, r[:, None] < ROWS, 0).to(
            tl.float32
        )
        w = tl.load(W + col).to(tl.float32)
        b = tl.load(B + col).to(tl.float32)
        y = tl.fma(w[None, :], rs[:, None] * (x - mean[:, None]), b[None, :])
        tl.store(Y + r[:, None] * 128 + col[None, :], y, r[:, None] < ROWS)


def norm(x, w, b, eps=1e-5, order=0, output_dtype=None, row_indices=None):
    sh = x.shape
    y = torch.empty(sh, device=x.device, dtype=output_dtype or x.dtype)
    xx = x
    while xx.ndim < 4:
        xx = xx.unsqueeze(0)
    assert xx.ndim == 4 and sh[-1] == 128
    norm128[(triton.cdiv(x.numel() // 128, 2),)](
        xx,
        w,
        b,
        y,
        x.numel() // 128,
        *xx.stride(),
        xx.shape[1],
        xx.shape[2],
        eps,
        2,
        order,
        row_indices,
        num_warps=2,
        enable_fp_fusion=False,
    )
    return y


@triton.jit
def fused(
    A,
    B,
    C,
    P,
    N: tl.constexpr,
    H: tl.constexpr,
    ROWS: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B3: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
):
    r = tl.program_id(0) * M + tl.arange(0, M)
    lanes = tl.arange(0, 32)
    bo = (r // (H * N)) * B0 + ((r // N) % H) * B1 + (r % N) * B2
    co = (r // (H * N)) * C0 + ((r // N) % H) * C1 + (r % N) * C2
    # Keep each 32-key group in warp-local registers. A wide [M,K] tensor
    # requires expensive layout conversions to reproduce the native lane sums.
    values = ()
    maximum = tl.full((M, 32), -float("inf"), tl.float32)
    for it in tl.static_range(K // 32):
        k = lanes + it * 32
        a = tl.load(
            A + r[:, None] * N + k[None, :], (r[:, None] < ROWS) & (k[None, :] < N), 0
        )
        b = tl.load(
            B + bo[:, None] + k[None, :] * B3, (r[:, None] < ROWS) & (k[None, :] < N), 0
        ).to(tl.float32)
        c = tl.load(
            C + co[:, None] + k[None, :] * C3, (r[:, None] < ROWS) & (k[None, :] < N), 0
        ).to(tl.float32)
        x = (a.to(tl.float32) + b).to(a.dtype).to(tl.float32)
        x = (x + c).to(a.dtype).to(tl.float32)
        x = tl.where(k[None, :] < N, x, -float("inf"))
        maximum = tl.maximum(maximum, x)
        values += (x,)
    maximum = tl.max(maximum, 1)
    exps = ()
    den = tl.full((M, 32), 0, tl.float32)
    for it in tl.static_range(K // 32):
        e = tl.extra.cuda.libdevice.exp(values[it] - maximum[:, None])
        exps += (e,)
        den += e
    for shift in tl.static_range(5):
        idx = tl.broadcast_to((lanes ^ (16 >> shift))[None, :], (M, 32))
        den += tl.gather(den, idx, 1)
    for it in tl.static_range(K // 32):
        k = lanes + it * 32
        p = tl.div_rn(exps[it], den)
        tl.store(
            P + r[:, None] * N + k[None, :], p, (r[:, None] < ROWS) & (k[None, :] < N)
        )


def probabilities(a, b, c):
    shape = a.shape
    a = a.reshape(-1, *shape[-3:])
    b = b.reshape(*b.shape[-4:]).expand_as(a)
    c = c.reshape(*c.shape[-4:]).expand_as(a)
    p = torch.empty_like(a)
    n = a.shape[-1]
    m = 1 if n > 256 else 2
    fused[(triton.cdiv(a.numel() // n, m),)](
        a,
        b,
        c,
        p,
        n,
        a.shape[-3],
        a.numel() // n,
        *b.stride(),
        *c.stride(),
        max(32, triton.next_power_of_2(n)),
        m,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return p.reshape(shape)
