"""Explicitly tiled layout conversion for packed Pairformer batches."""

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def qkv_tiled(Q, K, V, QO, KO, VO, A: gl.constexpr, N: gl.constexpr, T: gl.constexpr):
    L: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    O: gl.constexpr = gl.BlockedLayout([4, 1], [8, 4], [1, 4], [0, 1])
    r = gl.program_id(0) * T + gl.arange(0, T, layout=gl.SliceLayout(1, L))
    c = gl.arange(0, 128, layout=gl.SliceLayout(0, L))
    src = r[:, None] * 128 + c[None, :]
    valid = r[:, None] < A * N
    q = gl.load(Q + src, valid, 0).to(gl.float32) * (32**-0.5)
    v = gl.load(V + src, valid, 0)
    qo = (((r // N)[:, None] * 4 + c[None, :] // 32) * N + (r % N)[:, None]) * 32 + c[
        None, :
    ] % 32
    gl.store(QO + qo, q, valid)
    gl.store(VO + qo, v, valid)
    k = gl.load(K + src, valid, 0)
    k = gl.convert_layout(k, O)
    rr = gl.program_id(0) * T + gl.arange(0, T, layout=gl.SliceLayout(1, O))
    cc = gl.arange(0, 128, layout=gl.SliceLayout(0, O))
    ko = (((rr // N)[:, None] * 4 + cc[None, :] // 32) * 32 + cc[None, :] % 32) * N + (
        rr % N
    )[:, None]
    gl.store(KO + ko, k, rr[:, None] < A * N)


def pack_qkv(q, k, v, tile=32):
    a, n, _ = q.shape
    qo = torch.empty((a, 4, n, 32), device=q.device, dtype=q.dtype)
    ko = torch.empty((a, 4, 32, n), device=q.device, dtype=q.dtype)
    vo = torch.empty_like(qo)
    qkv_tiled[(triton.cdiv(a * n, tile),)](
        q, k, v, qo, ko, vo, a, n, tile, num_warps=4, enable_fp_fusion=False
    )
    return qo, ko, vo


@g.jit
def prepare_tiled(
    P,
    G,
    M,
    A,
    B,
    N: gl.constexpr,
    OUT: gl.constexpr,
    MR: gl.constexpr,
    MC: gl.constexpr,
    I: gl.constexpr,
    J: gl.constexpr,
    C: gl.constexpr,
):
    L: gl.constexpr = gl.BlockedLayout([1, 1, 2], [1, 4, 8], [4, 1, 1], [2, 1, 0])
    LA: gl.constexpr = gl.BlockedLayout([1, 2, 1], [2, 8, 2], [2, 1, 2], [1, 0, 2])
    LB: gl.constexpr = gl.BlockedLayout([2, 1, 1], [8, 2, 2], [1, 2, 2], [0, 1, 2])
    ii = gl.program_id(0) * I + gl.arange(
        0, I, layout=gl.SliceLayout(1, gl.SliceLayout(2, L))
    )
    jj = gl.program_id(1) * J + gl.arange(
        0, J, layout=gl.SliceLayout(0, gl.SliceLayout(2, L))
    )
    cc = gl.program_id(2) * C + gl.arange(
        0, C, layout=gl.SliceLayout(0, gl.SliceLayout(1, L))
    )
    off = (ii[:, None, None] * N + jj[None, :, None]) * 256 + cc[None, None, :]
    valid = (ii[:, None, None] < N) & (jj[None, :, None] < N)
    mask = gl.load(M + ii[:, None, None] * MR + jj[None, :, None] * MC, valid, 0)
    for half in gl.static_range(2):
        p = gl.load(P + off + half * 128, valid, 0)
        gate = gl.load(G + off + half * 128, valid, 0).to(gl.float32)
        gate = (
            gl.div_rn(1.0, 1.0 + gl.extra.libdevice.exp(-gate))
            .to(p.dtype)
            .to(gl.float32)
        )
        y = (p.to(gl.float32) * gate).to(p.dtype).to(gl.float32) * mask
        # Explicit conversion gives coalesced stores for both orientations.
        if (OUT and half == 0) or (not OUT and half == 1):
            y = gl.convert_layout(y, LA)
            i = gl.program_id(0) * I + gl.arange(
                0, I, layout=gl.SliceLayout(1, gl.SliceLayout(2, LA))
            )
            j = gl.program_id(1) * J + gl.arange(
                0, J, layout=gl.SliceLayout(0, gl.SliceLayout(2, LA))
            )
            c = gl.program_id(2) * C + gl.arange(
                0, C, layout=gl.SliceLayout(0, gl.SliceLayout(1, LA))
            )
            dst = c[None, None, :] * N * N + i[:, None, None] * N + j[None, :, None]
        else:
            y = gl.convert_layout(y, LB)
            i = gl.program_id(0) * I + gl.arange(
                0, I, layout=gl.SliceLayout(1, gl.SliceLayout(2, LB))
            )
            j = gl.program_id(1) * J + gl.arange(
                0, J, layout=gl.SliceLayout(0, gl.SliceLayout(2, LB))
            )
            c = gl.program_id(2) * C + gl.arange(
                0, C, layout=gl.SliceLayout(0, gl.SliceLayout(1, LB))
            )
            dst = c[None, None, :] * N * N + j[None, :, None] * N + i[:, None, None]
        if half == 0:
            gl.store(A + dst, y, (i[:, None, None] < N) & (j[None, :, None] < N))
        else:
            gl.store(B + dst, y, (i[:, None, None] < N) & (j[None, :, None] < N))


def prepare(p, gate, mask, outgoing, tile=(16, 16, 32)):
    n = p.shape[1]
    a, b = [torch.empty(128, n, n, device=p.device, dtype=p.dtype) for _ in range(2)]
    i, j, c = tile
    prepare_tiled[(triton.cdiv(n, i), triton.cdiv(n, j), triton.cdiv(128, c))](
        p,
        gate,
        mask,
        a,
        b,
        n,
        outgoing,
        *mask.stride()[-2:],
        i,
        j,
        c,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return a, b


@g.jit
def qkv_backward_tiled(
    DQ, DK, DV, XQ, XK, XV, A: gl.constexpr, N: gl.constexpr, T: gl.constexpr
):
    L: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    O: gl.constexpr = gl.BlockedLayout([4, 1], [8, 4], [1, 4], [0, 1])
    r = gl.program_id(0) * T + gl.arange(0, T, layout=gl.SliceLayout(1, L))
    c = gl.arange(0, 128, layout=gl.SliceLayout(0, L))
    dst = r[:, None] * 128 + c[None, :]
    valid = r[:, None] < A * N
    qo = (((r // N)[:, None] * 4 + c[None, :] // 32) * N + (r % N)[:, None]) * 32 + c[
        None, :
    ] % 32
    q = gl.load(DQ + qo, valid, 0).to(gl.float32) * (32**-0.5)
    v = gl.load(DV + qo, valid, 0)
    gl.store(XQ + dst, q, valid)
    gl.store(XV + dst, v, valid)
    rr = gl.program_id(0) * T + gl.arange(0, T, layout=gl.SliceLayout(1, O))
    cc = gl.arange(0, 128, layout=gl.SliceLayout(0, O))
    ko = (((rr // N)[:, None] * 4 + cc[None, :] // 32) * 32 + cc[None, :] % 32) * N + (
        rr % N
    )[:, None]
    k = gl.load(DK + ko, rr[:, None] < A * N, 0)
    gl.store(XK + dst, gl.convert_layout(k, L), valid)


@g.jit
def prepare_backward_tiled(
    P,
    G,
    M,
    DA,
    DB,
    DP,
    DG,
    N: gl.constexpr,
    OUT: gl.constexpr,
    MR: gl.constexpr,
    MC: gl.constexpr,
    I: gl.constexpr,
    J: gl.constexpr,
    C: gl.constexpr,
):
    L: gl.constexpr = gl.BlockedLayout([1, 1, 2], [1, 4, 8], [4, 1, 1], [2, 1, 0])
    LA: gl.constexpr = gl.BlockedLayout([1, 2, 1], [2, 8, 2], [2, 1, 2], [1, 0, 2])
    LB: gl.constexpr = gl.BlockedLayout([2, 1, 1], [8, 2, 2], [1, 2, 2], [0, 1, 2])
    ii = gl.program_id(0) * I + gl.arange(
        0, I, layout=gl.SliceLayout(1, gl.SliceLayout(2, L))
    )
    jj = gl.program_id(1) * J + gl.arange(
        0, J, layout=gl.SliceLayout(0, gl.SliceLayout(2, L))
    )
    cc = gl.program_id(2) * C + gl.arange(
        0, C, layout=gl.SliceLayout(0, gl.SliceLayout(1, L))
    )
    off = (ii[:, None, None] * N + jj[None, :, None]) * 256 + cc[None, None, :]
    valid = (ii[:, None, None] < N) & (jj[None, :, None] < N)
    mask = gl.load(M + ii[:, None, None] * MR + jj[None, :, None] * MC, valid, 0)
    for half in gl.static_range(2):
        if (OUT and half == 0) or (not OUT and half == 1):
            i = gl.program_id(0) * I + gl.arange(
                0, I, layout=gl.SliceLayout(1, gl.SliceLayout(2, LA))
            )
            j = gl.program_id(1) * J + gl.arange(
                0, J, layout=gl.SliceLayout(0, gl.SliceLayout(2, LA))
            )
            c = gl.program_id(2) * C + gl.arange(
                0, C, layout=gl.SliceLayout(0, gl.SliceLayout(1, LA))
            )
            src = c[None, None, :] * N * N + i[:, None, None] * N + j[None, :, None]
        else:
            i = gl.program_id(0) * I + gl.arange(
                0, I, layout=gl.SliceLayout(1, gl.SliceLayout(2, LB))
            )
            j = gl.program_id(1) * J + gl.arange(
                0, J, layout=gl.SliceLayout(0, gl.SliceLayout(2, LB))
            )
            c = gl.program_id(2) * C + gl.arange(
                0, C, layout=gl.SliceLayout(0, gl.SliceLayout(1, LB))
            )
            src = c[None, None, :] * N * N + j[None, :, None] * N + i[:, None, None]
        if half == 0:
            dy = gl.load(DA + src, (i[:, None, None] < N) & (j[None, :, None] < N), 0)
        else:
            dy = gl.load(DB + src, (i[:, None, None] < N) & (j[None, :, None] < N), 0)
        dy = gl.convert_layout(dy, L)
        p = gl.load(P + off + half * 128, valid, 0)
        gate = gl.load(G + off + half * 128, valid, 0).to(gl.float32)
        gate = (
            gl.div_rn(1.0, 1.0 + gl.extra.libdevice.exp(-gate))
            .to(p.dtype)
            .to(gl.float32)
        )
        dy = (dy.to(gl.float32) * mask).to(p.dtype).to(gl.float32)
        dp = dy * gate
        dgate = (dy * p.to(gl.float32)).to(p.dtype).to(gl.float32)
        one_minus = (1.0 - gate).to(p.dtype).to(gl.float32)
        dg = (dgate * one_minus).to(p.dtype).to(gl.float32) * gate
        gl.store(DP + off + half * 128, dp, valid)
        gl.store(DG + off + half * 128, dg, valid)


def prepare_backward(p, gate, mask, da, db, outgoing):
    n = p.shape[1]
    dp, dg = torch.empty_like(p), torch.empty_like(gate)
    i, j, c = 16, 16, 32
    prepare_backward_tiled[(triton.cdiv(n, i), triton.cdiv(n, j), 4)](
        p,
        gate,
        mask,
        da.contiguous(),
        db.contiguous(),
        dp,
        dg,
        n,
        outgoing,
        *mask.stride()[-2:],
        i,
        j,
        c,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return dp, dg


def pack_qkv_backward(dq, dk, dv):
    a, _, n, _ = dq.shape
    outputs = [
        torch.empty((a, n, 128), device=dq.device, dtype=dq.dtype) for _ in range(3)
    ]
    qkv_backward_tiled[(triton.cdiv(a * n, 32),)](
        dq.contiguous(),
        dk.contiguous(),
        dv.contiguous(),
        *outputs,
        a,
        n,
        32,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return tuple(outputs)
