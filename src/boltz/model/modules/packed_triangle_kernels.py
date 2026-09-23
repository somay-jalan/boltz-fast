"""Grouped triangle contractions and segmented triangle attention in Triton.

The grid spans actual record-local tiles, not a padded maximum-length square.
All pointers are bounded by the current record. Multiplication uses record-local
channel-major workspaces for coalesced matrix reads, without length padding.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rows(
    X,
    W,
    B,
    Out,
    ROWS: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    BC: tl.constexpr,
    BR: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    channels = tl.arange(0, BC)
    x = tl.load(
        X + rows[:, None] * C + channels[None, :],
        mask=(rows[:, None] < ROWS) & (channels[None, :] < C),
        other=0,
    ).to(tl.float32)
    mean = tl.sum(x, 1) / C
    centered = tl.where(channels[None, :] < C, x - mean[:, None], 0)
    var = tl.sum(centered * centered, 1) / C
    w = tl.load(W + channels, mask=channels < C, other=0).to(tl.float32)
    b = tl.load(B + channels, mask=channels < C, other=0).to(tl.float32)
    y = centered * tl.rsqrt(var[:, None] + EPS) * w[None, :] + b[None, :]
    tl.store(
        Out + rows[:, None] * C + channels[None, :],
        y,
        mask=(rows[:, None] < ROWS) & (channels[None, :] < C),
    )


def norm_rows(x, module, dtype):
    out = torch.empty(x.shape, device=x.device, dtype=dtype)
    weight, bias = module.weight, module.bias
    if x.dtype == torch.bfloat16 and not isinstance(module, torch.nn.LayerNorm):
        weight, bias = weight.to(x.dtype), bias.to(x.dtype)
    _norm_rows[(triton.cdiv(x.shape[0], 16),)](
        x,
        weight,
        bias,
        out,
        x.shape[0],
        x.shape[1],
        module.eps,
        triton.next_power_of_2(x.shape[1]),
        16,
        num_warps=4,
    )
    return out


@triton.jit
def _gated_product(
    X,
    G,
    Out,
    ROWS: tl.constexpr,
    C: tl.constexpr,
    XS: tl.constexpr,
    GS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // C, index % C
    x = tl.load(X + row * XS + col, mask=row < ROWS, other=0)
    g = tl.load(G + row * GS + col, mask=row < ROWS, other=0)
    gate = tl.sigmoid(g.to(tl.float32)).to(g.dtype)
    tl.store(Out + index, x.to(tl.float32) * gate.to(tl.float32), mask=row < ROWS)


def gated_product(x, gate):
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _gated_product[(triton.cdiv(x.numel(), 1024),)](
        x,
        gate,
        out,
        x.shape[0],
        x.shape[1],
        x.stride(0),
        gate.stride(0),
        1024,
    )
    return out


@triton.jit
def _project_multiply(
    X,
    PW,
    GW,
    Mask,
    A,
    B,
    Tasks,
    Lengths,
    Offsets,
    C: tl.constexpr,
    BP: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    task, channel_tile = tl.program_id(0), tl.program_id(1)
    record = tl.load(Tasks + task * 3)
    tile = tl.load(Tasks + task * 3 + 1)
    n = tl.load(Lengths + record)
    offset = tl.load(Offsets + record).to(tl.int64)
    pairs = tile * BP + tl.arange(0, BP)
    channels = channel_tile * BC + tl.arange(0, BC)
    inner = tl.arange(0, BK)
    p_acc = tl.full((BP, BC), 0, tl.float32)
    g_acc = tl.full((BP, BC), 0, tl.float32)
    for start in range(tl.cdiv(C, BK)):
        c = start * BK + inner
        x = tl.load(
            X + (offset + pairs[:, None]) * C + c[None, :],
            mask=(pairs[:, None] < n * n) & (c[None, :] < C),
            other=0,
        )
        pw = tl.load(
            PW + channels[None, :] * C + c[:, None],
            mask=(channels[None, :] < 2 * C) & (c[:, None] < C),
            other=0,
        )
        gw = tl.load(
            GW + channels[None, :] * C + c[:, None],
            mask=(channels[None, :] < 2 * C) & (c[:, None] < C),
            other=0,
        )
        p_acc += tl.dot(x, pw, input_precision="ieee")
        g_acc += tl.dot(x, gw, input_precision="ieee")
    projected = p_acc.to(X.dtype.element_ty)
    gate = tl.sigmoid(g_acc.to(X.dtype.element_ty).to(tl.float32)).to(
        X.dtype.element_ty
    )
    value = (projected.to(tl.float32) * gate.to(tl.float32)).to(X.dtype.element_ty)
    mask = tl.load(Mask + offset + pairs, mask=pairs < n * n, other=0)
    value *= mask[:, None].to(value.dtype)
    dest = offset * C + (channels[None, :] % C) * (n * n) + pairs[:, None]
    valid = (pairs[:, None] < n * n) & (channels[None, :] < 2 * C)
    tl.store(A + dest, value, mask=valid & (channels[None, :] < C))
    tl.store(B + dest, value, mask=valid & (channels[None, :] >= C))


def project_multiply(x, module, masks, layout, tasks):
    channels = x.shape[1]
    a, b = torch.empty_like(x), torch.empty_like(x)
    pw, gw = module.p_in.weight.to(x.dtype), module.g_in.weight.to(x.dtype)
    _project_multiply[(tasks.shape[0], triton.cdiv(2 * channels, 32))](
        x,
        pw,
        gw,
        masks,
        a,
        b,
        tasks,
        layout.lengths_gpu,
        layout.pair_offsets_gpu,
        channels,
        64,
        32,
        32,
        num_warps=4,
    )
    return a, b


@triton.jit
def _restore_norm(
    Source,
    W,
    B,
    Out,
    Tasks,
    Lengths,
    Offsets,
    C: tl.constexpr,
    EPS: tl.constexpr,
    BP: tl.constexpr,
    BC: tl.constexpr,
):
    task = tl.program_id(0)
    record = tl.load(Tasks + task * 3)
    tile = tl.load(Tasks + task * 3 + 1)
    n = tl.load(Lengths + record)
    offset = tl.load(Offsets + record).to(tl.int64)
    pairs = tile * BP + tl.arange(0, BP)
    channels = tl.arange(0, BC)
    valid = (pairs[:, None] < n * n) & (channels[None, :] < C)
    source = offset * C + channels[None, :] * (n * n) + pairs[:, None]
    x = tl.load(Source + source, mask=valid, other=0).to(tl.float32)
    mean = tl.sum(x, 1) / C
    centered = tl.where(channels[None, :] < C, x - mean[:, None], 0)
    var = tl.sum(centered * centered, 1) / C
    w = tl.load(W + channels, mask=channels < C, other=0).to(tl.float32)
    b = tl.load(B + channels, mask=channels < C, other=0).to(tl.float32)
    y = centered * tl.rsqrt(var[:, None] + EPS) * w[None, :] + b[None, :]
    tl.store(Out + (offset + pairs[:, None]) * C + channels[None, :], y, mask=valid)


@triton.jit
def _multiply(
    A,
    B,
    Out,
    Tasks,
    Lengths,
    Offsets,
    CHANNELS: tl.constexpr,
    OUTGOING: tl.constexpr,
    BLOCK: tl.constexpr,
    BK: tl.constexpr,
):
    task, channel = tl.program_id(0), tl.program_id(1)
    record = tl.load(Tasks + task * 3)
    ti = tl.load(Tasks + task * 3 + 1)
    tj = tl.load(Tasks + task * 3 + 2)
    n = tl.load(Lengths + record)
    offset = tl.load(Offsets + record).to(tl.int64)
    base = offset * CHANNELS + channel * (n * n)
    i = ti * BLOCK + tl.arange(0, BLOCK)
    j = tj * BLOCK + tl.arange(0, BLOCK)
    k0 = tl.arange(0, BK)
    acc = tl.full((BLOCK, BLOCK), 0, tl.float32)
    for start in range(tl.cdiv(n, BK)):
        k = start * BK + k0
        if OUTGOING:
            ai = base + i[:, None] * n + k[None, :]
            bi = base + j[None, :] * n + k[:, None]
        else:
            ai = base + k[None, :] * n + i[:, None]
            bi = base + k[:, None] * n + j[None, :]
        av = tl.load(A + ai, mask=(i[:, None] < n) & (k[None, :] < n), other=0)
        bv = tl.load(B + bi, mask=(k[:, None] < n) & (j[None, :] < n), other=0)
        acc += tl.dot(av, bv, input_precision="ieee")
    oi = base + i[:, None] * n + j[None, :]
    tl.store(Out + oi, acc, mask=(i[:, None] < n) & (j[None, :] < n))


def multiply(a, b, layout, tasks, copy_tasks, outgoing, norm, dtype):
    workspace = torch.empty(a.shape, dtype=a.dtype, device=a.device)
    _multiply[(tasks.shape[0], a.shape[1])](
        a,
        b,
        workspace,
        tasks,
        layout.lengths_gpu,
        layout.pair_offsets_gpu,
        a.shape[1],
        outgoing,
        64 if outgoing else 128,
        32,
        num_warps=4,
    )
    out = torch.empty(a.shape, dtype=dtype, device=a.device)
    _restore_norm[(copy_tasks.shape[0],)](
        workspace,
        norm.weight,
        norm.bias,
        out,
        copy_tasks,
        layout.lengths_gpu,
        layout.pair_offsets_gpu,
        a.shape[1],
        norm.eps,
        128,
        triton.next_power_of_2(a.shape[1]),
        num_warps=8,
    )
    return out


@triton.jit
def _attention(
    Q,
    K,
    V,
    Bias,
    Mask,
    Out,
    Tasks,
    Lengths,
    Offsets,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    STARTING: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    INF: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    task, head = tl.program_id(0), tl.program_id(1)
    record = tl.load(Tasks + task * 3)
    anchor = tl.load(Tasks + task * 3 + 1)
    tile = tl.load(Tasks + task * 3 + 2)
    n = tl.load(Lengths + record)
    offset = tl.load(Offsets + record).to(tl.int64)
    j = tile * BM + tl.arange(0, BM)
    d = tl.arange(0, BD)
    if STARTING:
        query_pair = offset + anchor * n + j
    else:
        query_pair = offset + j * n + anchor
    q = tl.load(
        Q + query_pair[:, None] * QS + head * DIM + d[None, :],
        mask=(j[:, None] < n) & (d[None, :] < DIM),
        other=0,
    )
    maximum = tl.full((BM,), float("-inf"), tl.float32)
    denominator = tl.full((BM,), 0, tl.float32)
    accumulator = tl.full((BM, BD), 0, tl.float32)
    for start in range(tl.cdiv(n, BN)):
        k_idx = start * BN + tl.arange(0, BN)
        if STARTING:
            key_pair = offset + anchor * n + k_idx
            bias_pair = offset + j[:, None] * n + k_idx[None, :]
        else:
            key_pair = offset + k_idx * n + anchor
            bias_pair = offset + k_idx[None, :] * n + j[:, None]
        key = tl.load(
            K + key_pair[None, :] * KS + head * DIM + d[:, None],
            mask=(k_idx[None, :] < n) & (d[:, None] < DIM),
            other=0,
        )
        scores = tl.dot(q, key, input_precision="ieee") * (DIM**-0.5)
        bias = tl.load(
            Bias + bias_pair * HEADS + head,
            mask=(j[:, None] < n) & (k_idx[None, :] < n),
            other=0,
        )
        valid = tl.load(Mask + key_pair, mask=k_idx < n, other=0).to(tl.float32)
        # Match the reference's finite mask penalty for real masked positions.
        # Only nonexistent tail positions get -inf (and never enter softmax).
        scores += bias.to(tl.float32) + (valid[None, :] - 1) * INF
        scores = tl.where(k_idx[None, :] < n, scores, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(scores, axis=1))
        correction = tl.exp(maximum - next_max)
        weights = tl.exp(scores - next_max[:, None])
        denominator = denominator * correction + tl.sum(weights, axis=1)
        value = tl.load(
            V + key_pair[:, None] * VS + head * DIM + d[None, :],
            mask=(k_idx[:, None] < n) & (d[None, :] < DIM),
            other=0,
        )
        accumulator = accumulator * correction[:, None]
        accumulator += tl.dot(weights.to(value.dtype), value, input_precision="ieee")
        maximum = next_max
    result = accumulator / denominator[:, None]
    tl.store(
        Out + (query_pair[:, None] * HEADS + head) * DIM + d[None, :],
        result,
        mask=(j[:, None] < n) & (d[None, :] < DIM),
    )


def attention(q, k, v, bias, mask, layout, tasks, starting, inf):
    out = torch.empty_like(v)
    heads, dim = q.shape[1:]
    _attention[(tasks.shape[0], heads)](
        q,
        k,
        v,
        bias,
        mask,
        out,
        tasks,
        layout.lengths_gpu,
        layout.pair_offsets_gpu,
        heads,
        dim,
        starting,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        inf,
        64,
        32,
        max(16, triton.next_power_of_2(dim)),
        num_warps=4,
    )
    return out
