"""Tiled record-local MSA kernels. All offsets use 64-bit descriptors."""
import triton
import triton.language as tl


@triton.jit
def pwa_weights(L, Mask, W, Tasks, H: tl.constexpr, INF: tl.constexpr, BN: tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
    n = tl.load(Tasks + t * 3)
    po = tl.load(Tasks + t * 3 + 1)
    i = tl.load(Tasks + t * 3 + 2)
    j = tl.arange(0, BN)
    p = po + i * n + j
    b = tl.load(L + p * H + h, j < n, 0).to(tl.float32)
    mask = tl.load(Mask + p, j < n, 0).to(tl.float32)
    b = tl.where(j < n, b + (1 - mask) * -INF, -float("inf"))
    e = tl.exp(b - tl.max(b, 0))
    tl.store(W + po * H + h * n * n + i * n + j, e / tl.sum(e, 0), j < n)


@triton.jit
def pwa_contract(W, V, G, O, Seg, Tasks, H: tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
    r = tl.load(Tasks + t * 3)
    ii = tl.load(Tasks + t * 3 + 1)
    jj = tl.load(Tasks + t * 3 + 2)
    s = tl.load(Seg + r * 4)
    n = tl.load(Seg + r * 4 + 1)
    mo = tl.load(Seg + r * 4 + 2)
    po = tl.load(Seg + r * 4 + 3)
    i = ii * 64 + tl.arange(0, 64)
    c = jj * 128 + tl.arange(0, 128)
    row, channel = c // 32, c % 32
    k = tl.arange(0, 32)
    acc = tl.full((64, 128), 0, tl.float32)
    for base in range(tl.cdiv(n, 32)):
        j = base * 32 + k
        a = tl.load(W + po * H + h * n * n + i[:, None] * n + j[None, :],
                    (i[:, None] < n) & (j[None, :] < n), 0)
        b = tl.load(V + (mo + row[None, :] * n + j[:, None]) * (H * 32) + h * 32 + channel[None, :],
                    (j[:, None] < n) & (row[None, :] < s), 0)
        acc += tl.dot(a, b, input_precision="ieee")
    offsets = (mo + row[None, :] * n + i[:, None]) * (H * 32) + h * 32 + channel[None, :]
    valid = (i[:, None] < n) & (row[None, :] < s)
    gate = tl.sigmoid(tl.load(G + offsets, valid, 0).to(tl.float32)).to(V.dtype.element_ty).to(tl.float32)
    # einsum materializes in projection dtype before gating.
    result = acc.to(V.dtype.element_ty).to(tl.float32) * gate
    tl.store(O + offsets, result, valid)


@triton.jit
def pwa_output(X, W, O, Seg, Tasks, START, H: tl.constexpr, CM: tl.constexpr):
    t = tl.program_id(0)
    r = tl.load(Tasks + t * 2)
    block = tl.load(Tasks + t * 2 + 1)
    s = tl.load(Seg + r * 4)
    n = tl.load(Seg + r * 4 + 1)
    mo = tl.load(Seg + r * 4 + 2)
    rows = block * 32 + tl.arange(0, 32)
    cols = tl.program_id(1) * 64 + tl.arange(0, 64)
    k = tl.arange(0, 32)
    acc = tl.full((32, 64), 0, tl.float32)
    for h in range(H):
        x = tl.load(X + (mo + rows[:, None]) * H * 32 + h * 32 + k[None, :], rows[:, None] < s * n, 0)
        w = tl.load(W + cols[None, :] * H * 32 + h * 32 + k[:, None], cols[None, :] < CM, 0)
        part = tl.dot(x, w, input_precision="ieee")
        if n > 384:
            acc = (acc + part.to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
        else:
            acc += part
    tl.store(O + (START + mo + rows[:, None]) * CM + cols[None, :], acc,
             (rows[:, None] < s * n) & (cols[None, :] < CM))


@triton.jit
def opm_contract(A, B, Counts, F, Seg, Tasks, BN: tl.constexpr):
    t = tl.program_id(0)
    r = tl.load(Tasks + t * 3)
    it = tl.load(Tasks + t * 3 + 1)
    jt = tl.load(Tasks + t * 3 + 2)
    s = tl.load(Seg + r * 7)
    n = tl.load(Seg + r * 7 + 1)
    mo = tl.load(Seg + r * 7 + 2)
    first = tl.load(Seg + r * 7 + 4)
    count = tl.load(Seg + r * 7 + 5)
    off = tl.load(Seg + r * 7 + 6)
    ic = it * 128 + tl.arange(0, 128)
    jd = jt * BN + tl.arange(0, BN)
    i, c = first + ic // 32, ic % 32
    j, d = jd // 32, jd % 32
    k = tl.arange(0, 64)
    acc = tl.full((128, BN), 0, tl.float32)
    for base in range(tl.cdiv(s, 64)):
        rows = base * 64 + k
        a = tl.load(A + (mo + rows[None, :] * n + i[:, None]) * 32 + c[:, None],
                    (ic[:, None] < count * 32) & (rows[None, :] < s), 0)
        b = tl.load(B + (mo + rows[:, None] * n + j[None, :]) * 32 + d[None, :],
                    (j[None, :] < n) & (rows[:, None] < s), 0)
        acc += tl.dot(a, b, input_precision="ieee")
    po = tl.load(Seg + r * 7 + 3)
    mi = first + it * 4 + tl.arange(0, 4)
    mj = jt * (BN // 32) + tl.arange(0, BN // 32)
    counts = tl.load(Counts + po + mi[:, None] * n + mj[None, :],
                     (mi[:, None] < first + count) & (mj[None, :] < n), 1)
    denom = tl.reshape(tl.broadcast_to(counts[:, None, :, None], (4, 32, BN // 32, 32)), (128, BN))
    normalized = acc.to(A.dtype.element_ty).to(tl.float32) / tl.maximum(denom, 1)
    offset = (off + (i[:, None] - first) * n + j[None, :]) * 1024 + c[:, None] * 32 + d[None, :]
    tl.store(F + offset, normalized, (ic[:, None] < count * 32) & (j[None, :] < n))


@triton.jit
def opm_output(F, W, Bias, O, Seg, Tasks, CO: tl.constexpr):
    t = tl.program_id(0)
    r = tl.load(Tasks + t * 2)
    block = tl.load(Tasks + t * 2 + 1)
    n = tl.load(Seg + r * 7 + 1)
    po = tl.load(Seg + r * 7 + 3)
    first = tl.load(Seg + r * 7 + 4)
    count = tl.load(Seg + r * 7 + 5)
    off = tl.load(Seg + r * 7 + 6)
    rows = block * 32 + tl.arange(0, 32)
    cols = tl.program_id(1) * 64 + tl.arange(0, 64)
    k = tl.arange(0, 128)
    acc = tl.full((32, 64), 0, tl.float32)
    for ch in range(8):
        x = tl.load(F + (off + rows[:, None]) * 1024 + ch * 128 + k[None, :], rows[:, None] < count * n, 0)
        w = tl.load(W + cols[None, :] * 1024 + ch * 128 + k[:, None], cols[None, :] < CO, 0)
        part = tl.dot(x, w, input_precision="ieee")
        if n > 384:
            acc = (acc + part.to(F.dtype.element_ty).to(tl.float32)).to(F.dtype.element_ty).to(tl.float32)
        else:
            acc += part
    bias = tl.load(Bias + cols, cols < CO, 0).to(tl.float32)
    acc += bias[None, :]
    if n <= 384:
        acc = acc.to(F.dtype.element_ty).to(tl.float32)
    tl.store(O + (po + first * n + rows[:, None]) * CO + cols[None, :], acc,
             (rows[:, None] < count * n) & (cols[None, :] < CO))


@triton.jit
def grouped_counts(Mask, Out, Records, Tasks):
    t = tl.program_id(0)
    r = tl.load(Tasks + t * 3)
    it = tl.load(Tasks + t * 3 + 1)
    jt = tl.load(Tasks + t * 3 + 2)
    s = tl.load(Records + r * 4)
    n = tl.load(Records + r * 4 + 1)
    mo = tl.load(Records + r * 4 + 2)
    po = tl.load(Records + r * 4 + 3)
    i = it * 32 + tl.arange(0, 32)
    j = jt * 32 + tl.arange(0, 32)
    k = tl.arange(0, 64)
    acc = tl.full((32, 32), 0, tl.float32)
    for base in range(tl.cdiv(s, 64)):
        rows = base * 64 + k
        a = tl.load(Mask + mo + rows[None, :] * n + i[:, None], (rows[None, :] < s) & (i[:, None] < n), 0).to(tl.bfloat16)
        b = tl.load(Mask + mo + rows[:, None] * n + j[None, :], (rows[:, None] < s) & (j[None, :] < n), 0).to(tl.bfloat16)
        acc += tl.dot(a, b)
    tl.store(Out + po + i[:, None] * n + j[None, :], tl.maximum(acc, 1), (i[:, None] < n) & (j[None, :] < n))


@triton.jit
def norm_cast(X, W, B, Out, ROWS, C: tl.constexpr, EPS: tl.constexpr, BC: tl.constexpr):
    rows = (tl.program_id(0) * 16 + tl.arange(0, 16)).to(tl.int64)
    ch = tl.arange(0, BC)
    x = tl.load(X + rows[:, None] * C + ch[None, :], (rows[:, None] < ROWS) & (ch[None, :] < C), 0).to(tl.float32)
    mean = tl.sum(x, 1) / C
    centered = tl.where(ch[None, :] < C, x - mean[:, None], 0)
    var = tl.sum(centered * centered, 1) / C
    w = tl.load(W + ch, ch < C, 0).to(tl.float32)
    b = tl.load(B + ch, ch < C, 0).to(tl.float32)
    y = centered * tl.rsqrt(var[:, None] + EPS) * w[None, :] + b[None, :]
    tl.store(Out + rows[:, None] * C + ch[None, :], y, (rows[:, None] < ROWS) & (ch[None, :] < C))
