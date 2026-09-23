"""Inference-only binary MSA mask counts without S-by-N-by-N temporaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mask_counts(Mask, Out, S: tl.constexpr, N: tl.constexpr,
                 CHUNKED: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    j = tl.program_id(1) * TILE + tl.arange(0, TILE)
    rows = tl.arange(0, 64)
    total = tl.full((TILE, TILE), 0, tl.float32)
    for start in range(tl.cdiv(S, 64)):
        s = start * 64 + rows
        a = tl.load(Mask + s[None, :] * N + i[:, None],
                    (s[None, :] < S) & (i[:, None] < N), other=0).to(tl.bfloat16)
        b = tl.load(Mask + s[:, None] * N + j[None, :],
                    (s[:, None] < S) & (j[None, :] < N), other=0).to(tl.bfloat16)
        count = tl.dot(a, b)
        if CHUNKED:
            # Match the reference's in-place low-precision addition every 64 rows.
            count = count.to(Out.dtype.element_ty).to(tl.float32)
            total = (total + count).to(Out.dtype.element_ty).to(tl.float32)
        else:
            total += count
    tl.store(Out + i[:, None] * N + j[None, :], tl.maximum(total, 1),
             (i[:, None] < N) & (j[None, :] < N))


def binary_mask_counts(mask, dtype, chunked):
    """Count valid MSA row intersections; mask must contain only zero and one.

    Packed inference obtains binary masks from the feature pipeline. This helper
    intentionally does not accept general fractional attention weights. Shape is
    [1, S, N, 1], matching OuterProductMean after its mask expansion.
    """
    if mask.ndim != 4 or mask.shape[0] != 1 or mask.shape[-1] != 1:
        raise ValueError("Packed MSA mask must have shape [1, S, N, 1]")
    if not mask.is_cuda or mask.shape[1] == 0 or mask.shape[2] == 0:
        raise ValueError("Packed MSA mask requires nonempty CUDA input")
    if dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Unsupported packed MSA mask dtype")
    mask = mask.contiguous()
    s, n = mask.shape[1:3]
    out = torch.empty((1, n, n, 1), device=mask.device, dtype=dtype)
    _mask_counts[(triton.cdiv(n, 32), triton.cdiv(n, 32))](
        mask, out, s, n, chunked, 32, num_warps=4)
    return out
