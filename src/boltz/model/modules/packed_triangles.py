"""Inference-only triangle operations on concatenated record-local pair squares.

Projections and gates run across packed rows. Each contraction/attention launch
contains tiles from every record. No common-length pair tensor is allocated.
The sequential packed backend remains the numerical and performance reference.
"""

import torch


def _check(module, values, masks, layout):
    if module.training or torch.is_grad_enabled():
        raise RuntimeError(
            "Packed Triton triangles require eval() and no_grad()/inference_mode()"
        )
    if not values.is_cuda:
        raise ValueError("Packed Triton triangles require CUDA")
    if values.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"Unsupported packed triangle dtype: {values.dtype}")
    if values.ndim != 2 or values.shape[0] != layout.pair_offsets[-1]:
        raise ValueError("Pair buffer does not match packed layout")
    # Tile indices use int32 even though record offsets/pointers use int64.
    projection_width = getattr(module, "no_heads", 0) * getattr(module, "c_hidden", 0)
    if values.shape[0] * max(values.shape[1], projection_width) >= 2**31:
        raise ValueError(
            "Packed Triton triangles require fewer than 2^31 pair/projection "
            "elements; reduce batch size or select the sequential backend"
        )
    if not values.is_contiguous():
        raise ValueError("Packed Triton triangles require contiguous packed pair rows")
    if masks.shape != values.shape[:1] or masks.device != values.device:
        raise ValueError("Pair mask does not match packed values")
    if layout.lengths_gpu.device != values.device:
        raise ValueError("Packed layout and values must be on the same CUDA device")


def _tasks(layout, kind, block):
    """Cache record/tile descriptors once per layout, reused across the stack."""
    key = (kind, block)
    if key not in layout.kernel_tasks:
        tasks = []
        for record, length in enumerate(layout.lengths):
            count = (length + block - 1) // block
            if kind == "multiply":
                tasks.extend((record, i, j) for i in range(count) for j in range(count))
            elif kind == "attention":
                tasks.extend(
                    (record, i, j) for i in range(length) for j in range(count)
                )
            else:
                tasks.extend(
                    (record, i, 0)
                    for i in range((length * length + block - 1) // block)
                )
        layout.kernel_tasks[key] = torch.tensor(
            tasks,
            dtype=torch.int32,
            device=layout.lengths_gpu.device,
        )
    return layout.kernel_tasks[key]


def triangle_multiply(module, values, masks, layout):
    """Apply outgoing or incoming multiplication to independent packed squares."""
    from boltz.model.layers.triangular_mult import TriangleMultiplicationOutgoing
    from boltz.model.modules.packed_triangle_kernels import (
        multiply,
        project_multiply,
        norm_rows,
        gated_product,
    )

    _check(module, values, masks, layout)
    dtype = (
        torch.get_autocast_dtype("cuda")
        if torch.is_autocast_enabled("cuda")
        else values.dtype
    )
    x = norm_rows(values, module.norm_in, dtype)
    a, b = project_multiply(
        x, module, masks.contiguous(), layout, _tasks(layout, "project", 64)
    )
    out_gate = module.g_out(x)
    copy_tasks = _tasks(layout, "copy", 128)
    # Keep the projection dtype (BF16/FP16 products accumulate in FP32).
    # For FP32 inputs the kernel explicitly requests IEEE multiplication.
    outgoing = isinstance(module, TriangleMultiplicationOutgoing)
    normalized = multiply(
        a,
        b,
        layout,
        _tasks(layout, "multiply", 64 if outgoing else 128),
        copy_tasks,
        outgoing=outgoing,
        norm=module.norm_out,
        dtype=dtype,
    )
    return gated_product(module.p_out(normalized), out_gate)


def triangle_attention(module, values, masks, layout):
    """Apply starting/ending triangle attention without cubic score storage."""
    from boltz.model.modules.packed_triangle_kernels import (
        attention,
        norm_rows,
        gated_product,
    )

    _check(module, values, masks, layout)
    dtype = (
        torch.get_autocast_dtype("cuda")
        if torch.is_autocast_enabled("cuda")
        else values.dtype
    )
    x = norm_rows(values, module.layer_norm, dtype)
    mha = module.mha
    heads, dim = mha.no_heads, mha.c_hidden
    projections = [mha.linear_q, mha.linear_k, mha.linear_v]
    if mha.linear_g is not None:
        projections.append(mha.linear_g)
    weights = torch.cat([linear.weight for linear in projections])
    projected = torch.nn.functional.linear(x, weights).split(heads * dim, -1)
    q, k, v = [part.reshape(-1, heads, dim) for part in projected[:3]]
    bias = module.linear(x)
    out = attention(
        q,
        k,
        v,
        bias,
        masks.contiguous(),
        layout,
        _tasks(layout, "attention", 64),
        starting=module.starting,
        inf=module.inf,
    )
    # Output is already in original pair order, including ending-node attention.
    out = out.reshape(-1, heads * dim)
    if mha.linear_g is not None:
        out = gated_product(out, projected[3])
    return mha.linear_o(out)
