"""First-order derivatives for the packed backend."""

import torch
from torch.autograd.function import once_differentiable
from . import _pointwise_kernels as kernels
from . import _layout_kernels


def needed(*xs):
    return torch.is_grad_enabled() and any(x.requires_grad for x in xs)


def norm_backward(x, w, b, grad, eps):
    with torch.autocast("cuda", enabled=False):
        xf, wf, bf = x.float(), w.float(), b.float()
        _, mean, rstd = torch.ops.aten.native_layer_norm.default(
            xf, [x.shape[-1]], wf, bf, eps
        )
        dx, dw, db = torch.ops.aten.native_layer_norm_backward.default(
            grad.float().contiguous(),
            xf,
            [x.shape[-1]],
            mean,
            rstd,
            wf,
            bf,
            [True, True, True],
        )
    return dx.to(x.dtype), dw.to(w.dtype), db.to(b.dtype)


class RoundedNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, eps):
        from boltz.model.modules.packed_pairformer._math_kernels import norm

        ctx.save_for_backward(x, w, b)
        ctx.eps = eps
        return norm(x, w, b, eps, output_dtype=torch.bfloat16).float()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        return (*norm_backward(*ctx.saved_tensors, grad, ctx.eps), None)


class Silu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return kernels.silu_mul(x, y)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, y = ctx.saved_tensors
        return torch.ops.aten.silu_backward.default(
            grad * y, x
        ), grad * torch.nn.functional.silu(x)


class Prepare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, gate, mask, outgoing):
        ctx.save_for_backward(p, gate, mask)
        ctx.outgoing = outgoing
        return _layout_kernels.prepare(p, gate, mask, outgoing)

    @staticmethod
    @once_differentiable
    def backward(ctx, da, db):
        p, gate, mask = ctx.saved_tensors
        return (
            *_layout_kernels.prepare_backward(p, gate, mask, da, db, ctx.outgoing),
            None,
            None,
        )


class PackQKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        return _layout_kernels.pack_qkv(q, k, v)

    @staticmethod
    @once_differentiable
    def backward(ctx, dq, dk, dv):
        return _layout_kernels.pack_qkv_backward(dq, dk, dv)
