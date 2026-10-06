"""Native-arithmetic normalization, softmax and token attention."""

import torch
from torch.autograd.function import once_differentiable


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps, output_dtype):
        from boltz.model.modules.packed_pairformer._math_kernels import norm

        ctx.save_for_backward(x, weight, bias)
        ctx.eps = eps
        return norm(x, weight, bias, eps, output_dtype=output_dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight, bias = ctx.saved_tensors
        # Native mixed-precision layer norm evaluates in FP32. Recompute its
        # statistics instead of retaining an extra pair-sized forward tensor.
        with torch.autocast("cuda", enabled=False):
            xf, wf, bf = x.float(), weight.float(), bias.float()
            _, mean, rstd = torch.ops.aten.native_layer_norm.default(
                xf, [128], wf, bf, ctx.eps
            )
            dx, dw, db = torch.ops.aten.native_layer_norm_backward.default(
                grad.float().contiguous(),
                xf,
                [128],
                mean,
                rstd,
                wf,
                bf,
                list(ctx.needs_input_grad[:3]),
            )
        return (
            None if dx is None else dx.to(x.dtype),
            None if dw is None else dw.to(weight.dtype),
            None if db is None else db.to(bias.dtype),
            None,
            None,
        )


class _Probabilities(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, mask_bias, triangle_bias):
        from boltz.model.modules.packed_pairformer._math_kernels import probabilities

        out = probabilities(scores, mask_bias, triangle_bias)
        ctx.save_for_backward(out)
        ctx.mask_shape = mask_bias.shape
        ctx.bias_shape = triangle_bias.shape
        ctx.mask_dtype = mask_bias.dtype
        ctx.bias_dtype = triangle_bias.dtype
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (out,) = ctx.saved_tensors
        ds = torch.ops.aten._softmax_backward_data.default(
            grad.contiguous(), out, -1, out.dtype
        )
        return (
            ds,
            (
                ds.sum_to_size(ctx.mask_shape).to(ctx.mask_dtype)
                if ctx.needs_input_grad[1]
                else None
            ),
            (
                ds.sum_to_size(ctx.bias_shape).to(ctx.bias_dtype)
                if ctx.needs_input_grad[2]
                else None
            ),
        )


def _norm(module, x, projection=True):
    # Multiple projection branches must accumulate their gradients in FP32,
    # as they do at the native LayerNorm output. Sharing a BF16 norm tensor
    # would silently round the sum of those gradients.
    dtype = (
        torch.bfloat16 if projection and not torch.is_grad_enabled() else torch.float32
    )
    return _Norm.apply(x, module.weight, module.bias, module.eps, dtype)


def _token_attention(module, s, z, mask):
    # Preserve native FP32 token attention, replacing only its pair LayerNorm.
    b, n, _ = s.shape
    h, d = module.num_heads, module.head_dim
    q = module.proj_q(s).view(b, n, h, d)
    k = module.proj_k(s).view(b, n, h, d)
    v = module.proj_v(s).view(b, n, h, d)
    bias = module.proj_z[1](_norm(module.proj_z[0], z, projection=False))
    bias = bias.permute(0, 3, 1, 2)
    gate = module.proj_g(s).sigmoid()
    scores = torch.einsum("bihd,bjhd->bhij", q, k) / (d**0.5) + bias
    scores = scores + (1 - mask[:, None, None].float()) * -module.inf
    out = torch.einsum("bhij,bjhd->bihd", scores.softmax(-1), v)
    return module.proj_o(gate * out.reshape(b, n, module.c_s))
