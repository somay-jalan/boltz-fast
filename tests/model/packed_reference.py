"""Test-only adapters around original PyTorch operations; no alternative backend."""

import torch


class NativeBatch:
    def __init__(self, model, lengths):
        self.model, self.lengths = model, lengths

    @torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
    def __call__(self, s, z, mask, pair_mask):
        singles, pairs = [], []
        width = s.shape[1]
        for i, n in enumerate(self.lengths):
            si, zi = s[i : i + 1, :n], z[i : i + 1, :n, :n]
            # Native reentrant checkpointing needs a grad-requiring input even
            # when only the weights are trained from cached teacher states.
            if (
                self.model.training
                and self.model.activation_checkpointing
                and not si.requires_grad
                and not zi.requires_grad
            ):
                si = si.detach().requires_grad_()
            so, zo = self.model(
                si,
                zi,
                mask[i : i + 1, :n],
                pair_mask[i : i + 1, :n, :n],
                use_kernels=False,
            )
            singles.append(torch.nn.functional.pad(so, (0, 0, 0, width - n)))
            pairs.append(
                torch.nn.functional.pad(zo, (0, 0, 0, width - n, 0, width - n))
            )
        return torch.cat(singles), torch.cat(pairs)


def prepare(p, gate, mask, outgoing):
    y = ((p * gate.sigmoid()) * mask[..., None]).to(p.dtype)
    a, b = (v.squeeze(0).permute(2, 0, 1) for v in y.chunk(2, -1))
    return (a, b.transpose(-1, -2)) if outgoing else (a.transpose(-1, -2), b)


def pack_qkv(q, k, v):
    q, k, v = [x.reshape(*x.shape[:-1], 4, 32).transpose(1, 2) for x in (q, k, v)]
    return q * 32**-0.5, k.transpose(-1, -2), v
