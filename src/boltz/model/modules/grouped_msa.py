"""Grouped inference MSA contractions with record-local, bounded workspaces.

Python builds metadata once and iterates workspace waves, never attention heads
or feature channels. Each wave can contain multiple differently sized records.
Reduction loops inside GPU kernels do not launch work from Python.
"""

import torch
import triton


class MSAPlan:
    def __init__(self, shapes, device, token_budget=1048576, pair_budget=32768, opm_tile_n=128):
        if not shapes or any(s < 1 or n < 1 for s, n in shapes):
            raise ValueError("Grouped MSA requires nonempty records")
        if token_budget < 1 or pair_budget < 1 or opm_tile_n not in (128, 256):
            raise ValueError("Invalid grouped MSA workspace or tile size")
        self.opm_tile_n = opm_tile_n
        self.shapes = tuple(shapes)
        self.m_offsets, self.p_offsets = [0], [0]
        records, softmax, counts = [], [], []
        for record, (s, n) in enumerate(shapes):
            mo, po = self.m_offsets[-1], self.p_offsets[-1]
            records.append((s, n, mo, po))
            softmax.extend((n, po, i) for i in range(n))
            counts.extend((record, i, j) for i in range(triton.cdiv(n, 32)) for j in range(triton.cdiv(n, 32)))
            self.m_offsets.append(mo + s * n)
            self.p_offsets.append(po + n * n)
        self.records = torch.tensor(records, device=device, dtype=torch.int64)
        self.softmax = torch.tensor(softmax, device=device, dtype=torch.int64)
        self.count_tasks = torch.tensor(counts, device=device, dtype=torch.int64)
        self.max_n = max(n for _, n in shapes)
        self.pwa_waves, self.opm_waves = [], []
        # Wave boundaries always fall between full MSA rows / token rows.
        # These are CPU metadata loops, shared by every MSA layer in the stack.
        wave, used, start = [], 0, 0
        for s, n, mo, po in records:
            done = 0
            while done < s:
                take = min(s - done, max(1, (token_budget - used) // n))
                wave.append((take, n, used, po))
                used += take * n
                done += take
                if used + n > token_budget or (done == s and used >= token_budget):
                    self._pwa_wave(wave, start, used, device)
                    start += used
                    wave, used = [], 0
        if wave:
            self._pwa_wave(wave, start, used, device)
        wave, used = [], 0
        for s, n, mo, po in records:
            i = 0
            while i < n:
                take = min(n - i, max(4, ((pair_budget - used) // n // 4) * 4))
                wave.append((s, n, mo, po, i, take, used))
                used += take * n
                i += take
                if used + 4 * n > pair_budget:
                    self._opm_wave(wave, used, device)
                    wave, used = [], 0
        if wave:
            self._opm_wave(wave, used, device)

    def _pwa_wave(self, wave, start, size, device):
        tasks, output = [], []
        for r, (s, n, off, po) in enumerate(wave):
            tasks.extend((r, i, j) for i in range(triton.cdiv(n, 64))
                         for j in range(triton.cdiv(s * 32, 128)))
            output.extend((r, i) for i in range(triton.cdiv(s * n, 32)))
        tensor = lambda x: torch.tensor(x, device=device, dtype=torch.int64)
        self.pwa_waves.append((start, size, tensor(wave), tensor(tasks), tensor(output)))

    def _opm_wave(self, wave, size, device):
        tasks, output = [], []
        for r, (s, n, mo, po, i, count, off) in enumerate(wave):
            tasks.extend((r, x, y) for y in range(triton.cdiv(n, self.opm_tile_n // 32))
                         for x in range(triton.cdiv(count, 4)))
            output.extend((r, x) for x in range(triton.cdiv(count * n, 32)))
        tensor = lambda x: torch.tensor(x, device=device, dtype=torch.int64)
        self.opm_waves.append((size, tensor(wave), tensor(tasks), tensor(output)))


def _check(module, m, plan):
    if module.training or torch.is_grad_enabled() or not m.is_cuda:
        raise RuntimeError("Grouped MSA requires CUDA eval inference")
    if m.ndim != 2 or not m.is_contiguous() or m.shape[0] != plan.m_offsets[-1]:
        raise ValueError("MSA buffer does not match plan")
    if plan.records.device != m.device:
        raise ValueError("MSA plan and values must be on the same CUDA device")
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else m.dtype
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Unsupported grouped MSA dtype")
    return dtype



def _normalize(x, norm, dtype):
    from .grouped_msa_kernels import norm_cast
    out = torch.empty_like(x, dtype=dtype)
    norm_cast[(triton.cdiv(x.shape[0], 16),)](
        x, norm.weight, norm.bias, out, x.shape[0], x.shape[1], norm.eps,
        triton.next_power_of_2(x.shape[1]), num_warps=4)
    return out


def pair_weighted_averaging(module, m, z, mask, plan):
    from .grouped_msa_kernels import pwa_weights, pwa_contract, pwa_output
    dtype = _check(module, m, plan)
    if module.c_h != 32:
        raise ValueError("Grouped PWA requires head width 32")
    if z.shape != (plan.p_offsets[-1], module.c_z) or not z.is_contiguous() or z.device != m.device:
        raise ValueError("Pair buffer does not match MSA plan")
    if mask.shape != z.shape[:1] or not mask.is_contiguous() or mask.device != m.device:
        raise ValueError("Pair mask does not match MSA plan")
    if m.shape[1] != module.c_m:
        raise ValueError("MSA channel width does not match PWA")
    h = module.num_heads
    logits = module.proj_z(module.norm_z(z).to(dtype))
    weights = torch.empty_like(logits, dtype=dtype)
    pwa_weights[(plan.softmax.shape[0], h)](
        logits, mask, weights, plan.softmax, h, module.inf,
        triton.next_power_of_2(plan.max_n), num_warps=4)
    del logits
    result = torch.empty_like(m, dtype=dtype)
    ow = module.proj_o.weight.to(dtype)
    for start, size, segments, tasks, out_tasks in plan.pwa_waves:
        x = _normalize(m[start:start + size], module.norm_m, dtype)
        v = module.proj_m(x)
        g = module.proj_g(x)
        del x
        mixed = torch.empty_like(v)
        pwa_contract[(tasks.shape[0], h)](
            weights, v, g, mixed, segments, tasks, h, num_warps=4)
        del v, g
        pwa_output[(out_tasks.shape[0], triton.cdiv(module.c_m, 64))](
            mixed, ow, result, segments, out_tasks, start, h, module.c_m,
            num_warps=4)
    return result


def outer_product_mean(module, m, mask, plan):
    from .grouped_msa_kernels import opm_contract, opm_output, grouped_counts
    dtype = _check(module, m, plan)
    if module.c_hidden != 32:
        raise ValueError("Grouped OPM requires hidden width 32")
    if mask.shape != m.shape[:1] or not mask.is_contiguous() or mask.device != m.device:
        raise ValueError("MSA mask does not match plan")
    x = _normalize(m, module.norm, dtype)
    typed_mask = mask[:, None].to(dtype)
    a = module.proj_a(x) * typed_mask
    b = module.proj_b(x) * typed_mask
    del x, typed_mask
    counts = torch.empty(plan.p_offsets[-1], device=m.device, dtype=torch.float32)
    grouped_counts[(plan.count_tasks.shape[0],)](mask, counts, plan.records, plan.count_tasks, num_warps=4)
    # BF16 chunked reference adds FP32 bias after rounded partial projections.
    out = torch.empty((plan.p_offsets[-1], module.proj_o.out_features),
                      device=m.device, dtype=torch.float32)
    weight = module.proj_o.weight.to(dtype)
    for size, segments, tasks, out_tasks in plan.opm_waves:
        features = torch.empty((size, 1024), device=m.device, dtype=dtype)
        opm_contract[(tasks.shape[0],)](
            a, b, counts, features, segments, tasks, plan.opm_tile_n, num_warps=8, num_stages=2 if dtype == torch.float32 else 3)
        opm_output[(out_tasks.shape[0], triton.cdiv(out.shape[1], 64))](
            features, weight, module.proj_o.bias, out, segments, out_tasks,
            out.shape[1], num_warps=4)
    return out
