"""Measure complete packed MSA operations against per-record Triton baseline."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from boltz.model.layers.outer_product_mean import OuterProductMean
from boltz.model.layers.pair_averaging import PairWeightedAveraging
from boltz.model.modules.grouped_msa import MSAPlan, pair_weighted_averaging, outer_product_mean


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="8192x139,65x385,1x412,1025x79")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--token-budget", type=int, default=1048576)
    p.add_argument("--pair-budget", type=int, default=32768)
    args = p.parse_args()
    shapes = [tuple(map(int, s.split("x"))) for s in args.shapes.split(",")]
    torch.manual_seed(42)
    plan = MSAPlan(shapes, "cuda", args.token_budget, args.pair_budget)
    m = torch.randn(plan.m_offsets[-1], 64, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(plan.p_offsets[-1], 128, device="cuda")
    mask = (torch.rand(m.shape[0], device="cuda") > .2).float()
    pair_mask = torch.ones(z.shape[0], device="cuda")
    results = []
    for kind, module, grouped in [("pwa", PairWeightedAveraging(64, 128, 32, 8), pair_weighted_averaging),
                                 ("opm", OuterProductMean(64, 32, 128), outer_product_mean)]:
        module = module.cuda().eval()
        module.packed_pair_backend = "triton"
        module.proj_o.weight.normal_(0, .03)
        def call(backend):
            if backend == "grouped":
                return grouped(module, m, z, pair_mask, plan) if kind == "pwa" else grouped(module, m, mask, plan)
            output = []
            for r, (s, n) in enumerate(shapes):
                mo, me = plan.m_offsets[r:r + 2]
                po, pe = plan.p_offsets[r:r + 2]
                x = m[mo:me].view(1, s, n, 64)
                if kind == "pwa":
                    v = module(x, z[po:pe].view(1, n, n, 128), pair_mask[po:pe].view(1, n, n), n > 384)
                else:
                    v = module(x, mask[mo:me].view(1, s, n), 4 if n > 384 else None)
                output.append(v.reshape(-1, v.shape[-1]))
            return torch.cat(output)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference = call("reference")
            actual = call("grouped")
            error = (actual.float() - reference.float())
            metrics = {"relative_l2": float(error.norm() / reference.float().norm()),
                       "max_abs": float(error.abs().max()), "finite": bool(torch.isfinite(actual).all())}
            del actual, reference, error
            for backend in ("reference", "grouped"):
                for _ in range(2):
                    call(backend)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    out = call(backend)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)
                    del out
                row = {"operation": kind, "backend": backend, "shapes": shapes, "seconds": times,
                       "median_seconds": statistics.median(times), "comparison": metrics,
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
                if args.profile:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                        call(backend)
                        torch.cuda.synchronize()
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.with_name(args.output.stem + f"_{kind}_{backend}.txt").write_text(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
                results.append(row)
                print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
