"""Warm OPM/PWA timing and optional PyTorch operator profiling."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from boltz.model.layers.outer_product_mean import OuterProductMean
from boltz.model.layers.pair_averaging import PairWeightedAveraging


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=8192)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    m = torch.randn(1, args.depth, args.tokens, 64, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(m.shape[:3], device="cuda") > 0.2).float()
    z = torch.randn(1, args.tokens, args.tokens, 128, device="cuda")
    pair_mask = torch.ones(z.shape[:3], device="cuda")
    rows = []
    for kind, module in [("opm", OuterProductMean(64, 32, 128)),
                         ("pwa", PairWeightedAveraging(64, 128, 32, 8))]:
        module = module.cuda().eval()
        module.proj_o.weight.normal_(0, 0.03)
        for backend in ["sequential", "triton"]:
            module.packed_pair_backend = backend
            def call():
                if kind == "opm":
                    return module(m, mask, 4 if args.tokens > 384 else None)
                return module(m, z, pair_mask, args.tokens > 384)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(2):
                    output = call()
                torch.cuda.synchronize()
                if backend == "sequential":
                    reference = output.clone()
                else:
                    torch.testing.assert_close(output, reference, rtol=0, atol=0)
                del output
                torch.cuda.reset_peak_memory_stats()
                measurements = []
                for _ in range(args.repeats):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    output = call()
                    torch.cuda.synchronize()
                    measurements.append(time.perf_counter() - t)
                    del output
                row = {"operation": kind, "backend": backend,
                       "depth": args.depth, "tokens": args.tokens,
                       "seconds": measurements,
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
                rows.append(row)
                print(json.dumps(row), flush=True)
                if args.profile:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        call()
                        torch.cuda.synchronize()
                    (args.output / f"{kind}_{backend}.txt").write_text(
                        prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
            gc.collect()
    (args.output / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
