"""Bounded warm CUDA benchmark of packed triangle backends (no checkpoint).

Run with PYTHONPATH=src python scripts/benchmark_packed_triangles.py.
Measures projections, gates, metadata lookup and contractions together. Compilation
and metadata creation are warmed first. This is not a full-model speedup claim.
"""

import argparse
import json
import time

import torch

from boltz.model.layers.triangular_mult import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from boltz.model.layers.triangular_attention.attention import TriangleAttention
from boltz.model.modules.packed import Layout, pair_operation


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[127, 193, 257, 321])
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.manual_seed(42)
    layout = Layout(args.lengths, max(args.lengths), torch.device("cuda"))
    z = torch.randn(layout.pair_offsets[-1], args.channels, device="cuda")
    mask = torch.ones(z.shape[0], device="cuda")
    records = []
    for name, module in [
        ("outgoing", TriangleMultiplicationOutgoing(args.channels)),
        ("incoming", TriangleMultiplicationIncoming(args.channels)),
        ("starting", TriangleAttention(args.channels, 32, 4, starting=True)),
        ("ending", TriangleAttention(args.channels, 32, 4, starting=False)),
    ]:
        module = module.cuda().eval()
        for param in module.parameters():
            if param.ndim >= 2:
                param.normal_(0, 0.05)
        outputs = {}
        for backend in ("sequential", "triton"):
            module.packed_pair_backend = backend
            with torch.autocast("cuda", dtype=torch.bfloat16):

                def call():
                    return pair_operation(
                        module,
                        z,
                        mask,
                        layout,
                        True,
                        isinstance(module, TriangleAttention),
                    )

                for _ in range(2):
                    call()
                outputs[backend] = call().float().cpu()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                for _ in range(args.repeats):
                    call()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000 / args.repeats
                row = {
                    "operation": name,
                    "backend": backend,
                    "ms": elapsed,
                    "peak_GiB": torch.cuda.max_memory_allocated() / 2**30,
                }
                records.append(row)
                print(json.dumps(row), flush=True)
        diff = (outputs["triton"].float() - outputs["sequential"].float()).abs()
        print(
            json.dumps(
                {
                    "operation": name,
                    "max_abs_error": diff.max().item(),
                    "mean_abs_error": diff.mean().item(),
                }
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "lengths": args.lengths,
                "results": records,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
