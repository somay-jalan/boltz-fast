"""Warm operator/kernel profile of existing and grouped packed triangles."""

import argparse
import json
from pathlib import Path

import torch

from boltz.model.layers.triangular_mult import TriangleMultiplicationOutgoing
from boltz.model.layers.triangular_attention.attention import TriangleAttention
from boltz.model.modules.packed import Layout, pair_operation


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int, default=[744, 812, 286, 761])
    parser.add_argument("--out", type=Path, default=Path("profiles"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    layout = Layout(args.lengths, max(args.lengths), torch.device("cuda"))
    z = torch.randn(layout.pair_offsets[-1], 128, device="cuda")
    mask = torch.ones(z.shape[0], device="cuda")
    for kind, module in [
        ("multiply", TriangleMultiplicationOutgoing(128)),
        ("attention", TriangleAttention(128, 32, 4)),
    ]:
        module = module.cuda().eval()
        for param in module.parameters():
            if param.ndim > 1:
                param.normal_(0, 0.05)
        for backend in ["sequential", "triton"]:
            module.packed_pair_backend = backend
            with torch.autocast("cuda", dtype=torch.bfloat16):

                def call():
                    return pair_operation(
                        module, z, mask, layout, True, kind == "attention"
                    )

                for _ in range(3):
                    call()
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    for _ in range(3):
                        with torch.profiler.record_function(f"{kind}_{backend}"):
                            call()
                    torch.cuda.synchronize()
            print(kind, backend, flush=True)
            print(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25),
                flush=True,
            )
            prof.export_chrome_trace(str(args.out / f"{kind}_{backend}.json"))
            events = sorted(
                (e for e in prof.key_averages() if e.self_device_time_total),
                key=lambda e: e.self_device_time_total,
                reverse=True,
            )
            (args.out / f"{kind}_{backend}_summary.json").write_text(
                json.dumps(
                    [
                        {
                            "name": e.key,
                            "count": e.count,
                            "device_us": e.self_device_time_total,
                        }
                        for e in events
                    ],
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
