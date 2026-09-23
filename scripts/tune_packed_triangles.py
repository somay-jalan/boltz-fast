"""Measure contraction tile choices without projection/normalization overhead."""

import json
import statistics
import torch
from boltz.model.modules.packed import Layout
from boltz.model.modules.packed_triangles import _tasks
from boltz.model.modules.packed_triangle_kernels import _multiply, _attention


@torch.inference_mode()
def main():
    lengths = [744, 812, 286, 761]
    layout = Layout(lengths, max(lengths), torch.device("cuda"))
    pairs, channels = layout.pair_offsets[-1], 128
    a = torch.randn(pairs, channels, dtype=torch.bfloat16, device="cuda")
    b = torch.randn_like(a)
    out = torch.empty_like(a, dtype=torch.float32)

    def measure(fn):
        fn()
        fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return statistics.median(samples)

    for outgoing in (True, False):
        for block in (32, 64, 128):
            tasks = _tasks(layout, "multiply", block)
            for bk in (32, 64, 128):
                for warps in (4, 8):
                    try:

                        def call():
                            _multiply[(tasks.shape[0], channels)](
                                a,
                                b,
                                out,
                                tasks,
                                layout.lengths_gpu,
                                layout.pair_offsets_gpu,
                                channels,
                                outgoing,
                                block,
                                bk,
                                num_warps=warps,
                                num_stages=3,
                            )

                        print(
                            json.dumps(
                                dict(
                                    op="mul",
                                    outgoing=outgoing,
                                    block=block,
                                    bk=bk,
                                    warps=warps,
                                    ms=measure(call),
                                )
                            ),
                            flush=True,
                        )
                    except Exception as exc:
                        print(
                            json.dumps(
                                dict(error=str(exc), block=block, bk=bk, warps=warps)
                            ),
                            flush=True,
                        )
    del a, b, out
    heads, dim = 4, 32
    q = torch.randn(pairs, heads, dim, dtype=torch.bfloat16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    bias = torch.randn(pairs, heads, dtype=torch.bfloat16, device="cuda")
    mask = torch.ones(pairs, device="cuda")
    out = torch.empty_like(q)
    for starting in (True, False):
        for bm in (16, 32, 64, 128):
            tasks = _tasks(layout, "attention", bm)
            for bn in (32, 64, 128):
                for warps in (4, 8):
                    try:

                        def call():
                            _attention[(tasks.shape[0], heads)](
                                q,
                                k,
                                v,
                                bias,
                                mask,
                                out,
                                tasks,
                                layout.lengths_gpu,
                                layout.pair_offsets_gpu,
                                heads,
                                dim,
                                starting,
                                *[t.stride(0) for t in (q, k, v)],
                                1e9,
                                bm,
                                bn,
                                dim,
                                num_warps=warps,
                                num_stages=2,
                            )

                        print(
                            json.dumps(
                                dict(
                                    op="attn",
                                    starting=starting,
                                    bm=bm,
                                    bn=bn,
                                    warps=warps,
                                    ms=measure(call),
                                )
                            ),
                            flush=True,
                        )
                    except Exception as exc:
                        print(
                            json.dumps(dict(error=str(exc), bm=bm, bn=bn, warps=warps)),
                            flush=True,
                        )


if __name__ == "__main__":
    main()
