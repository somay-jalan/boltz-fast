"""Measure warmed CUDA-graph inference and training for the supported packed backend."""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
import triton

from boltz.model.layers.pairformer import PairformerModule
from boltz.model.modules.packed_pairformer import (
    PackedInferenceGraph,
    PackedTrainingGraph,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[73, 97])
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--seed", type=int, default=321)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    state = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )["state_dict"]
    model = (
        PairformerModule(
            384,
            128,
            64,
            num_heads=16,
            dropout=0,
            v2=True,
            activation_checkpointing=True,
        )
        .cuda()
        .float()
    )
    model.load_state_dict(
        {
            k.removeprefix("pairformer_module."): v
            for k, v in state.items()
            if k.startswith("pairformer_module.")
        },
        strict=True,
    )
    lengths = (193, 193, 193, 167) if args.real_dir else tuple(args.lengths)
    width, batch = max(lengths), len(lengths)
    s = torch.randn(batch, width, 384, device="cuda")
    z = torch.randn(batch, width, width, 128, device="cuda")
    mask = (
        torch.arange(width, device="cuda")[None]
        < torch.tensor(lengths, device="cuda")[:, None]
    ).float()
    mask[:, min(2, width - 1)] = 0
    pair_mask = mask[:, :, None] * mask[:, None, :]
    if args.real_dir:
        for i, (target, n) in enumerate(zip(("8K69", "8VR3", "9K7M", "8UDS"), lengths)):
            data = torch.load(
                args.real_dir / (target + ".input.pt"), weights_only=False
            )
            s[i, :n], z[i, :n, :n] = data["s"], data["z"]
            mask[i, :n], pair_mask[i, :n, :n] = data["mask"], data["pair_mask"]
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        triton=triton.__version__,
        lengths=lengths,
        seed=args.seed,
        backend="packed",
        checkpointing_training=True,
        protocol="Median CUDA events; includes replay input/output device copies; excludes compilation/capture, optimizer and data loading; backward includes checkpoint recomputation.",
    )
    model.eval()
    inputs = (s, z, mask, pair_mask)
    start = time.perf_counter()
    graph = PackedInferenceGraph(model, inputs, lengths)
    report["inference_capture_seconds"] = time.perf_counter() - start
    samples = []
    with torch.no_grad():
        for i in range(18):
            torch.cuda.synchronize()
            begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            begin.record()
            outputs = graph(*inputs)
            end.record()
            end.synchronize()
            if i >= 3:
                samples.append(begin.elapsed_time(end))
    report["inference"] = dict(median_ms=statistics.median(samples), samples_ms=samples)
    del graph, outputs
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    s.requires_grad_()
    z.requires_grad_()
    probes = tuple(torch.randn_like(x) * 0.01 for x in (s, z))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    graph = PackedTrainingGraph(model, inputs)
    report["training_capture_seconds"] = time.perf_counter() - start
    samples = []
    for i in range(9):
        model.zero_grad(set_to_none=True)
        s.grad = z.grad = None
        torch.cuda.synchronize()
        begin, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        begin.record()
        outputs = graph(*inputs)
        middle.record()
        torch.autograd.backward(outputs, probes)
        end.record()
        end.synchronize()
        if i >= 2:
            samples.append(
                dict(
                    forward_ms=begin.elapsed_time(middle),
                    backward_ms=middle.elapsed_time(end),
                    total_ms=begin.elapsed_time(end),
                )
            )
    report["training"] = {
        key: statistics.median(x[key] for x in samples) for key in samples[0]
    }
    report["training"]["samples"] = samples
    report["training"]["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
