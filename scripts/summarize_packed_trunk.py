"""Summarize paired cached-trunk benchmark receipts and optional output samples."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    profiles = [
        json.loads((directory / "profile.json").read_text())
        for directory in (options.reference, options.candidate)
    ]
    ids = [
        [record for row in profile["rows"] for record in row["ids"]]
        for profile in profiles
    ]
    assert ids[0] == ids[1]
    summaries = []
    for profile in profiles:
        assert profile["receipt"]["status"] == "complete"
        assert not any(profile["receipt"]["forbidden_calls"].values())
        rows = profile["rows"]
        assert all(
            row["calls"]["pairformer"] == 4 and row["output"]["finite"] for row in rows
        )
        summaries.append(
            {
                "backend": profile["receipt"]["backend"],
                "inputs": len(ids[0]),
                "total_to_outputs_s": profile["global_seconds"][
                    "process_start_to_all_pairformer_outputs"
                ],
                "startup_s": profile["global_seconds"]["startup_before_first_forward"],
                "trunk_s": sum(row["seconds"]["forward_to_pairformer"] for row in rows),
                "pairformer_s": sum(row["seconds"]["pairformer"] for row in rows),
                "peak_allocation_GiB": max(row["peak_memory_gib"] for row in rows),
                "allocator_retries": sum(row.get("allocator_retries", 0) for row in rows),
            }
        )
    report = {
        "conditions": summaries,
        "speedups": {
            key: summaries[0][key] / summaries[1][key]
            for key in ("total_to_outputs_s", "trunk_s", "pairformer_s")
        },
    }
    if (options.reference / "samples_0.pt").exists():
        import torch

        comparisons = []
        for i, row in enumerate(profiles[0]["rows"]):
            reference = torch.load(
                options.reference / f"samples_{i}.pt",
                map_location="cpu",
                weights_only=True,
            )
            candidate = torch.load(
                options.candidate / f"samples_{i}.pt",
                map_location="cpu",
                weights_only=True,
            )
            result = {"ids": row["ids"]}
            for name in ("s", "z"):
                want, got = reference[name].double(), candidate[name].double()
                assert torch.isfinite(got).all()
                diff = got - want
                result[name] = {
                    "count": want.numel(),
                    "mean_abs_error": diff.abs().mean().item(),
                    "max_abs_error": diff.abs().max().item(),
                    "relative_l2": (diff.norm() / want.norm().clamp_min(1e-20)).item(),
                    "cosine_similarity": torch.nn.functional.cosine_similarity(
                        want, got, dim=0
                    ).item(),
                }
            comparisons.append(result)
        report["sampled_representation_comparisons"] = comparisons
    text = json.dumps(report, indent=2)
    print(text)
    if options.output:
        options.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
