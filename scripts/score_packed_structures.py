"""Score confidence-selected full predictions against Structure116 references."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", nargs="+", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ost", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="Normalized baseline JSON containing a targets mapping")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    targets, receipts = {}, []
    for phase in args.phases:
        receipt = json.loads((phase / "receipt.json").read_text())
        assert receipt["status"] == "completed", phase
        receipts.append(receipt)
        pred = phase / "results/boltz_results_inputs/predictions"
        for target in receipt["targets"]:
            assert target not in targets, target
            files = sorted((pred / target).glob("confidence_*.json"))
            assert len(files) == receipt["samples_per_target"], target
            samples = []
            for file in files:
                confidence = json.loads(file.read_text())
                path = file.with_name(file.name.removeprefix("confidence_")).with_suffix(".cif")
                assert path.is_file()
                samples.append({"path": path, "confidence": confidence})
            top = max(samples, key=lambda item: item["confidence"]["confidence_score"])
            fallback = all(abs(s["confidence"]["iptm"]) <= 1e-8 for s in samples)

            def per_record_score(item):
                c = item["confidence"]
                return (4 * c["complex_plddt"] + c["ptm" if fallback else "iptm"]) / 5

            corrected = max(samples, key=per_record_score)
            targets[target] = {
                "selected_path": str(top["path"]),
                "selected_confidence": top["confidence"]["confidence_score"],
                "per_record_selected_path": str(corrected["path"]),
                "ranking_changed": top["path"] != corrected["path"],
                "max_confidence_formula_difference": max(
                    abs(s["confidence"]["confidence_score"] - per_record_score(s)) for s in samples
                ),
            }
    assert len(targets) == 116, len(targets)
    assert len({r["profile"] for r in receipts}) == 1
    assert len({r["variant"] for r in receipts}) == 1
    assert len({r["checkpoint_sha256"] for r in receipts}) == 1
    assert all(r["source_hashes"] == receipts[0]["source_hashes"] for r in receipts)

    def evaluate(job):
        target, model = job
        reference = args.references / (target + ".cif")
        assert reference.is_file()
        digest = hashlib.sha256(model.read_bytes() + reference.read_bytes()).hexdigest()
        dest = args.output / (target + "_" + model.stem + "_" + digest[:12] + ".json")
        if not dest.exists():
            temporary = dest.with_suffix(".pending.json")
            command = [str(args.ost), "compare-structures", "-m", str(model), "-r", str(reference),
                       "--fault-tolerant", "--min-pep-length", "4", "--lddt", "--bb-lddt",
                       "--dockq", "--rigid-scores", "-o", str(temporary)]
            with dest.with_suffix(".log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
            data = json.loads(temporary.read_text())
            for key in ("lddt", "bb_lddt"):
                assert isinstance(data.get(key), (int, float)) and math.isfinite(data[key]), (target, key)
            temporary.replace(dest)
        data = json.loads(dest.read_text())
        score = {k: data.get(k) for k in ("lddt", "bb_lddt", "dockq_ave_full", "rmsd")}
        score["reference_interfaces"] = len(data.get("dockq_reference_interfaces") or [])
        if not score["reference_interfaces"]:
            score["dockq_ave_full"] = None
        return (target, str(model)), score

    jobs = sorted({(target, Path(row[key])) for target, row in targets.items()
                   for key in ("selected_path", "per_record_selected_path")})
    with ThreadPoolExecutor(max_workers=2) as executor:
        scores = {}
        for index, (key, result) in enumerate(executor.map(evaluate, jobs), 1):
            scores[key] = result
            if index % 10 == 0:
                print("SCORED", index, "/", len(jobs), flush=True)
    for target, row in targets.items():
        row["scores"] = scores[target, row["selected_path"]]
        row["per_record_scores"] = scores[target, row["per_record_selected_path"]]
    result = {
        "target_count": len(targets), "targets": targets,
        "profile": receipts[0]["profile"], "variant": receipts[0]["variant"],
        "checkpoint_sha256": receipts[0]["checkpoint_sha256"],
        "prediction_job_seconds": sum(r["prediction_job_seconds"] for r in receipts),
        "peak_gpu_allocation_gib": max(
            (r["peak_gpu_allocation_gib"] for r in receipts if r.get("peak_gpu_allocation_gib") is not None),
            default=None,
        ),
        "verification_only": any(r.get("verification_only", False) for r in receipts),
        "selection": "highest reported confidence per target; per-record confidence formula audited separately",
        "scored_structures": len(jobs), "ranking_changes": sum(r["ranking_changed"] for r in targets.values()),
        "metrics": {},
    }
    for metric in ("lddt", "bb_lddt", "dockq_ave_full"):
        values = [row["scores"][metric] for row in targets.values() if row["scores"][metric] is not None]
        assert values and all(math.isfinite(v) for v in values)
        result["metrics"][metric] = {"targets": len(values), "mean": sum(values) / len(values)}
    if args.baseline:
        import numpy as np

        baseline = json.loads(args.baseline.read_text())
        assert baseline["profile"] == result["profile"], "Cannot compare different recycling/sample settings"
        assert baseline["checkpoint_sha256"] == result["checkpoint_sha256"]
        assert set(baseline["targets"]) == set(targets)
        result["baseline_path"] = str(args.baseline.resolve())
        result["baseline_prediction_job_seconds"] = baseline["prediction_job_seconds"]
        result["time_ratio_baseline_over_candidate"] = baseline["prediction_job_seconds"] / result["prediction_job_seconds"]
        result["baseline_is_historical"] = baseline.get("historical", False)
        result["comparison"] = {}
        rng = np.random.default_rng(20260923)
        for metric in result["metrics"]:
            ids = [t for t in sorted(targets) if baseline["targets"][t]["scores"][metric] is not None]
            a = np.array([baseline["targets"][t]["scores"][metric] for t in ids])
            b = np.array([targets[t]["scores"][metric] for t in ids])
            delta = b - a
            bootstrap = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
            result["comparison"][metric] = {
                "targets": len(ids), "baseline_mean": float(a.mean()), "candidate_mean": float(b.mean()),
                "paired_mean_delta": float(delta.mean()),
                "paired_target_bootstrap_95pct": np.quantile(bootstrap, [.025, .975]).tolist(),
                "candidate_wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()),
                "candidate_losses": int((delta < 0).sum()),
            }
        result["uncertainty_note"] = "One seed; target bootstrap does not establish equivalence or seed-to-seed variance. Historical runtime is not a contemporaneous controlled speed benchmark."
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "targets"}, indent=2))


if __name__ == "__main__":
    main()
