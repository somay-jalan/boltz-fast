"""Full structure inference on a frozen Structure116 phase.

Select the source explicitly: the saved pristine upstream source for B1, or
the candidate source for packed B4. This harness never stops at Pairformer.
Each invocation creates a new output directory and refuses partial success.
"""

import time

START = time.perf_counter()

import argparse
import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Source directory containing boltz/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["upstream", "packed"], required=True)
    parser.add_argument("--profile", choices=["saved", "defaults"], required=True)
    parser.add_argument("--stage", choices=["smoke", "remaining"], required=True)
    args = parser.parse_args()
    reference, source, output = args.reference.resolve(), args.source.resolve(), args.output.resolve()
    meta = json.loads((reference / "metadata.json").read_text())
    targets = meta[args.stage + "_ids"]
    assert len(targets) == (4 if args.stage == "smoke" else 112)
    assert len(set(meta["smoke_ids"] + meta["remaining_ids"])) == 116
    recycles, samples = (5, 5) if args.profile == "saved" else (3, 1)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "started", "variant": args.variant, "profile": args.profile,
        "stage": args.stage, "targets": targets, "source": str(source),
        "reference": str(reference), "seed": 42, "recycles": recycles,
        "samples_per_target": samples, "solver": "euler", "steps": 200,
        "step_scale": 1.5, "batch_size": 4 if args.variant == "packed" else 1,
        "sample_weighted_denoiser_evaluations": 0, "batches": [],
        "allocator": os.environ.get("PYTORCH_ALLOC_CONF"),
    }

    def persist():
        receipt["elapsed_including_validation_seconds"] = time.perf_counter() - START
        temp = output / "receipt.json.tmp"
        temp.write_text(json.dumps(receipt, indent=2))
        temp.replace(output / "receipt.json")

    def finish():
        if receipt["status"] != "completed":
            receipt["status"] = "failed_or_interrupted"
        persist()

    atexit.register(finish)
    persist()
    # Input setup and integrity checks occur before the separately reported CLI
    # prediction-job clock. No preprocessing or MSA server requests are allowed.
    receipt["source_hashes"] = {
        str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob("*.py"))
    }
    if args.variant == "upstream":
        assert receipt["source_hashes"] == meta["source_hashes"]["b1"]
    checkpoint = reference / "cache/boltz2_conf.ckpt"
    receipt["checkpoint_sha256"] = sha256(checkpoint)
    assert receipt["checkpoint_sha256"] == meta["checkpoint_sha256"]
    inputs = output / "inputs"
    inputs.mkdir()
    for target in targets:
        path = reference / "inputs" / (target + ".yaml")
        assert path.is_file()
        (inputs / path.name).symlink_to(path)
    original = reference / "shared/processed"
    manifest = json.loads((original / "manifest.json").read_text())
    records = {r["id"]: r for r in manifest["records"]}
    assert set(records) == set(meta["smoke_ids"] + meta["remaining_ids"])
    assert all(not records[t].get("affinity") for t in targets)
    processed = output / "results/boltz_results_inputs/processed"
    processed.mkdir(parents=True)
    for path in original.iterdir():
        if path.name != "manifest.json":
            (processed / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    (processed / "manifest.json").write_text(json.dumps({"records": [records[t] for t in targets]}))
    receipt["input_hashes"] = {t: sha256(inputs / (t + ".yaml")) for t in targets}
    for target, actual in receipt["input_hashes"].items():
        assert actual == meta["input_hashes"]["inputs/" + target + ".yaml"], target
    receipt["manifest_sha256"] = sha256(processed / "manifest.json")
    prediction_start = time.perf_counter()
    sys.path.insert(0, str(source))
    import torch
    import gemmi
    import boltz.main as cli

    assert Path(cli.__file__).resolve().is_relative_to(source)
    assert torch.cuda.is_available()
    receipt.update(torch=torch.__version__, python=sys.version, gpu=torch.cuda.get_device_name())
    torch.set_float32_matmul_precision("highest")

    def frozen_inputs(*, data, out_dir, **kwargs):
        loaded = json.loads((Path(out_dir) / "processed/manifest.json").read_text())
        assert [r["id"] for r in loaded["records"]] == targets
        assert {p.stem for p in data} == set(targets)
        print("FROZEN_INPUTS", len(targets), flush=True)

    cli.process_inputs = frozen_inputs
    load_original = cli.Boltz2.load_from_checkpoint

    def load(path, *positional, **kwargs):
        assert Path(path).resolve() == checkpoint.resolve()
        if args.profile == "saved":
            # Match the historical experiment's explicit settings, rather than
            # misrepresenting these as exact CLI defaults.
            kwargs["use_kernels"] = True
            kwargs["steering_args"]["contact_guidance_update"] = False
        model = load_original(path, *positional, **kwargs)
        receipt.update(
            predict_args=kwargs["predict_args"],
            diffusion_process_args=kwargs["diffusion_process_args"],
            steering_args=kwargs["steering_args"],
            use_kernels=model.use_kernels,
        )
        assert model.predict_args["recycling_steps"] == recycles
        assert model.predict_args["diffusion_samples"] == samples
        assert model.predict_args["sampling_steps"] == 200
        assert model.structure_module.step_scale == 1.5
        assert getattr(model.structure_module, "solver", "euler") == "euler"
        assert not model.affinity_prediction
        network = model.structure_module.preconditioned_network_forward

        def count(coords, *a, **k):
            receipt["sample_weighted_denoiser_evaluations"] += len(coords)
            return network(coords, *a, **k)

        model.structure_module.preconditioned_network_forward = count
        if args.variant == "packed" and samples > 1:
            heads = model.confidence_module.confidence_heads
            head_forward = heads.forward

            def compact_heads(*a, **k):
                result = head_forward(*a, **k)
                # These outputs are unused during prediction. The saved packed
                # full-structure harness used the identical memory amendment.
                for key in ("pde_logits", "pae_logits", "plddt_logits", "resolved_logits"):
                    result.pop(key, None)
                return result

            heads.forward = compact_heads
            receipt["discard_unused_confidence_training_logits"] = True
        original_forward = model.forward

        def forward(feats, *a, **k):
            torch.cuda.synchronize()
            started = time.perf_counter()
            ids = [r.id for r in feats["record"]]
            print("BATCH_ENTER", ids, flush=True)
            try:
                result = original_forward(feats, *a, **k)
            except torch.OutOfMemoryError as error:
                # Boltz catches RuntimeError OOMs; use a different exception to
                # stop immediately instead of accepting silently skipped inputs.
                raise MemoryError("Full benchmark aborted on GPU OOM") from error
            torch.cuda.synchronize()
            receipt["batches"].append({"ids": ids, "model_seconds": time.perf_counter() - started})
            persist()
            print("BATCH_COMPLETE", ids, flush=True)
            return result

        model.forward = forward
        return model

    cli.Boltz2.load_from_checkpoint = staticmethod(load)
    command = ["predict", str(inputs), "--out_dir", str(output / "results"),
               "--cache", str(reference / "cache"), "--model", "boltz2", "--seed", "42"]
    if args.profile == "saved":
        command += ["--recycling_steps", "5", "--diffusion_samples", "5",
                    "--max_parallel_samples", "8", "--sampling_steps", "200", "--num_workers", "2"]
    if args.variant == "packed":
        command += ["--batch_size", "4", "--batch_layout", "packed", "--packed_pair_backend", "triton"]
    receipt["cli_arguments"] = command
    receipt["preflight_seconds"] = prediction_start - START
    persist()
    cli.cli(command, standalone_mode=False)
    torch.cuda.synchronize()
    receipt["prediction_job_seconds"] = time.perf_counter() - prediction_start
    receipt["peak_gpu_allocation_gib"] = torch.cuda.max_memory_allocated() / 2**30
    receipt["peak_gpu_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    pred = output / "results/boltz_results_inputs/predictions"
    assert {p.name for p in pred.iterdir() if p.is_dir()} == set(targets)
    for target in targets:
        structures = sorted((pred / target).glob("*_model_*.cif"))
        confidences = sorted((pred / target).glob("confidence_*.json"))
        assert len(structures) == len(confidences) == samples, target
        for path in structures:
            block = gemmi.cif.read_file(str(path)).sole_block()
            for axis in ("x", "y", "z"):
                coordinates = list(block.find_values("_atom_site.Cartn_" + axis))
                assert coordinates and all(math.isfinite(float(v)) for v in coordinates), path
        for path in confidences:
            conf = json.loads(path.read_text())
            assert math.isfinite(conf["confidence_score"]), path
            assert all(math.isfinite(v) for v in conf.values() if isinstance(v, (int, float))), path
    assert [t for batch in receipt["batches"] for t in batch["ids"]] == targets
    assert receipt["sample_weighted_denoiser_evaluations"] == len(targets) * samples * 200
    receipt.update(status="completed", structures=len(targets) * samples, finite=True)
    persist()
    print("PHASE_COMPLETE", json.dumps({k: receipt[k] for k in (
        "variant", "profile", "stage", "structures", "prediction_job_seconds", "peak_gpu_allocation_gib"
    )}), flush=True)


if __name__ == "__main__":
    main()
