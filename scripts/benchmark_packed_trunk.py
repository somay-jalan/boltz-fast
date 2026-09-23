"""Reuses cached inputs; never invokes diffusion or writes structures."""

import time

START = time.perf_counter()
import atexit, functools, gc, hashlib, json, os, subprocess, sys, traceback, argparse, shutil
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Compare packed triangle backends with cached full-checkpoint trunk inference"
)
parser.add_argument(
    "--reference",
    type=Path,
    required=True,
    help="Directory of the existing pairformer116 experiment",
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--backend", choices=["sequential", "triton"], required=True)
parser.add_argument("--count", type=int, default=8)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument(
    "--samples",
    action="store_true",
    help="Save bounded representation samples for numerical comparisons",
)
parser.add_argument("--start", type=int, default=0, help="Start index within the reference input order")
parser.add_argument("--native-triangles", action="store_true", help="Diagnostic: mathematical PyTorch triangles with the sequential backend")
parser.add_argument(
    "--profile-msa", action="store_true",
    help="Record per-operation MSA CUDA event spans, dispatch time and actual shapes",
)
options = parser.parse_args()
B = options.reference
BS = options.batch_size
C = options.output
assert options.count > 0 and options.count % BS == 0
C.mkdir(parents=True, exist_ok=False)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch
import boltz.main as main

assert Path(main.__file__).resolve().is_relative_to(REPO)
meta = json.loads((B / "metadata.json").read_text())
meta["targets"] = meta["targets"][options.start : options.start + options.count]
rows = []
current = None
msa_events = []
globals_ = {"imports": time.perf_counter() - START}
forbidden = {}
receipt = {
    "status": "started",
    "batch_size": BS,
    "batch_layout": "packed",
    "torch": torch.__version__,
    "python": sys.version,
    "model_path": str(main.__file__),
}
receipt["profile_msa"] = options.profile_msa
receipt["backend"] = options.backend
receipt["native_triangles"] = options.native_triangles
if options.native_triangles:
    assert options.backend == "sequential"
    from boltz.model.modules import packed
    original_pair_operation = packed.pair_operation
    def native_pair_operation(module, values, masks, layout, use_kernels, attention=False):
        return original_pair_operation(module, values, masks, layout, False, attention)
    packed.pair_operation = native_pair_operation
receipt["gpu"] = torch.cuda.get_device_name()
source_hashes = {
    str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (REPO / "src").rglob("*.py")
}
receipt["source_hashes"] = source_hashes
reference = B / f"packed_b{BS}"
(C / "inputs").mkdir()
for target in meta["targets"]:
    matches = list((reference / "inputs").glob(target + ".*"))
    assert len(matches) == 1, (target, matches)
    (C / "inputs" / matches[0].name).symlink_to(matches[0])
original = reference / "results/boltz_results_inputs/processed"
processed = C / "results/boltz_results_inputs/processed"
processed.mkdir(parents=True)
manifest = json.loads((original / "manifest.json").read_text())
records = {r["id"]: r for r in manifest["records"]}
manifest["records"] = [records[t] for t in meta["targets"]]
(processed / "manifest.json").write_text(json.dumps(manifest))
for name in ["constraints", "mols", "msa", "structures", "templates"]:
    (processed / name).symlink_to((original / name).resolve(), target_is_directory=True)
(processed / "records").mkdir()
for target in meta["targets"]:
    shutil.copy2(
        original / "records" / f"{target}.json",
        processed / "records" / f"{target}.json",
    )


def sync():
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def persist():
    (C / "profile.json").write_text(
        json.dumps(
            {
                "metadata": meta,
                "receipt": receipt,
                "global_seconds": globals_,
                "rows": rows,
                "elapsed_seconds": time.perf_counter() - START,
            },
            indent=2,
        )
    )


atexit.register(persist)


class PairformerReady(Exception):
    def __init__(self, output, ended):
        self.output = output
        self.ended = ended


def timed(obj, attr, key):
    orig = getattr(obj, attr)

    @functools.wraps(orig)
    def wrapper(*a, **k):
        sync()
        t = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            sync()
            current["seconds"][key] = (
                current["seconds"].get(key, 0) + time.perf_counter() - t
            )
            current["calls"][key] = current["calls"].get(key, 0) + 1

    setattr(obj, attr, wrapper)


def instrument_msa(model):
    """Use stream events without synchronizing after each record operation."""
    for layer in model.msa_module.layers:
        for key in ("pair_weighted_averaging", "outer_product_mean"):
            operation = getattr(layer, key)
            original = operation.forward

            def measured(*args, _original=original, _key=key, **kwargs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                before = time.perf_counter()
                output = _original(*args, **kwargs)
                dispatch = time.perf_counter() - before
                end.record()
                msa_events.append((_key, tuple(args[0].shape), start, end, dispatch))
                return output

            operation.forward = measured


def collect_msa_profile():
    # The caller already synchronized at the final Pairformer boundary.
    grouped = {}
    for key, shape, start, end, dispatch in msa_events:
        name = key + ":" + "x".join(map(str, shape))
        row = grouped.setdefault(name, {
            "operation": key, "input_shape": list(shape), "calls": 0,
            "cuda_event_ms": 0.0, "cpu_dispatch_seconds": 0.0,
        })
        row["calls"] += 1
        row["cuda_event_ms"] += start.elapsed_time(end)
        row["cpu_dispatch_seconds"] += dispatch
    current["msa_operation_profile"] = list(grouped.values())
    msa_events.clear()


trainer_predict_orig = main.Trainer.predict


def trainer_predict(self, *a, **k):
    globals_["trainer_predict_called_at"] = time.perf_counter() - START
    return trainer_predict_orig(self, *a, **k)


main.Trainer.predict = trainer_predict

process_orig = main.process_inputs


def process(**kwargs):
    t = time.perf_counter()
    process_orig(**kwargs)
    p = Path(kwargs["out_dir"]) / "processed/manifest.json"
    data = json.loads(p.read_text())
    records = {r["id"]: r for r in data["records"]}
    assert set(records) == set(meta["targets"]) and not any(
        r.get("affinity") for r in records.values()
    )
    data["records"] = [records[x] for x in meta["targets"]]
    p.write_text(json.dumps(data, indent=2))
    globals_["cached_input_checking"] = time.perf_counter() - t


main.process_inputs = process
# The endpoint is representations, not coordinates: no structure writer is needed.
main.BoltzWriter.write_on_batch_end = lambda *a, **k: None
load_orig = main.Boltz2.load_from_checkpoint


def load(path, *args, **kwargs):
    assert Path(path).name == "boltz2_conf.ckpt"
    t = time.perf_counter()
    model = load_orig(path, *args, **kwargs)
    globals_["checkpoint_loading"] = time.perf_counter() - t
    receipt.update(
        predict_args=kwargs["predict_args"],
        diffusion_process_args=kwargs["diffusion_process_args"],
        msa_args=kwargs["msa_args"],
        pairformer_args=kwargs["pairformer_args"],
        steering_args=kwargs["steering_args"],
        use_kernels=model.use_kernels,
    )
    assert (
        model.predict_args["recycling_steps"] == 3
        and model.structure_module.solver == "euler"
        and model.structure_module.step_scale == 1.5
    )
    assert (
        not model.is_pairformer_compiled
        and not model.is_msa_compiled
        and not model.affinity_prediction
    )
    for attr, key in [
        ("input_embedder", "input_embedding"),
        ("msa_module", "msa"),
        ("template_module", "template"),
    ]:
        if getattr(model, attr, None) is not None:
            timed(getattr(model, attr), "forward", key)
    if options.profile_msa:
        instrument_msa(model)
    pair_orig = model.pairformer_module.forward

    def pair(*a, **k):
        sync()
        t = time.perf_counter()
        output = pair_orig(*a, **k)
        sync()
        end = time.perf_counter()
        current["seconds"]["pairformer"] = (
            current["seconds"].get("pairformer", 0) + end - t
        )
        current["calls"]["pairformer"] = current["calls"].get("pairformer", 0) + 1
        if current["calls"]["pairformer"] == model.predict_args["recycling_steps"] + 1:
            raise PairformerReady(output, end)
        return output

    model.pairformer_module.forward = pair

    def block(name):
        def blocked(*a, **k):
            forbidden[name] = forbidden.get(name, 0) + 1
            raise AssertionError("Forbidden downstream execution: " + name)

        return blocked

    for attr in [
        "distogram_module",
        "diffusion_conditioning",
        "confidence_module",
        "bfactor_module",
        "affinity_module",
    ]:
        if getattr(model, attr, None) is not None:
            forbidden[attr] = 0
            getattr(model, attr).forward = block(attr)
    forbidden["diffusion_sample"] = 0
    model.structure_module.sample = block("diffusion_sample")
    forbidden["diffusion_forward"] = 0
    model.structure_module.forward = block("diffusion_forward")
    forbidden["denoiser"] = 0
    model.structure_module.preconditioned_network_forward = block("denoiser")
    original_predict = model.predict_step

    def predict(batch, batch_idx, *a, **k):
        global current
        sync()
        torch.cuda.reset_peak_memory_stats()
        allocation_retries = torch.cuda.memory_stats()["num_alloc_retries"]
        msa_events.clear()
        current = {
            "index": len(rows),
            "ids": [r.id for r in batch["record"]],
            "tokens": [int(v) for v in batch["token_pad_mask"].sum(-1).tolist()],
            "atoms": [int(v) for v in batch["atom_pad_mask"].sum(-1).tolist()],
            "seconds": {},
            "calls": {},
            "input_msa_kind": type(batch["msa"]).__name__,
        }
        assert len(current["ids"]) == BS
        print("BATCH_START", BS, batch_idx, current["ids"], flush=True)
        current["process_start_to_batch_start_seconds"] = time.perf_counter() - START
        if not rows:
            globals_["startup_before_first_forward"] = current[
                "process_start_to_batch_start_seconds"
            ]
            globals_["trainer_setup_and_first_batch"] = (
                globals_["startup_before_first_forward"]
                - globals_["trainer_predict_called_at"]
            )
        t = time.perf_counter()
        try:
            original_predict(batch, batch_idx, *a, **k)
        except PairformerReady as done:
            current["seconds"]["forward_to_pairformer"] = done.ended - t
            current["process_start_to_output_seconds"] = done.ended - START
            s, z = done.output
            check_start = time.perf_counter()
            assert torch.isfinite(s).all().item() and torch.isfinite(z).all().item()
            current["output"] = {
                "s_shape": list(s.shape),
                "z_shape": list(z.shape),
                "s_dtype": str(s.dtype),
                "z_dtype": str(z.dtype),
                "finite": True,
            }
            current["seconds"]["output_validation"] = time.perf_counter() - check_start
            if options.samples:
                # Fixed equally spaced flat positions keep the validation artifact bounded.
                samples = {}
                for name, tensor in [("s", s), ("z", z)]:
                    flat = tensor.flatten()
                    count = min(65536, flat.numel())
                    indices = (
                        torch.arange(count, device=flat.device, dtype=torch.int64)
                        * (flat.numel() - 1)
                        // max(1, count - 1)
                    )
                    samples[name] = flat.index_select(0, indices).float().cpu()
                torch.save(samples, C / f"samples_{batch_idx}.pt")
            del s, z
        else:
            raise AssertionError("Did not stop after the final Pairformer recycle")
        assert (
            current["calls"]["pairformer"] == 4
            and current["calls"]["msa"] == 4
            and not any(forbidden.values())
        )
        if options.profile_msa:
            collect_msa_profile()
        current["allocator_retries"] = torch.cuda.memory_stats()["num_alloc_retries"] - allocation_retries
        current["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        current["forbidden_calls"] = dict(forbidden)
        rows.append(current)
        persist()
        print("BATCH_DONE", json.dumps(current), flush=True)
        return {"pairformer_only": True}

    model.predict_step = predict
    return model


main.Boltz2.load_from_checkpoint = staticmethod(load)
globals_["imports_and_instrumentation"] = time.perf_counter() - START
try:
    # Leave model, precision, workers, MSA depth, recycling and solver defaults intact.
    main.cli(
        [
            "predict",
            str(C / "inputs"),
            "--out_dir",
            str(C / "results"),
            "--cache",
            meta["cache"],
            "--batch_size",
            str(BS),
            "--batch_layout",
            "packed",
            "--packed_pair_backend",
            options.backend,
            "--seed",
            "42",
        ],
        standalone_mode=False,
    )
    sync()
    globals_["total_process"] = time.perf_counter() - START
    ids = [x for row in rows for x in row["ids"]]
    assert ids == meta["targets"] and len(rows) == len(meta["targets"]) // BS
    assert not any(forbidden.values())
    assert not list((C / "results").rglob("*.cif"))
    for rel, h in source_hashes.items():
        assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == h
    receipt.update(
        status="complete",
        targets_completed=len(ids),
        batches_completed=len(rows),
        pairformer_passes_per_batch=4,
        forbidden_calls=forbidden,
        source_unchanged=True,
    )
    globals_["process_start_to_all_pairformer_outputs"] = rows[-1][
        "process_start_to_output_seconds"
    ]
    globals_["mean_startup_amortized_seconds_per_input"] = globals_[
        "process_start_to_all_pairformer_outputs"
    ] / len(meta["targets"])
    persist()
    (C / "COMPLETE").write_text(
        f"{options.count} inputs, four Pairformer passes per batch, no downstream execution\n"
    )
    print("CONDITION_COMPLETE", BS, json.dumps(globals_), flush=True)
except BaseException as e:
    receipt.update(status="failed", error=repr(e), forbidden_calls=forbidden)
    persist()
    traceback.print_exc()
    raise
