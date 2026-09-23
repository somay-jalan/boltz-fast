# Native batching and EDM Heun branches

`boltz-fast` publishes the native batching and EDM Heun work originally developed in the `/home/boltz-original` server checkout. The canonical repository for this work is now `https://github.com/somay-jalan/boltz-fast.git`. The Python package and command remain named `boltz`.

## Branches and lineage

| Branch | Purpose | Code provenance |
|---|---|---|
| `main` | Default branch: native batching, optional Heun solver, and original publication documentation | Same implementation as `feature/native-batched-edm-heun` |
| `feature/native-batched-inference` | Native batch handling, packed layout, per-record randomness, and upstream-compatible native-padding centering; Euler solver | `c0ff971`, `1453124`, `d32b60f` |
| `feature/native-batched-edm-heun` | All native batching changes plus an opt-in Heun corrector | Based on `d32b60f`; implementation commit `e11b933` |
| `feature/packed-triangle-kernels` | Optimized fixed packed version: grouped Triton triangle kernels, BF16 contraction rounding fix, and full Structure116 report | Based on `1e3f292`; enable with `--packed_pair_backend triton` |

These branches descend from upstream commit `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`. The original implementation commits are preserved in the history. In this standalone repository, `main` and `feature/native-batched-edm-heun` initially share the same tip; `feature/native-batched-inference` retains the pre-Heun Euler implementation at `d32b60f`. The original upstream baseline can be inspected by its pinned commit.

## Running

Install the checked-out branch into a suitable environment:

```sh
git clone --branch feature/packed-triangle-kernels https://github.com/somay-jalan/boltz-fast.git
cd boltz-fast
pip install -e '.[cuda]'
```

Use prepared YAML inputs and cached MSAs for reproducible comparisons. With an input directory named `inputs`, native packed Euler can be invoked with:

```sh
boltz predict inputs --batch_size 4 --batch_layout packed \
  --diffusion_solver euler --sampling_steps 200 --seed 42
```

Heun is opt-in; Euler remains the default. A structure-only run requires YAMLs without affinity properties:

```sh
boltz predict inputs --batch_size 4 --batch_layout packed \
  --diffusion_solver heun --sampling_steps 50 \
  --step_scale 1.0 --no_contact_guidance --seed 42
```

For affinity inputs, set `--sampling_steps_affinity` independently. `--diffusion_samples` controls samples per target, not the number of different targets in a batch. See [the solver details](edm_heun.md) for restrictions and evaluation counts. Heun uses a unit step scale, rejects potential/contact guidance, and is supported only for Boltz-2. A 50-step Heun trajectory uses 99 denoiser evaluations; a 100-step trajectory uses 199.

Packed layout groups compatible records and restores record-specific outputs; atom padding and random-number order are part of numerical behavior. Different target grouping, sample counts, precision, kernel settings, or seeds can change performance and predictions. Lower step counts are experimental and do not imply equivalent accuracy.

Packed triangle operations can optionally use a grouped Triton backend with
`--packed_pair_backend triton`. See [packed triangle kernels](packed_triangles.md)
for implementation details, CUDA validation and benchmarking commands. The
existing per-record triangle backend remains the default.

## Reproduction and validation

The experiment repository is [somay-jalan/boltz_finetuning_trial](https://github.com/somay-jalan/boltz_finetuning_trial). Its publication branch `chore/publish-reproducible-experiments-20260922` documents harnesses, settings, input provenance and artifact retention. Checkpoints and bulky raw artifacts must be restored separately.

On 22 September 2026 the working `src` matched every Python file in both `heun87_20260909/src` and `structure116_20260909/src_b4`. No model arithmetic was changed during publication. Validation against the installed Boltz environment passed:

- Three existing batched contact/template-gradient tests (`tests/model/test_batched_guidance.py`).
- Heun analytical update, terminal Euler step, 99/199-call counts, heterogeneous B4 padding, direct/chunked multiplicity-five consistency and corrector rigid-frame invariance.
- Bitwise preservation of Euler outputs relative to the saved pre-Heun sampler with alignment on/off.
- Native-padding centering against pristine upstream at multiplicities one and five.

Run the in-repository gradient tests with `python -m pytest -q tests/model/test_batched_guidance.py`. The analytical sampler and centering harnesses are retained under `heun87_20260909/test_heun.py` and `upstream_center87_20260908/test_centering.py` in the experiment repository; their historical path assumptions and the exact validation commands are documented there. These are focused regression checks, not a new full benchmark or an accuracy guarantee.
