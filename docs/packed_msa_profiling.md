# Profiling the remaining packed MSA loops

Pair-weighted averaging and outer-product mean still dispatch per record. To
measure their contribution with the full checkpoint and real cached MSA depths:

```sh
python scripts/benchmark_packed_trunk.py \
  --reference /home/boltz_finetuning_trial/pairformer116_20260923 \
  --output /path/to/fresh/msa-profile --backend triton \
  --batch-size 4 --count 116 --profile-msa
```

Each batch in `profile.json` includes `msa_operation_profile`, grouped by
operation and actual `[1, MSA rows, tokens, channels]` shape. Entries contain call
counts, CUDA event milliseconds, and CPU dispatch seconds. Events are resolved
after the existing final Pairformer synchronization, without adding a barrier
after each record. CUDA event spans include stream idle gaps from CPU dispatch;
they are not the sum of GPU kernel durations. CPU dispatch and CUDA event times
overlap and must not be added together. Module wall-clock timings include the
profiling overhead. The first batch can include lazy initialization.

Run the same input order without `--profile-msa` into another fresh output folder
to quantify instrumentation overhead. This stops after four Pairformer passes;
it does not measure diffusion, confidence, or full prediction time. Any proposed
optimization must preserve masks, record isolation, and the existing precision
and chunking behavior, then pass numerical checks and an uninstrumented runtime
comparison. No MSA speedup is claimed until these GPU measurements complete.

## Measured bottlenecks and candidate fixes

On the RTX PRO 6000 Blackwell, the instrumented 116-target packed B4 run took
718.122 seconds in the trunk, including 172.591 seconds in the MSA module.
Pair-weighted averaging accounted for 60.891 seconds of CUDA event spans (8.48%
of trunk wall time), and outer-product mean for 53.513 seconds (7.45%). Each
operation was called 1,856 times. Event spans include dispatch gaps; these are
not pure kernel busy-time percentages. An uninstrumented A/B comparison is
required to quantify the actual improvement.

Two targeted fixes are enabled with the existing `--packed_pair_backend triton`:

- Pair-weighted averaging casts normalized inputs to the autocast projection
  dtype once, avoiding repeated full-input conversions for each head. The
  original matrix multiplication and head accumulation order stay intact.
- Outer-product mean computes binary MSA mask intersections with a tiled Triton
  reduction. It avoids materializing the MSA-depth-by-token-by-token product and
  removes the Python loop over 64-row mask chunks. The reduction preserves the
  reference's chunk boundaries and intermediate count dtype.

The per-record loops in packed MSA execution remain. These changes optimize
work within each record; they are not grouped kernels for the complete MSA
operations. The sequential backend and training path retain their original
implementation. The count kernel expects the binary masks produced by Boltz's
feature pipeline, not fractional weights.

## Correctness and warm operation measurements

The candidate passed 62 CUDA tests, covering FP32 and BF16 module execution,
FP32/BF16/FP16 count reduction, uneven lengths, MSA depths up to 16,224, zero-mask
positions, and both chunked and unchunked operations. All comparisons use exact
equality. Warm benchmarks also require exact equality of the complete operation
outputs with nonzero output weights.

Seven timed iterations follow two warm-ups in each condition. These are
single-record complete-operation timings, not whole-model speedups:

| MSA rows × tokens | PWA before / after (ms) | OPM before / after (ms) |
|---|---:|---:|
| 8,192 × 139 | 8.649 / 8.355 | 4.816 / 4.370 |
| 8,192 × 512 | 59.810 / 42.802 | 32.015 / 27.246 |
| 16,224 × 1,062 | 262.245 / 190.207 | 199.279 / 156.747 |
| 1 × 412 | 1.287 / 1.183 | 1.922 / 1.975 |

The shallow MSA case shows that an individual operation can be slightly slower.

## Full 116-target trunk comparison

Both arms use packed B4, the grouped Triton triangle backend, seed 42 and default
three recycles (four Pairformer passes). The control is commit `e331765`; the
candidate adds only these MSA changes to model execution. All cached targets and
their order are identical. Each arm runs in a fresh process, stops before diffusion
and downstream heads, and validates finite outputs. Hardware is an RTX PRO 6000
Blackwell Server Edition, 97,887 MiB, driver 595.58.03. Software: Python 3.11.16
and PyTorch 2.14.0+cu130; model precision follows the CLI defaults.

| Measurement | Before MSA changes | After MSA changes |
|---|---:|---:|
| Complete benchmark process | 742.99 s | 727.42 s |
| Startup-inclusive time to last Pairformer output | 738.84 s | 723.28 s |
| Startup before first forward | 19.50 s | 20.44 s |
| Model trunk | 717.91 s | 701.41 s |
| MSA module | 172.70 s | 156.68 s |
| Pairformer module | 523.97 s | 524.06 s |
| Peak tensor allocation | 61.08 GiB | 61.08 GiB |
| Allocator retries | 0 | 0 |

Complete process time decreased **2.10%** (1.021× throughput); MSA time decreased
**9.28%**. The process clock includes startup, finite-value checks, representation
sample writing and shutdown. The output clock ends at the final recycled
Pairformer output; earlier batches' validation overhead is included. Lazy kernel
initialization encountered during inference remains in these measurements.
This was not an isolated cold-cache or warm-cache experiment.

Twenty-two of 29 batches had lower trunk time. One run per condition, one seed
and one GPU do not establish a runtime confidence interval or universal speedup.
The larger operation-level improvements must not be applied to the whole model:
Pairformer time is almost unchanged, and diffusion is excluded here.

All 58 single/pair representation sample comparisons were **bitwise identical**,
checking up to 65,536 fixed positions per tensor per batch after the final recycle.
This is sampled equality, not a full-tensor bitwise comparison. All 112 candidate
source hashes matched the local code; the 111-file control matched its frozen
source manifest. No targets were skipped.

These measurements compare against the previous packed Triton version, not
pristine upstream Boltz-2. The earlier [full Structure116 prediction report](full_structure116.md)
is pinned to `e331765` and predates these MSA changes.


## Full prediction check and evidence

A separate four-target before/after test ran the complete prediction pipeline with
seed 42, default three recycles, one sample, and Euler 200 steps. Saved coordinates
and all confidence JSON values were identical for **8K69, 8UDS, 8VR3 and 9K7M**.
This validates complete outputs for those four targets; it does not replace a
full 116-target structural-accuracy evaluation.

[Compact evidence and checksums](benchmarks/packed_msa_20260923/README.md) include
per-batch timings, source manifests, operation profiles and validation receipts.
The separately retained raw archive is 1,074,748 bytes, SHA-256:

```text
6cfc8622a960f9c8bf4d78bfb9ced8f8a36b0beea637fcfd70a4c9a5ea9a745e
```

Managed comparison run: `r_01fdeed6`; full-prediction check: `r_79662525`.

VM 514030 was verified Paused after the evidence archive was downloaded and all
60 recorded file checksums were verified locally.

## Reproduction

For uninstrumented timing, run `benchmark_packed_trunk.py` from each source
checkout (control `e331765`, then this version) with `--backend triton --count 116
--batch-size 4 --samples`, the same `--reference`, and separate fresh `--output`
directories. Omit `--profile-msa` for the timing comparison. The complete command
above supplies the reference/output argument format. Compare saved
`samples_<batch>.pt` files at corresponding positions.

```sh
PYTHONPATH=src python -m pytest -q tests/model/test_packed_msa.py
PYTHONPATH=src python scripts/benchmark_packed_msa.py \
  --depth 16224 --tokens 1062 --repeats 7 --profile --output /path/to/fresh/micro
```

For the full smoke check, use `scripts/benchmark_packed_structures.py` with
`--variant packed --profile defaults --stage smoke` and the Structure116 reference
for both source checkouts; its `--source` argument selects each checkout's `src/`
directory. Use separate output directories and compare the saved coordinates and
confidence JSONs. This check does not invoke an MSA server or rescore against
experimental structures.
