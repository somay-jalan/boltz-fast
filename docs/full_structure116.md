# Full Structure116 comparison

The grouped Triton packed B4 implementation completed all **116 targets and
580 structures** on September 23, 2026. Against the saved original Boltz-2 B1
baseline, total process time fell from **78.72 to 56.22 minutes**: **1.40×
throughput and 28.6% less time**. Mean structural scores were slightly lower,
with paired uncertainty intervals spanning zero. Individual targets sometimes
changed substantially; this is not a claim of equivalent predictions or accuracy.

## Configuration and scope

Both configurations use the same Boltz-2 checkpoint, seed 42, Euler 200 steps,
step scale 1.5, five recycling steps, five samples per target, maximum eight
parallel samples, BF16 model precision and native FP32 diffusion arithmetic.
Five recycles and five samples are benchmark overrides: the CLI defaults are
three recycles and one sample. Structure kernels are enabled. The new arm uses
`--batch_size 4 --batch_layout packed --packed_pair_backend triton`.

The original B1 source is pristine upstream
`b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`. The candidate is based on
`1e3f292306a542dd8dd68a0c350d95386a8ebcbd` plus the grouped triangle changes,
published as [`e331765`](https://github.com/somay-jalan/boltz-fast/commit/e3317655b64041b6f866585ca3c03c620a40f384).
These full-prediction measurements predate the subsequent MSA optimizations;
check out that commit to reproduce this exact source version.
The comparison measures the complete packed implementation against pristine
upstream; it does not isolate the incremental triangle-kernel contribution.
The separate [Pairformer experiment](packed_triangles.md) compares triangle
backends within packed B4.

Each arm has the original four-target smoke phase and remaining 112-target phase,
with seed 42 initialized in each fresh process. Target order and phase boundaries
match the baseline. All 833 cached input/feature/MSA files matched the historical
hashes; checkpoint and all 111 candidate Python source hashes were also verified.
No templates, experimental coordinates, affinity inference or contact/physics
guidance are supplied to the model. Experimental coordinates are used for scoring.

The independently selected Structure116 set contains 60 single-chain and 56
multichain protein targets. It is not a verified subset of the Boltz-2 paper's
evaluation set. Provenance, selection filters and original baseline receipts are
in `structure116_20260909` in the
[experiment repository](native_batching.md#reproduction-and-validation).

## Full inference time and memory

Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, driver
595.58.03. The original baseline used the same GPU model. Its timings are
historical, not contemporaneous repeated controls.

| Measurement | Original B1 | Packed B4 + Triton |
|---|---:|---:|
| Structures completed | 580 / 580 | 580 / 580 |
| Smoke process time | 329.09 s | 250.94 s |
| Remaining process time | 4,394.06 s | 3,122.44 s |
| Total process time | 4,723.15 s / 78.72 min | 3,373.38 s / 56.22 min |
| Peak observed device memory | 68.50 GiB | 94.20 GiB |

The primary 1.40× ratio uses full process clocks. Both include model imports,
loading, prediction, output writing and two process startups; the new clock also
includes source/input preflight, final validation and driver polling overhead.
Shared MSA/input preparation, offline scoring and transfers are excluded. The
candidate's internal prediction-job clock, excluding preflight/final validation,
was 3,356.01 s; it is not the primary timing used above. Device memory comes from
`nvidia-smi` polling; peak candidate tensor allocation was 77.88 GiB.

There were zero failed/skipped targets and no process restarts. Nine CUDA
expandable-segment allocation warnings were recovered internally; their overhead
remains included in timing. The five-sample packed harness discards unused
confidence training logits after confidence scores are calculated, as in the
historical packed harness. It does not change checkpoint weights or score arithmetic.

## Experimental structure accuracy

OpenStructure 2.12.0 scores the highest reported-confidence sample per target
against the same experimental references. DockQ averages only the 56 applicable
protein-complex targets. Protein lDDT and DockQ do not establish ligand pose or
binding affinity accuracy.

| Metric | Targets | Original B1 | Packed B4 | Packed − original | Paired target bootstrap 95% interval |
|---|---:|---:|---:|---:|---|
| Protein all-atom lDDT | 116 | 0.8541 | 0.8534 | −0.00065 | [−0.00302, +0.00163] |
| Protein backbone lDDT | 116 | 0.9086 | 0.9075 | −0.00109 | [−0.00343, +0.00117] |
| Protein-interface DockQ | 56 | 0.6139 | 0.6125 | −0.00138 | [−0.02370, +0.01857] |

All intervals include zero. These are 10,000 paired target resamples at one seed;
they do not quantify seed-to-seed variance or demonstrate equivalence. All 580
structures were generated and checked for finite coordinates/confidences. The
primary scoring pass evaluated 116 selected structures plus two alternative-ranked
samples. The original 116 selected structures were rescored locally before this
run, reproducing every archived score exactly.

### Individual targets and ranking

The close averages hide some substantial target-level changes. The largest
protein lDDT loss was **−0.065 on 8WQ8**; the largest DockQ loss was **−0.429 on
8KA6**. The largest improvements were +0.056 lDDT and +0.281 DockQ, both on 8Y9K.

A post-hoc check of all five samples on the two largest-loss targets found:

- **8KA6:** the packed best-of-five DockQ was 0.560 versus original 0.478,
  even though its confidence-selected packed sample lost 0.429. A better packed
  sample was available; sample selection contributes to this result.
- **8WQ8:** packed best-of-five lDDT was 0.766 versus original 0.834.
  Ranking alone does not explain this target's loss.

These two-target oracle checks are diagnostics, not whole-dataset best-of-five
results or a deployable selection rule. A one-seed comparison cannot separate
systematic packing effects from changes in stochastic sampling trajectories.

The pre-existing packed confidence code selects pTM versus ipTM using a condition
across the batch. In mixed monomer/complex batches this changes reported scores
for 18 targets versus upstream's per-target formula. Re-ranking offline changes
the selected sample for **8IRQ and 8WX1**. With that formula, packed mean lDDT is
0.8536, backbone lDDT 0.9077 and DockQ 0.6125. Predictions and the primary table
above remain unchanged. This ranking issue was not patched mid-experiment.

## Reproduction and evidence

Use the preserved original environment and Structure116 server assets. Each phase
must use a fresh output directory. For example, for a new repeat:

```sh
PYTORCH_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/boltz-original/bin/python \
  scripts/benchmark_packed_structures.py \
  --reference /home/boltz_finetuning_trial/structure116_20260909 \
  --source /home/ubuntu/boltz-packed-triangles-20260923/src \
  --output /home/ubuntu/full-structure116-repeat/packed_smoke \
  --variant packed --profile saved --stage smoke
```

Repeat with `--stage remaining` and a new `packed_remaining` output. The harness
requires completed receipts, 580 structures, finite coordinates/confidences and
116,000 sample-weighted Euler denoiser evaluations across the phases. It fails
rather than accepting skipped targets.

`score_packed_structures.py` accepts the two phase directories, experimental
references, OpenStructure executable, an output directory and normalized baseline
JSON. It reports primary scores, the confidence-ranking audit and paired bootstrap
intervals. The separate post-hoc outlier scoring is retained in the evidence.

For literal CLI defaults, use `--profile defaults` for both arms and run a fresh
`--variant upstream` baseline with the preserved `src_b1` source. The scorer
rejects comparisons between the default profile and the five-sample saved baseline.

Managed run: `r_1b760dc1`. Server experiment:
`/home/ubuntu/full-structure116-20260923`. VM **513539 was verified Paused** after
successful evidence download and before local structural scoring.

The downloaded prediction/evidence archive contains all structures, confidence
summaries, per-token pLDDT, receipts and logs (57,027,629 bytes). SHA-256:

```text
95dda794d3b30d796812c5a2bfa5b2c1b2a8ea2de63710662a776b3e03bf0813
```

The [published compact evidence](benchmarks/structure116_20260923/README.md)
includes aggregate results, all 116 per-target scores, input verification,
benchmarked source hashes and archive metadata, with checksums. Detailed scoring
outputs, phase receipts and the raw prediction archive are retained locally;
the raw archive is not included in this repository.
