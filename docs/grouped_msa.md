# Grouped packed MSA inference

The Triton packed backend now groups pair-weighted averaging (PWA) and
outer-product mean (OPM), in addition to the existing grouped triangle kernels.
The control for this follow-up is `7d8233a`, whose MSA operations still dispatched
one record at a time. Checkpoint weights and model settings are unchanged.

```sh
boltz predict inputs --batch_layout packed --batch_size 4 \
  --packed_pair_backend triton --seed 42
```

## What changed

`grouped_msa.py` builds record descriptors and workspace plans, cached on the
packed layout across MSA layers and recycling passes. `grouped_msa_kernels.py`
executes the contractions. The optimized MSA layer no longer calls PWA or OPM
once per example and no longer dispatches each attention head or OPM feature
chunk from Python.

- PWA projects packed MSA rows together and computes record-local softmax
  weights. Its contraction grid spans records, attention heads, query tiles and
  MSA-row/channel tiles. The gate is fused with contraction output storage.
- OPM projects packed MSA rows together, counts valid binary-mask intersections
  across records, and computes all feature channels through a tiled contraction.
  A separate tiled projection produces pair updates. Tile ordering reuses inputs
  in GPU cache.
- Fused normalization computes statistics in FP32 and writes directly in the
  projection dtype. BF16/FP16 contractions accumulate in FP32 and round at the
  corresponding materialization boundaries. For long records, output projections
  retain rounded partial sums in the reference head/channel order. These small
  reduction loops execute within GPU kernels; they do not launch Python work.
- Large PWA projection/gating buffers and the OPM outer-product feature buffer
  use bounded workspace waves. Default budgets are 1,048,576 MSA token rows for PWA and 32,768 pair rows
  for OPM (a target of 64 MiB per BF16 OPM feature buffer, rounded to
  whole token-row tiles). A wave may contain several differently sized records. A single
  very large record spans multiple waves. Packed inputs/outputs and OPM input
  projections still scale with total input size; dense stack boundaries elsewhere
  in Boltz also remain.

Python still loops while preparing input features and descriptor metadata, over
model layers/recycles, and over bounded workspace waves. The sequential backend
still uses the original per-record operations. This is not a claim that every
loop in Boltz has disappeared. CUDA eval mode and no-grad/inference mode are
required; no backward implementation is provided. MSA masks must be binary.
The kernels currently require the standard Boltz-2 head/OPM hidden width of 32.

## Validation

On NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB, driver 595.58.03),
Python 3.11.16 and PyTorch 2.14.0+cu130, **142 CUDA tests passed**: 21 new grouped
MSA tests, 62 existing MSA tests and 59 triangle tests. New tests cover mixed
lengths, unequal MSA depths, masked/all-masked positions, partial tiles, forced
workspace splits, FP32 execution, BF16/FP16 autocast, inference guards, a real allocation crossing the 2³¹-element indexing boundary,
and exact isolation
of unaffected records. Output projection weights are nonzero.

The grouped implementation is **not bitwise identical** to the control. The
four operator benchmarks below produced BF16 relative L2 differences up to
0.001404 (0.1404%). This is an internal numerical difference, not a structural
accuracy percentage. Structural accuracy requires the full-prediction comparison.

## Complete-operation benchmarks

Each entry is a median of seven timed calls after two warm-ups. Both arms include
normalization, learned projections, contractions, output assembly and GPU
completion. Descriptor setup and compilation are excluded from these warm
operator numbers. The reference is the per-record Triton MSA implementation at
`7d8233a`; both arms use the same generated inputs and weights.

| MSA rows × tokens per record | PWA before / grouped (ms) | OPM before / grouped (ms) |
|---|---:|---:|
| 16,224×1,062 | 192.866 / 110.601 | 157.253 / 116.646 |
| 8,192×744; 1,322×812; 518×286; 6,832×761 | 141.652 / 71.558 | 115.743 / 66.351 |
| 8,192×139; 65×385; 1×412; 1,025×79 | 10.199 / 4.614 | 8.302 / 3.153 |
| 1×412; 1×387; 1×17; 1×79 | 2.166 / 0.632 | 3.734 / 1.298 |

These are operator speedups, not full-model speedups. Memory also depends on the
input mix: the largest single-record PWA peak fell from 13.25 to 6.77 GiB and OPM
from 11.08 to 7.87 GiB, while the mixed-batch OPM peak rose from 5.64 to 6.23 GiB.
The packed input buffers are included in these allocation measurements.

An eight-target trunk preflight measured MSA time 15.566 → 11.601 seconds and
trunk time 51.602 → 47.711 seconds. This is a small preflight, not a 116-target
throughput result.

## Full default-setting Structure116 comparison

Both arms completed all 116 targets with packed batch size four and the same
Triton triangle backend. The control is `7d8233a`; the candidate adds grouped MSA
operations. Each uses seed 42, **default three recycles, one diffusion sample,
Euler 200 steps and step scale 1.5**, producing 116 finite structures and 23,200
sample-weighted denoiser evaluations. Inputs, cached MSAs and checkpoint match.
This measures the incremental MSA improvement over the previous packed version;
it is not a new original-Boltz-2 baseline. See the [earlier report](full_structure116.md)
for the dataset's construction and limitations.

| Full-prediction measurement | Previous packed (`7d8233a`) | Grouped MSA |
|---|---:|---:|
| Prediction job time | 1,258.646 s | **1,215.025 s** |
| Model time, summed over batches | 1,192.730 s | 1,148.912 s |
| Peak tensor allocation | 77.875 GiB | **72.024 GiB** |
| Peak allocator reservation | 93.262 GiB | 82.799 GiB |
| Logged recovered allocation warnings | 11 | 0 |
| Completed structures | 116 | 116 |

The observed full-prediction reduction is **3.47% (43.62 seconds), or 1.036×
throughput**. Peak tensor allocation is 7.51% lower. Operator gains are larger
because diffusion, confidence and other model work are unaffected by this change.
All recovered allocation time remains included; no failed or skipped target is
accepted in either arm.

The primary clock runs from before model-library imports through checkpoint
loading, inference and output writing, after frozen-input verification. Each arm
uses two fresh processes (the four smoke targets, then the remaining 112), with
the same input order and seed reset in each phase. Batch model clocks synchronize
CUDA. Complete externally monitored process clocks were 1,270.248 versus
1,224.997 seconds; these include up to five seconds of polling lag per phase and
are not the primary speed metric. Device-memory sampling also occurs every five
seconds; the table uses exact PyTorch allocation/reservation maxima instead.

One pass per arm does not establish statistical runtime significance. Disk and
kernel caches were not isolated, and prior validation warmed some kernels.
The accepted source was frozen for each measurement and all 114 candidate source
hashes match the local implementation. An earlier candidate trial was discarded
when a large-buffer indexing issue was found; the accepted candidate was rerun
from fresh output directories after its 64-bit addressing fix and boundary test.
The completed control phases were preserved.

## Structural accuracy

Accuracy scoring uses OpenStructure 2.12.0 against the same experimental
references as the earlier benchmark. Both arms have one sample per target;
confidence ranking therefore cannot change which sample is selected. Protein
lDDT, backbone lDDT and protein-interface DockQ are evaluated separately from
internal tensor differences. This does not validate ligand poses or affinity.

| Metric | Previous packed | Grouped MSA | Mean change | Paired bootstrap 95% interval |
|---|---:|---:|---:|---:|
| Protein lDDT (116 targets) | 0.850164 | 0.849897 | -0.000267 | [-0.001483, +0.000828] |
| Backbone lDDT (116 targets) | 0.904724 | 0.904466 | -0.000259 | [-0.001595, +0.000828] |
| Protein-interface DockQ (56 targets) | 0.590696 | 0.588500 | -0.002196 | [-0.006500, +0.000393] |

The mean changes are slightly negative. All three intervals include zero, but
this **does not establish equivalence or no accuracy loss**. These are paired
10,000-resample target bootstrap intervals, seed 20260923, from one prediction
seed. They do not measure seed-to-seed variance. Scores have the precision
reported by OpenStructure; ties are ties at that reported precision.

The individual changes matter:

- **8WQ8:** protein lDDT 0.730 → 0.680 (−0.050), backbone lDDT 0.785 → 0.725
  (−0.060).
- **9BCE:** DockQ 0.793 → 0.692 (−0.101), protein lDDT 0.846 → 0.830 (−0.016).
- **8ZAD:** DockQ 0.864 → 0.839 (−0.025).
- **8TL8:** the largest protein lDDT improvement, +0.038.

Protein lDDT improved on 11 targets, tied on 81 and declined on 24. DockQ improved
on six interface targets, tied on 42 and declined on eight. All per-target scores
are included in the evidence. The preserved seed does not make diffusion
trajectories identical when upstream floating-point representations change.

The eight-target trunk preflight also found sampled single-representation relative
L2 differences of 0.149–0.161% and pair differences of 0.447–0.564%, after four
Pairformer passes. These samples do not cover the complete tensors and are not
substitutes for the structural scores above.

## Reproduction

```sh
PYTHONPATH=src python -m pytest -q \
  tests/model/test_grouped_msa.py tests/model/test_packed_msa.py \
  tests/model/test_packed_triangles.py

PYTHONPATH=src python scripts/benchmark_grouped_msa.py \
  --shapes 8192x744,1322x812,518x286,6832x761 \
  --output /path/to/measurement.json
```

`benchmark_packed_trunk.py --profile-msa` records grouped operation calls and
`packed_shapes` for the Triton backend, and per-record calls for the sequential
backend. Its CUDA event spans include dispatch gaps; they are not sums of pure
kernel busy time. All timings must be compared on identical inputs and settings.

## Evidence and cleanup

The [compact evidence](benchmarks/grouped_msa_20260923/README.md) contains aggregate
results, every target and batch, measured source hashes, operator timings, the
separate grouped-dispatch profile and checksums. The profile confirms 16 grouped
calls per MSA operation for a four-record batch (four MSA layers × four passes),
instead of 64 per-record calls.

Use `scripts/benchmark_packed_structures.py --variant packed --profile defaults`
with `--stage smoke` and `--stage remaining`, a fresh output directory per phase,
and the frozen Structure116 reference assets. Compare source `7d8233a` with this
version. Run `scripts/score_packed_structures.py` on each arm's two phase
directories using the same experimental references and OpenStructure executable;
pass the control summary as `--baseline` when scoring the candidate.

Server experiment: `/home/ubuntu/grouped-msa-20260923`. Accepted control phases:
`r_cb664829`; final candidate: `r_73d71455`; validation: `r_9714c5eb`; final
profile/evidence: `r_7213431e`. VM **514089 was verified Paused** after the complete
archive was downloaded and checked. Structural scoring then ran locally.

The separately retained raw archive is 23,272,699 bytes and contains all 232
predictions, confidence summaries, pLDDT files, receipts, logs, profiles and the
candidate source. SHA-256:

```text
ea6d2a9cfa0524bdcc8e9eb3a54d147bf577f1c616f2f49d2e20267dabb2b1ef
```
