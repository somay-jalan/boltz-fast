# Grouped triangles for packed inference

This is the optimized fixed packed implementation on
`feature/packed-triangle-kernels`. The fixes cover triangle dispatch and the
reference contraction precision boundary. The full benchmark retains a known
confidence-ranking limitation; see the [report](full_structure116.md).

The experimental `triton` pair backend keeps separate variable-length pair
squares in the existing packed buffer. It replaces the per-record calls for
outgoing/incoming triangle multiplication and starting/ending triangle attention.
The default `sequential` backend is unchanged.

```sh
PYTHONPATH=src python -m boltz.main predict inputs --batch_layout packed --batch_size 4 \
  --packed_pair_backend triton --seed 42
```

The regular prediction command still runs the complete model. To compare with a
Pairformer-only experiment, add the new option to that experiment's existing
early-stop harness. Do not compare a full prediction against a trunk-only time.

Python callers can use `enable_packed(model, pair_backend="triton")` after
`model.eval()`, and execute within `torch.no_grad()` or `torch.inference_mode()`.
This backend requires CUDA and Triton and does not implement backward. It does
not fall back silently if a kernel fails. `use_kernels`/`--no_kernels` continues
to select the cuEquivariance implementation for the sequential backend; selecting
the Triton pair backend explicitly replaces those triangle operations.

## Implementation

- `packed_triangles.py` applies existing learned projections, normalization and
  gates across packed rows, and caches work descriptors on `Layout`.
- `packed_triangle_kernels.py` launches one contraction grid across actual
  `(record, channel, pair tile)` work. BF16/FP16 operands accumulate in FP32, then round to the projection dtype
  before output normalization, matching the reference contraction boundary. FP32
  dot products explicitly use IEEE input precision.
- Normalization writes directly in the projection dtype. This removes repeated
  full-size FP32-to-BF16/FP16 conversion buffers.
- Multiplication fuses the input projections, gates, mask and layout conversion
  into one kernel. Its contraction works on channel-major record-local squares;
  a fused transpose/normalization writes the output projection input directly.
- Attention combines Q/K/V/gate projections that share the same normalized input.
  Both triangle operations fuse the final sigmoid and elementwise gate.
- Tile sizes were selected using measured GPU timings, including partial tiles
  and ending-node attention's transposed indexing. They are not hardware-agnostic
  autotuning results.
- Triangle attention uses online softmax across each record's keys. It reads
  the triangle bias at `(query, key)` within that record. Ending-node attention
  transposes pair indexing in the kernel and writes in original pair order.
- Nonexistent tail positions are masked. Real masked pairs retain the existing
  finite mask penalty. No maximum-length pair square or cubic score tensor is
  allocated by these kernels.
- Existing dense tensors at stack compatibility boundaries still exist. This
  change does not make the complete Boltz pipeline allocation-free or remove
  its boundary padding.
- MSA pair layers use the triangle backend too. The subsequent
  [grouped MSA implementation](grouped_msa.md) also groups pair-weighted averaging
  and outer-product mean. The earlier [MSA profiling report](packed_msa_profiling.md)
  documents the intermediate per-record optimization at `7d8233a`.

## Validation and measurement

```sh
PYTHONPATH=src python -m pytest -q tests/model/test_packed_triangles.py
PYTHONPATH=src python scripts/benchmark_packed_triangles.py
```

The tests compare against the existing per-record modules with nonzero learned
projections, cover unequal lengths and partial tiles, check record isolation,
and exercise four passes through a small Pairformer. These tests do not establish
scientific accuracy or parity of a full checkpoint prediction.

The benchmark uses synthetic pair representations, includes projections and
gates, and warms compilation/descriptor creation before timing. It compares
against the existing cuEquivariance-backed packed path. A smaller launch count
is not a speed guarantee: layout/precision costs and an already-saturated GPU
can make the grouped implementation slower. Measure on representative shapes
before using it for throughput runs.

The final RTX PRO 6000 Blackwell validation passed 59 CUDA tests. In the 116-input
cached full-checkpoint comparison (packed batch 4, four Pairformer passes), total
time to all outputs was **838.88 → 736.41 s (1.139×)**,
and Pairformer time was **616.34 → 522.50 s (1.180×)**.
Peak whole-trunk allocation was approximately 61.09 GiB for both. All inputs
finished with finite representations and zero downstream calls.

Sampled final pair-representation relative L2 differences range from
0.310% to 4.436% (median 0.344%). The backend is not
bitwise equivalent. A subsequent [full Structure116 comparison](full_structure116.md)
completed 580 structures: 1.40× throughput versus historical pristine upstream B1,
with protein lDDT 0.8534 versus 0.8541. That comparison includes the complete packed
implementation, not only the incremental triangle-kernel change. The backend stays
opt-in. The original slower grouped prototype needed fusion of large intermediate
buffers and better tile choices; grouping alone did not produce a speedup.

### Reproducing the cached-trunk comparison

The harness accepts the existing `pairformer116` evidence directory, uses its
cached inputs and checkpoint, writes to a new output directory, and refuses to
skip inputs. It does not modify the original experiment. Use a fresh process
and output directory for each condition:

```sh
python scripts/benchmark_packed_trunk.py \
  --reference /path/to/pairformer116_20260923 \
  --output /path/to/new-triton-results \
  --backend triton --count 116 --batch-size 4 --samples
# Repeat with --backend sequential and a different --output.
python scripts/summarize_packed_trunk.py \
  /path/to/new-sequential-results /path/to/new-triton-results
```

The harness includes checkpoint loading and lazy initialization, synchronizes at
coarse module boundaries, checks finite final representations, and verifies that
no diffusion or downstream head executed. `--samples` saves a bounded subset of
representations after each timed forward; that validation and file-writing time
is included in subsequent inter-batch wall time. Trunk and Pairformer totals
exclude the sample-writing overhead. A single run per condition does not
establish statistical confidence or performance on other GPUs.
