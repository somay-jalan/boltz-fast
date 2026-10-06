# Packed Pairformer: forward, backward and ragged batching

This branch provides two Pairformer execution paths:

- **Native PyTorch:** `boltz.model.layers.pairformer.PairformerModule`, the unchanged mathematical reference. Pass `use_kernels=False` when comparing it with the packed backend.
- **Optimized packed:** `boltz.model.modules.packed_pairformer.PackedPairformer`, with eager autograd and optional `PackedInferenceGraph` / `PackedTrainingGraph` wrappers. The graph wrappers use the same packed implementation; they are not separate numerical backends.

There is no `packed_new`, approximate fast tail, `PackedExactBatch`, or legacy grouped-triangle Pairformer in this branch. The original fast implementation and its historical measurements remain recoverable at commit `9a8e890a1dd6ed87109530b7a5394572c7903b57`.

## What changed

The packed adapter flattens each record's token rows and pair square and keeps those states packed across all blocks. Layer normalization and pointwise operations process the packed buffers; matrix products retain each record's native shapes. Records never attend to one another. This is hybrid packing, not a single grouped GEMM for the entire batch.

The optimized triangle preparation and Q/K/V layout kernels use explicitly tiled, coalesced memory access. Their custom first-order backward fuses inverse layouts, masking, sigmoid derivatives and the required BF16 rounding boundaries. A single `torch.split` partitions each packed tensor, avoiding repeated full-buffer allocation/zeroing by independent slice backward nodes. Native-shaped GEMMs, softmax reduction behavior and FP32 single-track attention are preserved.

Backward propagates through the actual forward operations and computes both input and model-weight gradients. Graph capture retains layout/index buffers for its entire lifetime to prevent allocator reuse from invalidating CUDA pointers. No output cache, transformed checkpoint, frozen weights, or native-model replay is substituted for backward. Activation checkpointing deliberately recomputes the **same packed blocks** when enabled.

## Installation and supported configuration

Install the repository with `python -m pip install -e .`. The packed backend additionally requires a CUDA-enabled PyTorch and Triton's Gluon API. The tested environment is **Python 3.11, PyTorch 2.14.0+cu130, Triton 3.8.0, RTX PRO 6000 Blackwell (96 GB)**. Use a compatible PyTorch/Triton pair supplied together by your CUDA environment; the ordinary installation's `torch>=2.2` minimum does not imply packed-backend compatibility. Other hardware/software versions need their own validation.

The native backend does not import Gluon. The packed backend requires:

- A Boltz-2 `PairformerModule`: single width 384, pair width 128, triangle attention 4 heads × 32 channels. The pretrained 64-block module has **196,534,272 parameters**.
- FP32 parameters and all four input tensors on the same CUDA device. Pair projections internally use BF16 autocast; the single track remains FP32. Disable TF32 for the recorded numerical protocol.
- `dropout=0` during training. Inference uses `model.eval()` and `torch.no_grad()`.
- Explicit record lengths that retain internal masked positions and discard only trailing padding. Outputs are zero-padded back to the supplied batch width.

## Loading the original checkpoint

Only load trusted checkpoint/capture files: these examples use PyTorch's pickle-capable loader.

```python
from pathlib import Path
import torch
from boltz.model.layers.pairformer import PairformerModule

checkpoint = Path.home() / ".boltz" / "boltz2_conf.ckpt"
state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
weights = {k.removeprefix("pairformer_module."): v
           for k, v in state.items() if k.startswith("pairformer_module.")}
model = PairformerModule(
    384, 128, 64, num_heads=16, dropout=0, v2=True,
    activation_checkpointing=True,
).cuda().float()
model.load_state_dict(weights, strict=True)
torch.backends.cuda.matmul.allow_tf32 = False
```

Checkpoint keys are unchanged. Use `model.parameters()`, `model.state_dict()` and your optimizer as usual; save the original `model`, not the adapter's wrapper-prefixed state dict.

## Inputs and eager execution

| Tensor | Shape | Meaning |
|---|---|---|
| `s` | `[B, Nmax, 384]` | Single state at the Pairformer boundary |
| `z` | `[B, Nmax, Nmax, 128]` | Pair state at that boundary |
| `mask` | `[B, Nmax]` | FP32 token-validity mask |
| `pair_mask` | `[B, Nmax, Nmax]` | FP32 pair-validity mask |

There are no MSA, template or diffusion inputs here. Cached teacher states must be captured at the Pairformer boundary. This call executes **one Pairformer stack**; it does not perform recycling, MSA/template updates or diffusion.

```python
from boltz.model.modules.packed_pairformer import PackedPairformer

# s, z, mask, pair_mask are padded FP32 CUDA tensors loaded from your data.
lengths = (193, 193, 193, 167)
packed = PackedPairformer(model, lengths, width=s.shape[1])
model.eval()
with torch.no_grad():
    s_out, z_out = packed(s, z, mask, pair_mask)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
optimizer.zero_grad(set_to_none=True)
s_out, z_out = packed(s, z, mask, pair_mask)
# Teacher targets are detached and already aligned to this batch layout.
s_loss = ((s_out - teacher_s.detach()).square() * mask[..., None]).sum()
s_loss = s_loss / (mask.sum().clamp_min(1) * s_out.shape[-1])
z_loss = ((z_out - teacher_z.detach()).square() * pair_mask[..., None]).sum()
z_loss = z_loss / (pair_mask.sum().clamp_min(1) * z_out.shape[-1])
loss = s_loss + z_loss
loss.backward()
optimizer.step()
```

Teacher inputs do **not** need `requires_grad=True` for weight training. Set it on `s`/`z` only if you also need their gradients. The adapter supports block activation checkpointing to trade compute for activation memory. Custom backward supports first-order derivatives only.

For the native path, call `model(s, z, mask, pair_mask, use_kernels=False)` under the desired autocast context. Accuracy tests call native PyTorch independently on each unpadded record and concatenate/pad the results, matching the packed per-record mathematical problem. Native reentrant checkpointing requires a grad-requiring input; the test reference supplies one when testing weight-only training.

## CUDA graphs for repeated batches

```python
from boltz.model.modules.packed_pairformer import (
    PackedInferenceGraph, PackedTrainingGraph,
)

model.eval()
inference = PackedInferenceGraph(model, (s, z, mask, pair_mask), lengths)
with torch.no_grad():
    s_out, z_out = inference(s, z, mask, pair_mask)
# Release this graph before changing its model's execution configuration.
del inference

model.train()
training = PackedTrainingGraph(model, (s, z, mask, pair_mask))
optimizer.zero_grad(set_to_none=True)
s_out, z_out = training(s, z, mask, pair_mask)
loss = objective(s_out, z_out)  # your differentiable, mask-aware loss
loss.backward()
optimizer.step()
```

Graph construction warms up/JIT-compiles kernels and captures execution, so the first use can be slow. Keep a graph for repeated batches of the **same shape and record lengths**. The training wrapper derives lengths from the last valid token in each mask. Eager execution also uses a fixed-layout adapter: construct another adapter for different lengths/width.

Complete one backward before issuing the next training-graph forward. Accumulation across completed forward/backward pairs and in-place optimizer updates are supported. Retained-graph backward, higher-order gradients, concurrent calls and overlapping training forwards are unsupported. Rebuild after changing lengths, shape, dtype, device, input `requires_grad`, parameter storage or model configuration. Never place valid tokens beyond declared lengths. Inference also requires rebuilding after changing checkpointing/training configuration or replacing parameter storage; it is the caller's responsibility to obey that contract. Returned graph outputs are cloned so the next replay cannot overwrite a retained result.

## Accuracy results

The acceptance gate is **relative L2 < 0.01 independently** for `s_out`, `z_out`, both input gradients and every one of 3,392 parameter-gradient tensors:

`relative_l2 = ||candidate - native||₂ / max(||native||₂, 1e-30)`.

This is a per-tensor norm criterion, not a per-element percentage bound. Maximum absolute errors and per-record metrics are also recorded. The original validated run used all 64 pretrained blocks, FP32 weights/states, BF16 autocast, TF32 disabled, and native unchunked training arithmetic.

| Inputs | Seed | Checkpointing | Forward / input-gradient error | Worst weight-gradient error |
|---|---:|---|---:|---:|
| Random B4: 65/73/97/128 | 901 | On | 0% / 0% | 0.30451% |
| Real B4: 193/193/193/167 | 901 | On | 0% / 0% | 0.28153% |
| Random B2: 65/385 | 123 | On | 0% / 0% | 0.17378% |
| Random B2: 73/97 | 123 | Off | 0% / 0% | 0.18161% |

The real records are 8K69, 8VR3 and 9K7M cropped to 193 tokens (original lengths 1618/760/356), and full 167-token 8UDS. These are four teacher captures, not 116 full-length predictions. This change has **not** rerun the 116-target final diffusion/structure-accuracy benchmark. Older Structure116 reports are historical evidence for older commits, not validation of this backend. Tested numerical equality is not a universal guarantee across shapes, weights, GPUs or compiler versions.

## Performance results and limits

Matched CUDA-graph benchmarks on RTX PRO 6000, real B4 193/193/193/167:

| Operation | Previous packed adapter | Optimized packed | Speedup |
|---|---:|---:|---:|
| Inference | 404.88 ms | 359.45 ms | 1.13× |
| Training forward | 530.27 ms | 485.77 ms | 1.09× |
| Backward, including checkpoint recomputation | 2457.90 ms | 1764.94 ms | 1.39× |
| Total forward + backward | 2988.12 ms | 2250.78 ms | 1.33× |

Inference uses 15 measured replays after 3 warmups; training uses 7 measured pairs after 2 warmups. Medians exclude JIT/capture and include input/output device copies. Training enables block checkpointing; optimizer time is excluded. A total's median need not equal the sum of separate medians.

The removed original approximate fast branch at `9a8e890` is **still faster in inference: 242.82 ms**, but has no supported training backward and does not meet the same validated gradient contract. Sequential accurate graph execution measured 378.88 ms inference and 2236.50 ms total training: the new backend improves its inference by about 1.05×, while total training is approximately equal (0.64% slower). No universal batching speedup is claimed. Real B4 peak training allocation was about 9.54 GiB for packed versus 7.04 GiB sequential accurate.

Raw per-case metrics and timing samples live in [benchmarks/packed_pairformer](benchmarks/packed_pairformer/). Historical backend labels in those files identify measured baselines; they are not selectable implementations in this branch. The published timings precede source consolidation; cleanup verification is recorded separately.

## Cleanup verification (2026-10-06)

The consolidated source passed **100 GPU regression tests**: 17 packed Pairformer/kernel/replay/integration cases and 83 retained MSA cases. All four full 64-block validations above were rerun against direct native PyTorch and reproduced the same numerical errors. A CLI help smoke test confirms the renamed MSA option.

The cleaned implementation measured **367.90 ms inference**, **1766.33 ms backward**, and **2253.28 ms total training** on the same real B4 batch. These are a fresh sanity check, not a new matched rerun of removed baselines. [Cleanup evidence](benchmarks/packed_pairformer/cleanup/summary.json), per-parameter compressed metrics, timing samples and source hashes are included. The native checkpoint reference emitted PyTorch's existing warning about its unspecified `use_reentrant`; the packed path explicitly uses nonreentrant checkpointing. No final-diffusion accuracy claim is added by these tests.

## Running checks

```sh
PYTHONPATH=src python -m pytest -q \
  tests/model/test_packed_pairformer.py \
  tests/model/test_packed_pairformer_replay.py \
  tests/model/test_packed_pairformer_training.py

PYTHONPATH=src python scripts/validate_packed_pairformer.py \
  --checkpoint ~/.boltz/boltz2_conf.ckpt --graph --seed 901 \
  --lengths 65 73 97 128 --output results/random.json

PYTHONPATH=src python scripts/validate_packed_pairformer.py \
  --checkpoint ~/.boltz/boltz2_conf.ckpt --graph --seed 901 \
  --real-dir /path/to/teacher/captures --output results/real.json
```

Measure the supported backend's current performance (same warmup/sample counts):

```sh
PYTHONPATH=src python scripts/benchmark_packed_pairformer.py \
  --checkpoint ~/.boltz/boltz2_conf.ckpt \
  --real-dir /path/to/teacher/captures --output results/timing.json
```

Omit `--real-dir` and supply `--lengths 73 97` to benchmark random inputs. This script measures the current packed backend only; historical comparison data is included separately.

Also run `--lengths 65 385 --seed 123 --graph` and `--lengths 73 97 --seed 123 --graph --no-checkpointing` for the full recorded suite. The validator exits nonzero if any output/input/weight gradient exceeds the 1% gate. Its elapsed times include diagnostics and are **not** steady-state benchmark results. Regression tests include odd lengths, mask holes and fully masked rows, both triangle orientations, current input/weight replay, and layout-buffer lifetime checks.

## Branch cleanup and migration

Use the three public classes above in place of the experimental `PackedFast*` names. There are no compatibility aliases to removed numerical implementations. The native Pairformer no longer dispatches through implicit legacy `packed_inference` flags; optimized Pairformer execution is explicit through the Python adapter API.

Unrelated packed MSA, atom/diffusion execution and the optional Heun solver remain from the parent branch. The CLI option is now `--packed_msa_backend {sequential,triton}` (formerly `--packed_pair_backend`) because it only selects MSA grouping; it does **not** enable this Pairformer adapter. `boltz predict` uses the native Pairformer. End-to-end integration of the new packed adapter is separate from the standalone training/inference API validated here. Existing historical benchmark scripts now use the renamed MSA option and cannot reproduce the removed Pairformer by selecting it.
