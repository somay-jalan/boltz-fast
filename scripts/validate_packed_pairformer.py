import argparse, json, gc, time
from pathlib import Path
import torch
from boltz.model.layers.pairformer import PairformerModule
from boltz.model.modules.packed_pairformer import PackedPairformer, PackedTrainingGraph

p = argparse.ArgumentParser(
    description="Compare all 64 pretrained blocks to direct native PyTorch"
)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument(
    "--real-dir",
    type=Path,
    help="Trusted teacher captures: 8K69/8VR3/9K7M/8UDS.input.pt",
)
p.add_argument("--lengths", type=int, nargs="+", default=[73, 97])
p.add_argument("--seed", type=int, default=123)
p.add_argument("--graph", action="store_true")
p.add_argument("--no-checkpointing", action="store_true")
a = p.parse_args()
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(a.seed)
state = torch.load(a.checkpoint, map_location="cpu", weights_only=False, mmap=True)[
    "state_dict"
]
state = {
    k.removeprefix("pairformer_module."): v
    for k, v in state.items()
    if k.startswith("pairformer_module.")
}
model = (
    PairformerModule(
        384,
        128,
        64,
        num_heads=16,
        dropout=0,
        v2=True,
        activation_checkpointing=not a.no_checkpointing,
    )
    .cuda()
    .train()
)
model.load_state_dict(state)
lengths = (193, 193, 193, 167) if a.real_dir else tuple(a.lengths)
w = max(lengths)
b = len(lengths)
s = torch.randn(b, w, 384, device="cuda")
z = torch.randn(b, w, w, 128, device="cuda")
m = (
    torch.arange(w, device="cuda")[None] < torch.tensor(lengths, device="cuda")[:, None]
).float()
m[:, 2] = 0
pm = m[:, :, None] * m[:, None, :]
if a.real_dir:
    for i, (target, n) in enumerate(zip(["8K69", "8VR3", "9K7M", "8UDS"], lengths)):
        d = torch.load(a.real_dir / (target + ".input.pt"), weights_only=False)
        s[i, :n] = d["s"]
        z[i, :n, :n] = d["z"]
        m[i, :n] = d["mask"]
        pm[i, :n, :n] = d["pair_mask"]
xs = (s.requires_grad_(), z.requires_grad_(), m, pm)
new = PackedTrainingGraph(model, xs) if a.graph else PackedPairformer(model, lengths)


@torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
def native_reference(s, z, m, pm):
    ss = []
    zz = []
    for i, n in enumerate(lengths):
        si, zi = model(
            s[i : i + 1, :n],
            z[i : i + 1, :n, :n],
            m[i : i + 1, :n],
            pm[i : i + 1, :n, :n],
        )
        ss.append(torch.nn.functional.pad(si, (0, 0, 0, w - n)))
        zz.append(torch.nn.functional.pad(zi, (0, 0, 0, w - n, 0, w - n)))
    return torch.cat(ss), torch.cat(zz)


outputs = []
grads = []
times = []
for fn, name in [(native_reference, "native"), (new, "candidate")]:
    model.zero_grad(set_to_none=True)
    s.grad = z.grad = None
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn(*xs)
    torch.cuda.synchronize()
    fwd = time.perf_counter() - t
    if not outputs:
        probe = tuple(torch.randn_like(v) * 0.01 for v in out)
    t = time.perf_counter()
    torch.autograd.backward(out, probe)
    torch.cuda.synchronize()
    bwd = time.perf_counter() - t
    outputs.append([v.detach().cpu() for v in out])
    grads.append(
        [s.grad.cpu(), z.grad.cpu()] + [v.grad.cpu() for v in model.parameters()]
    )
    times.append([fwd, bwd])
    print(name, "complete", times[-1], flush=True)
    del out
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()


def metric(a, b):
    a = a.double()
    b = b.double()
    d = a - b
    return dict(
        relative_l2=float(d.norm() / b.norm().clamp_min(1e-30)),
        max_abs=float(d.abs().max()),
        error_sq=float(d.square().sum()),
        reference_sq=float(b.square().sum()),
        finite=bool(torch.isfinite(a).all()),
    )


om = [metric(a, b) for a, b in zip(outputs[1], outputs[0])]
gm = [metric(a, b) for a, b in zip(grads[1], grads[0])]
names = ["s_input", "z_input"] + list(dict(model.named_parameters()))
per_record = [
    dict(
        length=n,
        outputs=[
            metric(outputs[1][0][i, :n], outputs[0][0][i, :n]),
            metric(outputs[1][1][i, :n, :n], outputs[0][1][i, :n, :n]),
        ],
        input_gradients=[
            metric(grads[1][0][i, :n], grads[0][0][i, :n]),
            metric(grads[1][1][i, :n, :n], grads[0][1][i, :n, :n]),
        ],
    )
    for i, n in enumerate(lengths)
]
report = dict(
    seed=a.seed,
    graph=a.graph,
    checkpointing=not a.no_checkpointing,
    per_record=per_record,
    reference="native PyTorch PairformerModule, training shape convention",
    lengths=lengths,
    outputs=om,
    gradients=dict(zip(names, gm)),
    worst_gradient=max((g["relative_l2"], n) for n, g in zip(names, gm)),
    diagnostic_elapsed_seconds=times,
    passed=all(g["relative_l2"] < 0.01 and g["finite"] for g in om + gm),
)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "gradients"}), flush=True)


assert report["passed"], "Full-stack 1% accuracy gate failed"
