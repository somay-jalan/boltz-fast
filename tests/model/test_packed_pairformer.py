"""Optimized packed kernels must preserve first-order training accuracy."""

import gc

import pytest
import torch

from boltz.model.layers.pairformer import PairformerModule
from boltz.model.modules.packed_pairformer import (
    PackedPairformer,
    PackedInferenceGraph,
    PackedTrainingGraph,
)
from boltz.model.modules.packed_pairformer._autograd import Prepare, PackQKV
from packed_reference import NativeBatch, prepare, pack_qkv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("n", [17, 65, 193])
def test_layout_kernels_and_vjps(n):
    torch.manual_seed(29)
    p = torch.randn(
        1, n, n, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    g = torch.randn_like(p, requires_grad=True)
    mask = torch.ones(1, n, n, device="cuda")
    mask[:, 3] = 0
    mask[:, :, 4] = 0
    for outgoing in (True, False):
        actual = Prepare.apply(p, g, mask, outgoing)
        expected = prepare(p, g, mask, outgoing)
        for a, b in zip(actual, expected):
            assert torch.equal(a, b)
        probe = tuple(torch.randn_like(x) for x in actual)
        ga = torch.autograd.grad(actual, (p, g), probe)
        gb = torch.autograd.grad(expected, (p, g), probe)
        for a, b in zip(ga, gb):
            assert torch.equal(a, b)
    q, k, v = [
        torch.randn(n, n, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    actual, expected = PackQKV.apply(q, k, v), pack_qkv(q, k, v)
    probe = tuple(torch.randn_like(x) for x in actual)
    for a, b in zip(actual, expected):
        assert torch.equal(a, b)
    for a, b in zip(
        torch.autograd.grad(actual, (q, k, v), probe),
        torch.autograd.grad(expected, (q, k, v), probe),
    ):
        assert torch.equal(a, b)


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("input_grad", [False, True])
def test_training_graph_replay(checkpointing, input_grad):
    torch.manual_seed(76)
    model = (
        PairformerModule(
            384,
            128,
            2,
            num_heads=16,
            dropout=0,
            v2=True,
            activation_checkpointing=checkpointing,
        )
        .cuda()
        .train()
    )
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.normal_(std=0.02)
    s = torch.randn(2, 33, 384, device="cuda", requires_grad=input_grad)
    z = torch.randn(2, 33, 33, 128, device="cuda", requires_grad=input_grad)
    mask = (
        torch.arange(33, device="cuda")[None]
        < torch.tensor([17, 33], device="cuda")[:, None]
    ).float()
    mask[:, 3] = 0
    pm = mask[:, :, None] * mask[:, None, :]
    xs = (s, z, mask, pm)
    graph = PackedTrainingGraph(model, xs)
    gc.collect()
    # Exercise allocator reuse after capture: external layout metadata must
    # remain alive even when temporary tensors are collected.
    allocator_churn = [
        torch.full((4096,), -100, dtype=torch.int64, device="cuda") for _ in range(4)
    ]
    ref = NativeBatch(model, (17, 33))
    for step in range(2):
        with torch.no_grad():
            s.add_(0.13)
            z.mul_(0.9)
            model.layers[0].tri_mul_out.p_in.weight.add_(0.005)
        records = []
        for fn in (ref, graph):
            model.zero_grad(set_to_none=True)
            s.grad = z.grad = None
            out = fn(*xs)
            if not records:
                probe = tuple(torch.randn_like(v) * 0.01 for v in out)
            torch.autograd.backward(out, probe)
            records.append(
                (
                    tuple(v.detach().clone() for v in out),
                    ([s.grad.clone(), z.grad.clone()] if input_grad else [])
                    + [p.grad.clone() for p in model.parameters()],
                )
            )
        for actual, expected in zip(records[1][0], records[0][0]):
            assert torch.equal(actual, expected)
        for actual, expected in zip(records[1][1], records[0][1]):
            assert torch.isfinite(actual).all()
            assert (actual - expected).norm() < 0.01 * expected.norm().clamp_min(1e-20)


def test_inference_graph():
    model = (
        PairformerModule(384, 128, 2, num_heads=16, dropout=0, v2=True).cuda().eval()
    )
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.normal_(std=0.02)
    s = torch.randn(2, 33, 384, device="cuda")
    z = torch.randn(2, 33, 33, 128, device="cuda")
    mask = torch.ones(2, 33, device="cuda")
    mask[0, 17:] = 0
    pm = mask[:, :, None] * mask[:, None, :]
    graph = PackedInferenceGraph(model, (s, z, mask, pm), (17, 33))
    with torch.no_grad():
        expected = NativeBatch(model, (17, 33))(s, z, mask, pm)
        actual = graph(s, z, mask, pm)
        for a, b in zip(actual, expected):
            assert torch.equal(a, b)
    with pytest.raises(RuntimeError, match="no_grad"):
        graph(s, z, mask, pm)
