"""Graph replay must use current inputs/weights and enforce its lifetime rules."""

import copy

import pytest
import torch

from boltz.model.layers.pairformer import PairformerModule
from boltz.model.modules.packed_pairformer import PackedTrainingGraph

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("input_grad", [False, True])
def test_replay_and_weight_update(checkpointing, input_grad):
    torch.manual_seed(901)
    net = (
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
        for module in net.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.normal_(std=0.02)
    s = torch.randn(1, 17, 384, device="cuda", requires_grad=input_grad)
    z = torch.randn(1, 17, 17, 128, device="cuda", requires_grad=input_grad)
    mask = torch.ones(1, 17, device="cuda")
    pm = mask[:, :, None] * mask[:, None, :]
    from packed_reference import NativeBatch

    # Building a graph after ordinary training must work without changing
    # existing .grad buffers or globally overriding CUDA stream checks.
    previous = NativeBatch(net, (17,))(s, z, mask, pm)
    torch.autograd.backward(previous, tuple(torch.ones_like(x) for x in previous))
    saved_gradients = [p.grad.clone() for p in net.parameters()]
    graph = PackedTrainingGraph(net, (s, z, mask, pm))
    for p, saved in zip(net.parameters(), saved_gradients):
        torch.testing.assert_close(p.grad, saved, rtol=0, atol=0)
    del previous, saved_gradients
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    for step in range(2):
        # Compare to an independent eager native PyTorch execution after changing
        # both input values and parameters. A cached result cannot pass this.
        reference = copy.deepcopy(net)
        eager = NativeBatch(reference, (17,))
        ss = (s.detach() + step * 0.2).requires_grad_(input_grad)
        zz = (z.detach() - step * 0.1).requires_grad_(input_grad)
        sr = ss.detach().clone().requires_grad_(input_grad)
        zr = zz.detach().clone().requires_grad_(input_grad)
        net.zero_grad(set_to_none=True)
        actual = graph(ss, zz, mask, pm)
        expected = eager(sr, zr, mask, pm)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        probe = tuple(torch.randn_like(x) * 0.01 for x in actual)
        torch.autograd.backward(actual, probe)
        torch.autograd.backward(expected, probe)
        for a, b in zip(net.parameters(), reference.parameters()):
            assert (a.grad - b.grad).norm() < 0.01 * b.grad.norm().clamp_min(1e-20)
        if input_grad:
            torch.testing.assert_close(ss.grad, sr.grad, rtol=0, atol=0)
            torch.testing.assert_close(zz.grad, zr.grad, rtol=0, atol=0)
        optimizer.step()
    with pytest.raises(ValueError, match="lengths"):
        changed = mask.clone()
        changed[:, -1] = 0
        graph(s, z, changed, pm)
    output = graph(s, z, mask, pm)
    with pytest.raises(RuntimeError, match="Complete backward"):
        graph(s, z, mask, pm)
    torch.autograd.backward(output, tuple(torch.zeros_like(x) for x in output))
    output = graph(s, z, mask, pm)
    probe = tuple(torch.zeros_like(x) for x in output)
    torch.autograd.backward(output, probe, retain_graph=True)
    with pytest.raises(RuntimeError, match="exactly one backward"):
        torch.autograd.backward(output, probe, retain_graph=True)
    net.eval()
    with pytest.raises(ValueError, match="configuration"):
        graph(s, z, mask, pm)
