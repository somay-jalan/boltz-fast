"""Nontrivial packed-batch equivalence, replay, and backward regressions."""

import pytest
import torch

from boltz.model.layers.pairformer import PairformerModule
from boltz.model.modules.packed_pairformer import PackedPairformer, PackedInferenceGraph
from packed_reference import NativeBatch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def setup_case():
    torch.manual_seed(926)
    model = (
        PairformerModule(
            384, 128, 2, num_heads=16, dropout=0, v2=True, activation_checkpointing=True
        )
        .cuda()
        .eval()
    )
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.normal_(std=0.025)
    lengths = (17, 33)
    s = torch.randn(2, 33, 384, device="cuda")
    z = torch.randn(2, 33, 33, 128, device="cuda")
    mask = (
        torch.arange(33, device="cuda")[None]
        < torch.tensor(lengths, device="cuda")[:, None]
    ).float()
    mask[:, 5] = 0
    pm = mask[:, :, None] * mask[:, None, :]
    return model, lengths, (s, z, mask, pm)


@pytest.mark.parametrize("concurrent", [False, True])
def test_bitwise_replay_with_changed_inputs_and_weights(concurrent):
    model, lengths, xs = setup_case()
    packed = PackedPairformer(model, lengths, concurrent=concurrent)
    ref = NativeBatch(model, lengths)
    with torch.no_grad():
        for a, b in zip(packed(*xs), ref(*xs)):
            assert torch.equal(a, b)
        static = tuple(x.clone() for x in xs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                outputs = packed(*static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = packed(*static)
        for _ in range(2):
            xs[0].add_(0.13)
            xs[1].mul_(0.9)
            model.layers[0].tri_mul_out.p_in.weight.add_(0.007)
            for dst, src in zip(static, xs):
                dst.copy_(src)
            graph.replay()
            for a, b in zip(outputs, ref(*xs)):
                assert torch.equal(a, b)
        assert not outputs[0][0, lengths[0] :].count_nonzero()
        assert not outputs[1][0, lengths[0] :].count_nonzero()
        assert not outputs[1][0, :, lengths[0] :].count_nonzero()


def test_backward_and_layout_guards():
    model, lengths, xs = setup_case()
    model.train()
    xs = (xs[0].requires_grad_(), xs[1].requires_grad_(), *xs[2:])
    packed = PackedPairformer(model, lengths)
    gradients = []
    for fn in (NativeBatch(model, lengths), packed):
        model.zero_grad(set_to_none=True)
        xs[0].grad = xs[1].grad = None
        outputs = fn(*xs)
        if not gradients:
            probes = tuple(torch.randn_like(x) * 0.01 for x in outputs)
        torch.autograd.backward(outputs, probes)
        gradients.append(
            [xs[0].grad.clone(), xs[1].grad.clone()]
            + [p.grad.clone() for p in model.parameters()]
        )
    # Packed weight-gradient accumulation is not claimed to be bitwise.
    for actual, expected in zip(gradients[1], gradients[0]):
        assert torch.isfinite(actual).all()
        assert (actual - expected).norm() <= 0.01 * expected.norm().clamp_min(1e-20)
    with pytest.raises(ValueError, match="fixed batch layout"):
        packed(xs[0][:, :-1], *xs[1:])
    with pytest.raises(ValueError, match="nondifferentiable"):
        packed(*xs[:2], xs[2].clone().requires_grad_(), xs[3])


def test_public_inference_graph():
    model, lengths, xs = setup_case()
    graph = PackedInferenceGraph(model, xs, lengths)
    with pytest.raises(RuntimeError, match="no_grad"):
        graph(*xs)
    with torch.no_grad():
        old = graph(*xs)
        saved = tuple(x.clone() for x in old)
        xs[0].add_(0.2)
        xs[1].mul_(0.7)
        model.layers[0].tri_mul_in.p_in.weight.add_(0.005)
        actual = graph(*xs)
        expected = NativeBatch(model, lengths)(*xs)
        for a, b in zip(actual, expected):
            assert torch.equal(a, b)
        for a, b in zip(old, saved):
            assert torch.equal(a, b)
        with pytest.raises(ValueError, match="layout"):
            graph(xs[0][:, :-1], *xs[1:])
        model.train()
        with pytest.raises(ValueError, match="eval"):
            graph(*xs)


def test_retained_msa_pair_update_uses_native_operations():
    from boltz.model.layers.pairformer import PairformerNoSeqLayer
    from boltz.model.modules.packed import Layout, enable_packed, msa_pair_layer

    torch.manual_seed(32)
    layer = PairformerNoSeqLayer(128, dropout=0).cuda().eval()
    with torch.no_grad():
        for m in layer.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.normal_(std=0.02)
    enable_packed(layer, pair_backend="triton")  # This now selects only MSA kernels.
    layout = Layout((17, 33), 33, torch.device("cuda"))
    z = torch.randn(2, 33, 33, 128, device="cuda")
    mask = (
        torch.arange(33, device="cuda")[None]
        < torch.tensor([17, 33], device="cuda")[:, None]
    ).float()
    pm = mask[:, :, None] * mask[:, None, :]
    with torch.no_grad():
        actual = msa_pair_layer(
            layer, layout.pack(z, True), layout.pack(pm, True), layout, False
        )
        expected = torch.cat(
            [
                layer(zi, mi).reshape(-1, 128)
                for zi, mi in zip(
                    layout.parts(layout.pack(z, True), True),
                    layout.parts(layout.pack(pm, True), True),
                )
            ]
        )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
