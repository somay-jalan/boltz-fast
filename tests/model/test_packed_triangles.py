"""CUDA comparisons against the existing, independently executed modules."""

import pytest
import torch

from boltz.model.layers.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from boltz.model.layers.triangular_attention.attention import TriangleAttention
from boltz.model.modules.packed import Layout, enable_packed, pair_operation

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_case(kind, lengths, channels=32):
    torch.manual_seed(17)
    if kind in ("outgoing", "incoming"):
        cls = (
            TriangleMultiplicationOutgoing
            if kind == "outgoing"
            else TriangleMultiplicationIncoming
        )
        module = cls(channels)
    else:
        module = TriangleAttention(channels, 16, 2, starting=kind == "starting")
    module = module.cuda().eval()
    # The stock final projections are zero-initialized: leave no vacuous tests.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param.ndim >= 2:
                param.normal_(0, 0.12)
    layout = Layout(lengths, max(lengths), torch.device("cuda"))
    z = torch.randn(sum(n * n for n in lengths), channels, device="cuda")
    masks = []
    for n in lengths:
        # Asymmetric holes catch transpose/bias errors in ending attention.
        mask = (torch.rand(n, n, device="cuda") > 0.2).float()
        mask[0, :] = 1
        mask[:, 0] = 1
        masks.append(mask.flatten())
    return module, z, torch.cat(masks), layout


def run(module, z, mask, layout, backend, kernels=False):
    module.packed_pair_backend = backend
    return pair_operation(
        module,
        z,
        mask,
        layout,
        kernels,
        attention=isinstance(module, TriangleAttention),
    )


@pytest.mark.parametrize("kind", ["outgoing", "incoming", "starting", "ending"])
@pytest.mark.parametrize("lengths", [(1, 7, 17, 33), (31, 33, 65)])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("channels", [32, 128])
@torch.inference_mode()
def test_matches_reference(kind, lengths, bf16, channels):
    module, z, mask, layout = make_case(kind, lengths, channels=channels)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        expected = run(module, z, mask, layout, "sequential")
        actual = run(module, z, mask, layout, "triton")
    assert expected.abs().max() > 0.01
    assert torch.isfinite(actual).all()
    # Mixed precision changes reduction order and attention softmax rounding.
    tolerance = 0.025 if bf16 else 5e-5
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("kind", ["outgoing", "incoming", "starting", "ending"])
@torch.inference_mode()
def test_cuequivariance_reference(kind):
    pytest.importorskip("cuequivariance_torch")
    module, z, mask, layout = make_case(kind, (17, 33))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = run(module, z, mask, layout, "sequential", kernels=True)
        actual = run(module, z, mask, layout, "triton")
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.025, rtol=0.025)


@pytest.mark.parametrize("kind", ["outgoing", "incoming", "starting", "ending"])
@torch.inference_mode()
def test_record_isolation(kind):
    module, z, mask, layout = make_case(kind, (7, 17, 33))
    before = run(module, z, mask, layout, "triton")
    changed = z.clone()
    lo, hi = layout.pair_offsets[1:3]
    changed[lo:hi] = torch.randn_like(changed[lo:hi]) * 3
    after = run(module, changed, mask, layout, "triton")
    torch.testing.assert_close(before[:lo], after[:lo], atol=0, rtol=0)
    torch.testing.assert_close(before[hi:], after[hi:], atol=0, rtol=0)
    assert not torch.allclose(before[lo:hi], after[lo:hi])
    # No common-length pair allocation; every output has exactly sum(N^2) rows.
    assert after.shape == z.shape


@pytest.mark.parametrize("starting", [False, True])
@torch.inference_mode()
def test_all_masked_attention_matches_finite_penalty_reference(starting):
    module, z, mask, layout = make_case("starting" if starting else "ending", (7, 17))
    mask.zero_()
    expected = run(module, z, mask, layout, "sequential")
    actual = run(module, z, mask, layout, "triton")
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("bf16", [False, True])
@torch.inference_mode()
def test_pairformer_four_passes(bf16):
    from boltz.model.layers.pairformer import PairformerModule

    torch.manual_seed(23)
    module = (
        PairformerModule(
            token_s=64,
            token_z=32,
            num_blocks=2,
            num_heads=4,
            pairwise_head_width=16,
            pairwise_num_heads=2,
            v2=True,
        )
        .cuda()
        .eval()
    )
    for param in module.parameters():
        if param.ndim >= 2:
            param.normal_(0, 0.04)
    lengths = (7, 17, 33)
    mask = (
        torch.arange(33, device="cuda")[None, :]
        < torch.tensor(lengths, device="cuda")[:, None]
    )
    pair_mask = (mask[:, :, None] & mask[:, None, :]).float()
    s = torch.randn(3, 33, 64, device="cuda")
    z = torch.randn(3, 33, 33, 32, device="cuda")

    def evaluate(backend):
        enable_packed(module, pair_backend=backend)
        si, zi = s.clone(), z.clone()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            for _ in range(4):
                si, zi = module(si, zi, mask.float(), pair_mask, use_kernels=False)
        return si, zi

    expected = evaluate("sequential")
    actual = evaluate("triton")
    tolerance = 0.025 if bf16 else 2e-4
    for got, want in zip(actual, expected):
        torch.testing.assert_close(
            got.float(), want.float(), atol=tolerance, rtol=tolerance
        )


def test_requires_inference():
    module, z, mask, layout = make_case("outgoing", (7,))
    with torch.enable_grad(), pytest.raises(RuntimeError, match="no_grad"):
        run(module, z, mask, layout, "triton")


@pytest.mark.parametrize("kind", ["outgoing", "incoming", "starting", "ending"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_nontrivial_norm_parameters_and_input_dtypes(kind, dtype):
    module, z, mask, layout = make_case(kind, (5, 19, 67), channels=128)
    for name, parameter in module.named_parameters():
        if parameter.ndim == 1:
            parameter.normal_(0.7 if name.endswith("weight") else 0.0, 0.15)
    z = z.to(dtype)
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        expected = run(module, z, mask, layout, "sequential")
        actual = run(module, z, mask, layout, "triton")
    tolerance = 0.025 if dtype != torch.float32 else 1e-4
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=tolerance, rtol=tolerance
    )


@torch.inference_mode()
def test_rejects_unsafe_index_range_before_launch():
    from types import SimpleNamespace
    from boltz.model.modules.packed_triangles import _check

    # Exercise the address limit without allocating a multi-gigabyte tensor.
    values = SimpleNamespace(
        is_cuda=True, dtype=torch.float32, ndim=2, shape=(2**24, 128)
    )
    layout = SimpleNamespace(pair_offsets=[0, 2**24])
    with pytest.raises(ValueError, match=r"2\^31"):
        _check(SimpleNamespace(training=False), values, None, layout)


@torch.inference_mode()
def test_rejects_noncontiguous_pair_rows():
    module, z, mask, layout = make_case("outgoing", (7,))
    z = z.transpose(0, 1).contiguous().transpose(0, 1)
    assert not z.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        run(module, z, mask, layout, "triton")
