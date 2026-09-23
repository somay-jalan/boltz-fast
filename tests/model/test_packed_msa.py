"""Validate binary mask reduction, including reference low-precision rounding."""
import pytest
import torch

from boltz.model.layers.outer_product_mean import OuterProductMean

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference_counts(mask, chunked):
    if not chunked:
        return (mask[:, :, None, :] * mask[:, :, :, None]).sum(1).clamp(min=1)
    result = None
    for start in range(0, mask.shape[1], 64):
        m = mask[:, start:start + 64]
        count = (m[:, :, None, :] * m[:, :, :, None]).sum(1)
        if result is None:
            result = count
        else:
            result += count
    return result.clamp(min=1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("depth", [1, 65, 1025, 8193, 16224])
@pytest.mark.parametrize("chunked", [False, True])
@torch.inference_mode()
def test_binary_counts(dtype, depth, chunked):
    from boltz.model.modules.packed_msa import binary_mask_counts
    torch.manual_seed(42)
    mask = (torch.rand(1, depth, 37, 1, device="cuda") > 0.3).to(dtype)
    mask[:, :, 0] = 0
    mask[:, :, 1:3] = 1
    actual = binary_mask_counts(mask, dtype, chunked)
    expected = reference_counts(mask, chunked)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("chunk", [None, 4])
@pytest.mark.parametrize("depth,n", [(1, 43), (65, 79), (1025, 129), (8193, 17)])
@torch.inference_mode()
def test_outer_product_output(autocast, chunk, depth, n):
    torch.manual_seed(43)
    module = OuterProductMean(64, 32, 128).cuda().eval()
    module.proj_o.weight.normal_(0, 0.03)
    module.proj_o.bias.normal_(0, 0.03)
    dtype = torch.bfloat16 if autocast else torch.float32
    m = torch.randn(1, depth, n, 64, device="cuda", dtype=dtype)
    mask = (torch.rand(1, depth, n, device="cuda") > 0.3).float()
    mask[:, :, 0] = 0
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        module.packed_pair_backend = "sequential"
        expected = module(m, mask, chunk)
        module.packed_pair_backend = "triton"
        actual = module(m, mask, chunk)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("depth,n", [(1, 43), (65, 79), (1025, 129), (64, 413)])
@pytest.mark.parametrize("chunk_heads", [False, True])
@torch.inference_mode()
def test_pair_weighted_averaging_output(autocast, depth, n, chunk_heads):
    from boltz.model.layers.pair_averaging import PairWeightedAveraging
    torch.manual_seed(44)
    module = PairWeightedAveraging(64, 128, 32, 8).cuda().eval()
    module.proj_o.weight.normal_(0, 0.03)
    dtype = torch.bfloat16 if autocast else torch.float32
    m = torch.randn(1, depth, n, 64, device="cuda", dtype=dtype)
    z = torch.randn(1, n, n, 128, device="cuda")
    mask = (torch.rand(1, n, n, device="cuda") > 0.3).float()
    mask[:, 0, :] = 0
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        module.packed_pair_backend = "sequential"
        expected = module(m, z, mask, chunk_heads)
        module.packed_pair_backend = "triton"
        actual = module(m, z, mask, chunk_heads)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
