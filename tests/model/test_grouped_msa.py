"""Mixed lengths, masks, workspace boundaries and record isolation."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("shapes", [[(3, 17), (7, 39), (1, 9)], [(65, 385), (9, 47)], [(1025, 19), (1, 387)]])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("small_workspace", [True, False])
@torch.inference_mode()
def test_grouped_msa(shapes, dtype, small_workspace):
    autocast = dtype != torch.float32
    from boltz.model.modules.grouped_msa import MSAPlan, pair_weighted_averaging, outer_product_mean
    from boltz.model.layers.pair_averaging import PairWeightedAveraging
    from boltz.model.layers.outer_product_mean import OuterProductMean
    torch.manual_seed(19)
    plan = MSAPlan(shapes, "cuda", token_budget=2048 if small_workspace else 1048576,
                   pair_budget=2048 if small_workspace else 65536)
    ms = [torch.randn(s * n, 64, device="cuda", dtype=dtype) for s, n in shapes]
    zs = [torch.randn(n * n, 128, device="cuda") for s, n in shapes]
    masks = [(torch.rand(s, n, device="cuda") > .2).float() for s, n in shapes]
    pairs = [(torch.rand(n, n, device="cuda") > .2).float() for s, n in shapes]
    for mask, pair in zip(masks, pairs):
        mask[:, 0] = 0
        pair[0] = 0
    m, z = torch.cat(ms), torch.cat(zs)
    pwa = PairWeightedAveraging(64, 128, 32, 8).cuda().eval()
    opm = OuterProductMean(64, 32, 128).cuda().eval()
    pwa.proj_o.weight.normal_(0, .03)
    opm.proj_o.weight.normal_(0, .03)
    opm.proj_o.bias.normal_(0, .03)
    pwa.packed_pair_backend = opm.packed_pair_backend = "triton"
    with torch.autocast("cuda", dtype=dtype if autocast else torch.bfloat16, enabled=autocast):
        a = pair_weighted_averaging(pwa, m, z, torch.cat([p.flatten() for p in pairs]), plan)
        b = outer_product_mean(opm, m, torch.cat([p.flatten() for p in masks]), plan)
        ar, br = [], []
        for mi, zi, mask, pair, (s, n) in zip(ms, zs, masks, pairs, shapes):
            ar.append(pwa(mi.view(1, s, n, 64), zi.view(1, n, n, 128), pair[None], n > 384).reshape(-1, 64))
            br.append(opm(mi.view(1, s, n, 64), mask[None], 4 if n > 384 else None).reshape(-1, 128))
        for actual, expected in [(a, torch.cat(ar)), (b, torch.cat(br))]:
            error = (actual.float() - expected.float())
            rel = error.norm() / expected.float().norm()
            assert torch.isfinite(actual).all()
            print("ERROR", autocast, shapes, float(rel), float(error.abs().max()))
            if autocast:
                assert rel < .002
                torch.testing.assert_close(actual.float(), expected.float(), atol=.012, rtol=.04)
            else:
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


@torch.inference_mode()
def test_record_isolation():
    from boltz.model.modules.grouped_msa import MSAPlan, pair_weighted_averaging, outer_product_mean
    from boltz.model.layers.pair_averaging import PairWeightedAveraging
    from boltz.model.layers.outer_product_mean import OuterProductMean
    torch.manual_seed(23)
    plan = MSAPlan([(7, 29), (9, 41)], "cuda", token_budget=350, pair_budget=500)
    m = torch.randn(plan.m_offsets[-1], 64, device="cuda")
    z = torch.randn(plan.p_offsets[-1], 128, device="cuda")
    mm = torch.ones(m.shape[0], device="cuda")
    pm = torch.ones(z.shape[0], device="cuda")
    pwa = PairWeightedAveraging(64, 128, 32, 8).cuda().eval()
    opm = OuterProductMean(64, 32, 128).cuda().eval()
    pwa.proj_o.weight.normal_(0, .03)
    opm.proj_o.weight.normal_(0, .03)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a = pair_weighted_averaging(pwa, m, z, pm, plan)
        b = outer_product_mean(opm, m, mm, plan)
        m[plan.m_offsets[1]:].normal_(10, 4)
        z[plan.p_offsets[1]:].normal_(-8, 2)
        mm[plan.m_offsets[1]:] = 0
        pm[plan.p_offsets[1]:] = 0
        aa = pair_weighted_averaging(pwa, m, z, pm, plan)
        bb = outer_product_mean(opm, m, mm, plan)
    torch.testing.assert_close(a[:plan.m_offsets[1]], aa[:plan.m_offsets[1]], atol=0, rtol=0)
    torch.testing.assert_close(b[:plan.p_offsets[1]], bb[:plan.p_offsets[1]], atol=0, rtol=0)
    assert not torch.equal(a[plan.m_offsets[1]:], aa[plan.m_offsets[1]:])
    assert not torch.equal(b[plan.p_offsets[1]:], bb[plan.p_offsets[1]:])


@torch.inference_mode()
def test_inference_only():
    from boltz.model.modules.grouped_msa import MSAPlan, outer_product_mean
    from boltz.model.layers.outer_product_mean import OuterProductMean
    plan = MSAPlan([(2, 3)], "cuda")
    m = torch.ones(6, 64, device="cuda")
    mask = torch.ones(6, device="cuda")
    module = OuterProductMean(64, 32, 128).cuda()
    with pytest.raises(RuntimeError, match="eval inference"):
        outer_product_mean(module, m, mask, plan)


@torch.inference_mode()
def test_normalization_beyond_int32_elements():
    from boltz.model.modules.grouped_msa import _normalize
    if torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 12 * 1024**3:
        pytest.skip("Index-boundary test needs 12 GiB GPU memory")
    # The final tile starts beyond 2**31 elements, while the row count fits int32.
    x = torch.zeros((2**25 + 16, 64), device="cuda", dtype=torch.bfloat16)
    x[-16:].normal_()
    norm = torch.nn.LayerNorm(64).cuda().eval()
    norm.weight.uniform_(.7, 1.3)
    norm.bias.uniform_(-.1, .1)
    actual = _normalize(x, norm, torch.bfloat16)
    expected = norm(x[-16:].float()).to(torch.bfloat16)
    torch.testing.assert_close(actual[-16:], expected, atol=.008, rtol=.008)
    torch.testing.assert_close(actual[:16], norm.bias.to(torch.bfloat16).expand(16, -1), atol=0, rtol=0)
