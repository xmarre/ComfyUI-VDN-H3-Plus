import torch

from vdn_h3 import retained, window
from vdn_h3.softmax_provider import (
    KEY,
    KEY_V2,
    KEY_V3,
    PREPROCESS_KEY,
    PROVIDER_API_VERSION,
    dispatch,
    preprocess,
)


def test_grouped_v1_provider_receives_only_existing_window_domains():
    torch.manual_seed(7)
    q, k, v = (torch.randn(13, 2, 4) for _ in range(3))
    bounds = window.window_bounds(4, 0, 2)
    observed = []
    def provider(native, q, k, v, **contract):
        observed.append((q.shape[0], k.shape[0], contract))
        return native()
    def forbidden(*a, **kw):
        raise AssertionError("model-level dense override leaked into VDN local attention")
    options = {KEY: provider, "optimized_attention_override": forbidden}
    for anchor in ("none", "rows", "columns", "both"):
        got = retained.window_softmax_grouped_runtime(q, k, v, 2, 10, 4, 2, bounds, .5,
                                                     anchor_frames=anchor, transformer_options=options)
        want = window.window_softmax_grouped(q, k, v, 2, 10, 4, 2, bounds, .5, anchor_frames=anchor)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert any(c[2]["kind"] == "local" and c[0] < c[1] for c in observed)
    assert any(c[2]["kind"] == "global" for c in observed)
    assert any(c[2]["kind"] == "anchor" for c in observed)
    assert not any(c[2]["square_aligned"] for c in observed)


def test_square_local_v1_contract_requires_matching_row_domain():
    q, k, v = (torch.randn(8, 2, 4) for _ in range(3))
    bounds = window.window_bounds(4, 0, 2)
    seen = []
    def provider(native, q, k, v, **contract):
        seen.append(contract)
        assert contract["square_aligned"]
        assert q.shape == k.shape == v.shape == (4, 2, 4)
        return native()
    got = retained.window_softmax_grouped_runtime(q, k, v, 0, 8, 4, 2, bounds, .5,
                                                 transformer_options={KEY: provider})
    want = window.window_softmax_grouped(q, k, v, 0, 8, 4, 2, bounds, .5)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert len(seen) == 2


def test_v2_square_domain_matches_native_rectangular_attention():
    torch.manual_seed(17)
    q, k, v = (torch.randn(13, 2, 4) for _ in range(3))
    bounds = window.window_bounds(4, 0, 2)
    seen = []

    def provider(native, q, k, v, **contract):
        if contract["kind"] != "local":
            assert contract["square_q"] is None
            return native()
        square_q = contract["square_q"]
        positions = contract["query_positions"]
        assert square_q.shape == k.shape == v.shape
        assert positions.dtype == torch.long and positions.device == q.device
        torch.testing.assert_close(square_q.index_select(0, positions), q, rtol=0, atol=0)
        assert contract["sink_rows"] == 5
        seen.append((q.shape[0], square_q.shape[0]))
        square_out = window._sdpa(square_q, k, v, contract["scale"], None)
        return square_out.index_select(0, positions)

    got = retained.window_softmax_grouped_runtime(
        q, k, v, 2, 10, 4, 2, bounds, .5,
        transformer_options={KEY_V2: provider})
    want = window.window_softmax_grouped(q, k, v, 2, 10, 4, 2, bounds, .5)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    assert seen and all(requested < square for requested, square in seen)


def test_v3_direct_rectangular_domain_avoids_square_payload():
    assert PROVIDER_API_VERSION == 3
    torch.manual_seed(19)
    q, k, v = (torch.randn(13, 2, 4) for _ in range(3))
    bounds = window.window_bounds(4, 0, 2)
    seen = []

    def provider(native, q, k, v, **contract):
        assert "square_q" not in contract
        assert "query_positions" not in contract
        if contract["kind"] == "local":
            assert q.shape[0] < k.shape[0]
            assert contract["sink_rows"] == 5
            seen.append((q.shape[0], k.shape[0]))
        return native()

    got = retained.window_softmax_grouped_runtime(
        q, k, v, 2, 10, 4, 2, bounds, .5,
        transformer_options={KEY_V3: provider})
    want = window.window_softmax_grouped(q, k, v, 2, 10, 4, 2, bounds, .5)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert seen and all(requested < restricted_kv for requested, restricted_kv in seen)


def test_provider_precedence_v3_then_v2_then_v1():
    q = torch.randn(4, 2, 4)
    calls = []
    def v1(native, q, k, v, **contract):
        calls.append("v1")
        return native()
    def v2(native, q, k, v, **contract):
        calls.append("v2")
        return native()
    def v3(native, q, k, v, **contract):
        calls.append("v3")
        return native()
    dispatch({KEY: v1, KEY_V2: v2, KEY_V3: v3}, lambda: q, q, q, q, kind="local", scale=.5)
    assert calls == ["v3"]
    calls.clear()
    dispatch({KEY: v1, KEY_V2: v2}, lambda: q, q, q, q, kind="local", scale=.5)
    assert calls == ["v2"]
    calls.clear()
    dispatch({KEY: v1}, lambda: q, q, q, q, kind="local", scale=.5)
    assert calls == ["v1"]


def test_full_domain_preprocess_is_shape_preserving_and_explicit():
    q, k, v = (torch.randn(8, 2, 4) for _ in range(3))
    original_k = k.clone()
    calls = []
    def transform(q, k, v, *, heads, transformer_options):
        calls.append((heads, transformer_options["tag"]))
        return q, k * 2, v
    q2, k2, v2 = preprocess({PREPROCESS_KEY: transform, "tag": 9}, q, k, v, 2)
    assert q2 is q and v2 is v
    torch.testing.assert_close(k2, original_k * 2)
    assert calls == [(2, 9)]


def test_flex_contract_keeps_native_masked_operator():
    q = torch.randn(8, 2, 4)
    calls = []
    def provider(native, q, k, v, **contract):
        assert contract == {"kind": "flex_masked", "scale": .5, "square_aligned": False}
        return native()
    def native():
        calls.append(True)
        return q + 1
    got = dispatch({KEY: provider}, native, q, q, q, kind="flex_masked", scale=.5)
    assert torch.equal(got, q + 1) and len(calls) == 1
