import torch

from vdn_h3 import retained, window
from vdn_h3.softmax_provider import dispatch, KEY


def test_grouped_provider_receives_only_existing_window_domains():
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


def test_square_local_contract_requires_matching_row_domain():
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
