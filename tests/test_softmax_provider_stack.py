import torch

from vdn_h3 import retained, window
from vdn_h3.softmax_provider import PREPROCESS_KEY


def test_grouped_runtime_preprocesses_full_domain_before_gather():
    torch.manual_seed(29)
    q, k, v = (torch.randn(13, 2, 4) for _ in range(3))
    bounds = window.window_bounds(4, 0, 2)
    seen = []

    def transform(q, k, v, *, heads, transformer_options):
        seen.append((q.shape, k.shape, v.shape, heads))
        return q + 0.25, k, v

    got = retained.window_softmax_grouped_runtime(
        q, k, v, 2, 10, 4, 2, bounds, 0.5,
        anchor_frames="both",
        transformer_options={PREPROCESS_KEY: transform},
    )
    want = retained.window_softmax_grouped_runtime(
        q + 0.25, k, v, 2, 10, 4, 2, bounds, 0.5,
        anchor_frames="both",
        transformer_options=None,
    )

    assert seen == [((13, 2, 4), (13, 2, 4), (13, 2, 4), 2)]
    torch.testing.assert_close(got, want, rtol=0, atol=0)
