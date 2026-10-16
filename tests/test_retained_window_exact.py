from __future__ import annotations

import torch

from vdn_h3 import retained, window


def test_retained_windows_do_not_forward_transformer_options(monkeypatch):
    """Retained grouped windows must preserve released exact-SDPA semantics."""
    original = window._sdpa
    seen_options = []

    def recording_sdpa(q, k, v, scale, transformer_options=None):
        seen_options.append(transformer_options)
        return original(q, k, v, scale, transformer_options)

    monkeypatch.setattr(window, "_sdpa", recording_sdpa)

    torch.manual_seed(520)
    video_start, tokens, frames, heads, dim = 3, 4, 6, 2, 8
    video_end = video_start + frames * tokens
    seq = video_end + 2
    q = torch.randn(seq, heads, dim)
    k = torch.randn(seq, heads, dim)
    v = torch.randn(seq, heads, dim)
    bounds = window.window_bounds(frames, 1, 3)
    override = {"optimized_attention_override": object(), "sentinel": True}

    got = retained.window_softmax_grouped_runtime(
        q,
        k,
        v,
        video_start,
        video_end,
        frames,
        tokens,
        bounds,
        dim ** -0.5,
        anchor_frames="both",
        transformer_options=override,
    )

    assert seen_options
    assert all(option is None for option in seen_options)

    want = window.window_softmax_grouped(
        q,
        k,
        v,
        video_start,
        video_end,
        frames,
        tokens,
        bounds,
        dim ** -0.5,
        anchor_frames="both",
        transformer_options=None,
    )
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)


def test_grouped_contiguous_query_views_and_noncontiguous_fallback_match_reference():
    from vdn_h3.query_positions import describe_window_geometry

    torch.manual_seed(419)
    frames, tokens, heads, dim, start = 6, 4, 2, 8, 3
    end = start + frames * tokens
    seq = end + 2
    # Frames 0,1,3 share bounds: that group is NOT consecutive.
    # Frames 4,5 form separate contiguous spans.
    bounds = ((0, 2), (0, 2), (1, 3), (0, 2), (3, 5), (3, 5))
    plan = retained._build_window_plan(
        start, end, frames, tokens, bounds, "none", seq, torch.device("cpu"),
    )
    geom = describe_window_geometry(start, end, frames, tokens, bounds, "none", seq)
    assert geom.groups[0].query_frames == (0, 1, 3)
    assert plan["contiguous_query_spans"][0] is None
    assert plan["contiguous_query_spans"][-1] == (start + 4*tokens, end)
    for noncontiguous in (False, True):
        if noncontiguous:
            q = torch.randn(seq, heads, dim * 2)[..., ::2]
            assert not q.is_contiguous()
        else:
            q = torch.randn(seq, heads, dim)
        k = torch.randn(seq, heads, dim)
        v = torch.randn(seq, heads, dim)
        got = retained.window_softmax_grouped_runtime(
            q, k, v, start, end, frames, tokens, bounds, dim ** -0.5,
            anchor_frames="none",
        )
        want = window.window_softmax_grouped(
            q, k, v, start, end, frames, tokens, bounds, dim ** -0.5,
            anchor_frames="none",
        )
        torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)


def test_untrusted_v4_provider_receives_isolated_query_copy():
    from vdn_h3.softmax_provider import KEY_V4

    frames, tokens, heads, dim, start = 6, 4, 2, 8, 3
    end = start + frames * tokens
    seq = end + 2
    q = torch.randn(seq, heads, dim)
    original = q.clone()
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bounds = ((0, 2), (0, 2), (1, 3), (0, 2), (3, 5), (3, 5))
    mutated_groups = []

    def mutating_provider(native, local_q, local_k, local_v, **kwargs):
        del local_k, local_v
        if kwargs["kind"] == "local":
            mutated_groups.append(local_q.shape[0])
            local_q.add_(0.25)
        return native()

    retained.window_softmax_grouped_runtime(
        q, k, v, start, end, frames, tokens, bounds, dim**-0.5,
        transformer_options={KEY_V4: mutating_provider},
    )
    assert mutated_groups
    # Untrusted providers still get independent indexed Q arrays and cannot
    # corrupt the full Q buffer used by neighboring groups or future calls.
    assert torch.equal(q, original)
