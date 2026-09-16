from types import SimpleNamespace

import torch

from vdn_h3.query_positions import (
    WIRE_TAG,
    bind_query_map,
    describe_window_geometry,
    native_plan_summary,
)
from vdn_h3.softmax_provider import KEY_V3, KEY_V4, dispatch, has_v4


def _bounds(frames=52, radius=1, chunk=5):
    return tuple(
        (((frame // chunk) - radius) * chunk, ((frame // chunk) + radius + 1) * chunk - 1)
        for frame in range(frames)
    )


def _expand_runs(wire):
    positions = []
    for q_begin, q_end, kv_begin in wire[-1]:
        assert q_begin == len(positions)
        positions.extend(kv_begin + i - q_begin for i in range(q_begin, q_end))
    return positions


def test_geometry_map_is_the_exact_restricted_domain_mapping():
    # Small nontrivial packed layout with globals on both sides and endpoint anchors.
    geometry = describe_window_geometry(
        7, 7 + 12 * 6, 12, 6, _bounds(12), "both", 84
    )
    assert geometry.global_rows == tuple(range(7)) + tuple(range(79, 84))
    assert len(geometry.groups) == 3
    for group in geometry.groups:
        wire = bind_query_map(geometry, group.group_index, "owner-test")
        assert wire[0] == WIRE_TAG
        positions = _expand_runs(wire)
        assert len(positions) == group.q_rows

        domain_rows = list(geometry.global_rows)
        for frame in group.key_frames:
            start = geometry.video_start + frame * geometry.tokens_per_frame
            domain_rows.extend(range(start, start + geometry.tokens_per_frame))
        requested_rows = []
        for frame in group.query_frames:
            start = geometry.video_start + frame * geometry.tokens_per_frame
            requested_rows.extend(range(start, start + geometry.tokens_per_frame))
        assert [domain_rows[position] for position in positions] == requested_rows
        assert all(a < b for a, b in zip(positions, positions[1:]))


def test_group10_shape_and_affine_mapping_for_production_style_52_frames():
    # The exact token count is intentionally arbitrary here. The contract must be
    # scale-independent and the terminal chunk must remain represented exactly.
    geometry = describe_window_geometry(
        9, 9 + 52 * 17, 52, 17, _bounds(), "both", 900
    )
    assert len(geometry.groups) == 11
    group10 = geometry.groups[10]
    wire = bind_query_map(geometry, 10, "owner-production-shape")
    positions = _expand_runs(wire)
    assert len(positions) == group10.q_rows
    assert positions[0] >= group10.sink_rows
    assert positions[-1] < group10.kv_rows


def test_native_plan_summary_uses_supplied_layout_not_runtime_state():
    text_len, frames, lat_h, lat_w, audio_t = 5, 12, 4, 6, 8
    per_frame = (lat_h // 2) * (lat_w // 2)
    audio_start = text_len
    video_start = audio_start + audio_t
    video_end = video_start + frames * per_frame
    layout = SimpleNamespace(
        signature=(text_len, frames, lat_h, lat_w, audio_t),
        seq_len=video_end,
        segments=(
            (0, text_len, "text"),
            (audio_start, video_start, "audio"),
            (video_start, video_end, "video"),
        ),
    )
    summary = native_plan_summary(
        layout,
        {"radius": 1, "chunk": 5, "anchor_frames": "both"},
        "owner-summary",
    )
    assert summary.owner_generation == "owner-summary"
    assert summary.seq_len == video_end
    assert len(summary.groups) == 3
    assert all(group[2] == "owner-summary" for group in summary.groups)


def test_v4_has_priority_and_malformed_presence_fails_to_native_not_v3():
    q = torch.zeros((2, 1, 4))
    calls = []

    def native():
        calls.append("native")
        return q

    def v3(*args, **kwargs):
        calls.append("v3")
        return q

    assert has_v4({KEY_V4: None})
    result = dispatch(
        {KEY_V4: None, KEY_V3: v3}, native, q, q, q,
        kind="local", scale=0.5, query_position_map=None,
    )
    assert result is q
    assert calls == ["native"]


def test_callable_v4_receives_map_without_square_payload():
    q = torch.zeros((2, 1, 4))
    seen = {}
    marker = ("vdn_query_positions", 1, "owner", "0" * 64, 0, 2, 3, 1, ((0, 2, 1),))

    def native():
        raise AssertionError("native should not run")

    def v4(native_cb, got_q, got_k, got_v, **kwargs):
        seen.update(kwargs)
        assert got_q is q and got_k is q and got_v is q
        return got_q

    assert dispatch(
        {KEY_V4: v4}, native, q, q, q,
        kind="local", scale=0.5, query_position_map=marker,
    ) is q
    assert seen["query_position_map"] == marker
    assert "square_q" not in seen and "query_positions" not in seen
