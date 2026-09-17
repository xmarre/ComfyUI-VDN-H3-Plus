from types import SimpleNamespace

from vdn_h3.partitioned_runtime import (
    VDN_EXTERNAL_SEQUENCE_KEY,
    _mixed_layout_matches_plan,
    _partitioned_query_summary,
)
from vdn_h3.partitioned_sequence import (
    PARTITIONED_PREFIX_KEY,
    PartitionedSequence,
    make_vdn_partitioned_external_contract,
)


def _fixture():
    plan = PartitionedSequence(
        video_start=7,
        temporal=5,
        prefix_t=2,
        source_grid_h=2,
        source_grid_w=3,
        target_grid_h=3,
        target_grid_w=4,
    )
    contract = plan.canonical_contract()
    options = {
        PARTITIONED_PREFIX_KEY: contract,
        VDN_EXTERNAL_SEQUENCE_KEY: make_vdn_partitioned_external_contract(plan),
    }
    layout = SimpleNamespace(
        seq_len=plan.sequence_rows,
        segments=[(0, plan.video_start, "nonvideo"), (plan.video_start, plan.sequence_rows, "video")],
        signature=(PARTITIONED_PREFIX_KEY, "test"),
    )
    state = SimpleNamespace(
        softmax_backend="grouped",
        query_position_owner_generation="vdn-test-owner",
    )
    values = {
        "state": state,
        "cfg": {"radius": 1, "chunk": 1, "anchor_frames": "none"},
    }
    return plan, options, layout, values


def test_partitioned_mixed_layout_must_match_physical_sequence():
    plan, _options, layout, _values = _fixture()
    assert _mixed_layout_matches_plan(layout, plan)

    stale = SimpleNamespace(
        seq_len=plan.sequence_rows - 1,
        segments=layout.segments,
        signature=layout.signature,
    )
    assert not _mixed_layout_matches_plan(stale, plan)


def test_partitioned_query_summary_fails_closed_on_backend_layout_or_external_drift():
    plan, options, layout, values = _fixture()
    current = SimpleNamespace()
    summary = _partitioned_query_summary(current, values, options, layout)
    assert summary is not None
    assert summary.mode == "grouped"
    assert summary.seq_len == plan.sequence_rows
    assert summary.groups

    stale_layout = SimpleNamespace(
        seq_len=plan.sequence_rows - 1,
        segments=layout.segments,
        signature=layout.signature,
    )
    assert _partitioned_query_summary(current, values, options, stale_layout) is None

    values["state"].softmax_backend = "flex"
    assert _partitioned_query_summary(current, values, options, layout) is None
    values["state"].softmax_backend = "grouped"

    bad_options = {
        **options,
        VDN_EXTERNAL_SEQUENCE_KEY: {
            **options[VDN_EXTERNAL_SEQUENCE_KEY],
            "flow_semantic_digest": "f" * 64,
        },
    }
    assert _partitioned_query_summary(current, values, bad_options, layout) is None
