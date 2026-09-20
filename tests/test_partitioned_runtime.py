from types import SimpleNamespace

from vdn_h3.partitioned_runtime import (
    VDN_EXTERNAL_SEQUENCE_KEY,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_BYPASS,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_KEY,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_NORMAL,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_OPTIONS,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
    _mixed_layout_matches_plan,
    _partitioned_linear_diagnostic_mode,
    _partitioned_query_summary,
    _record_partitioned_cross_grid_temporal_suppression,
    _record_partitioned_linear_bypass,
    _resolve_partitioned_linear_runtime,
    _wrap_vdn_forward,
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



def test_partitioned_linear_diagnostic_defaults_to_normal_and_is_partition_scoped():
    layout = SimpleNamespace(full_cover=False)
    cfg = {"linear_enabled": True}

    active, bypassed, mode = _resolve_partitioned_linear_runtime(layout, cfg, {})
    assert active is True
    assert bypassed is False
    assert mode == VDN_PARTITIONED_LINEAR_DIAGNOSTIC_NORMAL

    active, bypassed, mode = _resolve_partitioned_linear_runtime(
        layout,
        cfg,
        {
            VDN_PARTITIONED_LINEAR_DIAGNOSTIC_KEY: VDN_PARTITIONED_LINEAR_DIAGNOSTIC_BYPASS,
        },
    )
    assert active is False
    assert bypassed is True
    assert mode == VDN_PARTITIONED_LINEAR_DIAGNOSTIC_BYPASS

    active, bypassed, mode = _resolve_partitioned_linear_runtime(
        layout,
        cfg,
        {
            VDN_PARTITIONED_LINEAR_DIAGNOSTIC_KEY:
                VDN_PARTITIONED_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
        },
    )
    assert active is True
    assert bypassed is False
    assert mode == VDN_PARTITIONED_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL


def test_partitioned_linear_bypass_does_not_invent_a_branch_when_released_linear_is_inactive():
    options = {
        VDN_PARTITIONED_LINEAR_DIAGNOSTIC_KEY: VDN_PARTITIONED_LINEAR_DIAGNOSTIC_BYPASS,
    }

    active, bypassed, mode = _resolve_partitioned_linear_runtime(
        SimpleNamespace(full_cover=True),
        {"linear_enabled": True},
        options,
    )
    assert active is False
    assert bypassed is False
    assert mode == VDN_PARTITIONED_LINEAR_DIAGNOSTIC_BYPASS

    active, bypassed, mode = _resolve_partitioned_linear_runtime(
        SimpleNamespace(full_cover=False),
        {"linear_enabled": False},
        options,
    )
    assert active is False
    assert bypassed is False
    assert mode == VDN_PARTITIONED_LINEAR_DIAGNOSTIC_BYPASS


def test_partitioned_linear_diagnostic_rejects_unknown_mode():
    import pytest

    with pytest.raises(RuntimeError, match="partitioned VDN linear diagnostic mode"):
        _partitioned_linear_diagnostic_mode(
            {VDN_PARTITIONED_LINEAR_DIAGNOSTIC_KEY: "silently_change_vdn"}
        )



def test_partitioned_linear_bypass_records_explicit_flow_metrics():
    class Metrics:
        def __init__(self):
            self.values = {}

        def increment(self, name, value=1):
            self.values[name] = self.values.get(name, 0) + value

    metrics = Metrics()
    options = {
        "h3_flow_partitioned_stage_v1": SimpleNamespace(metrics=metrics),
    }

    _record_partitioned_linear_bypass(options, 1234)
    _record_partitioned_linear_bypass(options, 1234)

    assert metrics.values["partitioned_vdn_linear_bypass_calls"] == 2
    assert metrics.values["partitioned_vdn_linear_bypass_video_rows"] == 2468


def test_partitioned_cross_grid_temporal_suppression_records_explicit_flow_metrics():
    class Metrics:
        def __init__(self):
            self.values = {}

        def increment(self, name, value=1):
            self.values[name] = self.values.get(name, 0) + value

    metrics = Metrics()
    options = {
        "h3_flow_partitioned_stage_v1": SimpleNamespace(metrics=metrics),
    }
    _record_partitioned_cross_grid_temporal_suppression(
        options,
        {"suppressed_taps": 7, "suppressed_rows": 1234},
    )
    assert metrics.values["partitioned_vdn_cross_grid_temporal_suppression_calls"] == 1
    assert metrics.values["partitioned_vdn_cross_grid_temporal_suppressed_taps"] == 7
    assert metrics.values["partitioned_vdn_cross_grid_temporal_suppressed_rows"] == 1234


def test_partitioned_linear_bridge_publishes_diagnostic_capability_api():
    base_branch = SimpleNamespace()
    block_index = 0
    cfg = {}
    head_dim = 128
    heads = 56
    k_norm = SimpleNamespace()
    out_proj = SimpleNamespace()
    q_norm = SimpleNamespace()
    qkv_proj = SimpleNamespace()
    state = SimpleNamespace()

    def current(x, rope_freqs=None, transformer_options=None):
        _ = (base_branch, block_index, cfg, head_dim, heads, k_norm, out_proj, q_norm, qkv_proj, state)
        return x

    current._vdn_forward = True
    wrapped = _wrap_vdn_forward(current)
    assert VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API == 1
    assert wrapped._vdn_partitioned_linear_diagnostic_api == VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API
    assert tuple(wrapped._vdn_partitioned_linear_diagnostic_modes) == VDN_PARTITIONED_LINEAR_DIAGNOSTIC_OPTIONS
    assert VDN_PARTITIONED_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL in wrapped._vdn_partitioned_linear_diagnostic_modes
