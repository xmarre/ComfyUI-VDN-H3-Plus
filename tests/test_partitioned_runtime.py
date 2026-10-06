from types import SimpleNamespace

from vdn_h3.partitioned_runtime import (
    VDN_EXTERNAL_SEQUENCE_KEY,
    VDN_PARTITIONED_BOUNDARY_QUERY_API,
    VDN_PARTITIONED_BOUNDARY_QUERY_POLICY,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_BYPASS,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_KEY,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_NORMAL,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_OPTIONS,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
    VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_API,
    VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_DENSE_SUFFIX,
    VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY,
    VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
    VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_OPTIONS,
    VDN_TEMPORAL_CARRIER_API,
    VDN_TEMPORAL_CARRIER_DESTINATION,
    VDN_TEMPORAL_CARRIER_KEY,
    VDN_TEMPORAL_CARRIER_NATIVE,
    VDN_TEMPORAL_CARRIER_POLICIES,
    _mixed_layout_matches_plan,
    _partitioned_linear_diagnostic_mode,
    _partitioned_local_force_dense,
    _partitioned_softmax_diagnostic_mode,
    _partitioned_query_summary,
    _record_partitioned_boundary_suffix_dense,
    _record_partitioned_cross_grid_temporal_suppression,
    _record_partitioned_dense_suffix_same_domain,
    _record_partitioned_linear_bypass,
    _record_partitioned_raw_token_measure,
    _resolve_partitioned_linear_runtime,
    _resolve_temporal_carrier_policy,
    _temporal_carrier_short_conv_spec,
    _wrap_vdn_forward,
)
from vdn_h3.partitioned_sequence import (
    PARTITIONED_PREFIX_KEY,
    PartitionedSequence,
    make_temporal_carrier_contract,
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

    active, bypassed, mode = _resolve_partitioned_linear_runtime(
        layout,
        cfg,
        {
            VDN_PARTITIONED_LINEAR_DIAGNOSTIC_KEY:
                VDN_PARTITIONED_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE,
        },
    )
    assert active is True
    assert bypassed is False
    assert mode == VDN_PARTITIONED_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE


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


def test_partitioned_raw_token_measure_records_explicit_flow_metrics():
    class Metrics:
        def __init__(self):
            self.values = {}

        def increment(self, name, value=1):
            self.values[name] = self.values.get(name, 0) + value

    metrics = Metrics()
    options = {
        "h3_flow_partitioned_stage_v1": SimpleNamespace(metrics=metrics),
    }
    plan = SimpleNamespace(prefix_log_key_measure=-0.678, prefix_t=12)
    _record_partitioned_raw_token_measure(options, plan)
    assert metrics.values["partitioned_vdn_raw_token_measure_calls"] == 1
    assert metrics.values["partitioned_vdn_raw_token_measure_prefix_frames"] == 12


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
    base_branch = SimpleNamespace(
        short_conv=("k", "v"),
        delta_rule="vdn_solve",
        num_heads=56,
        head_dim=128,
        a_fp32=True,
    )
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
    assert VDN_PARTITIONED_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE in wrapped._vdn_partitioned_linear_diagnostic_modes
    assert wrapped._vdn_partitioned_temporal_carrier_api == VDN_TEMPORAL_CARRIER_API
    assert tuple(wrapped._vdn_partitioned_temporal_carrier_policies) == VDN_TEMPORAL_CARRIER_POLICIES
    assert wrapped._vdn_partitioned_temporal_carrier_short_conv_spec == _temporal_carrier_short_conv_spec(base_branch)


def test_native_bridge_install_does_not_require_destination_stencil_capability():
    base_branch = SimpleNamespace(
        short_conv=(),
        delta_rule="different_rule",
        num_heads=56,
        head_dim=128,
        a_fp32=True,
    )
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
    assert tuple(wrapped._vdn_partitioned_temporal_carrier_policies) == (VDN_TEMPORAL_CARRIER_NATIVE,)
    assert wrapped._vdn_partitioned_temporal_carrier_short_conv_spec is None


def test_destination_temporal_carrier_contract_binds_plan_diagnostic_and_checkpoint_spec():
    plan, _options, _layout, _values = _fixture()
    branch = SimpleNamespace(
        short_conv=("k", "v"),
        delta_rule="vdn_solve",
        num_heads=56,
        head_dim=128,
        a_fp32=True,
    )
    spec = _temporal_carrier_short_conv_spec(branch)
    flow_digest = plan.canonical_contract()["semantic_digest"]
    contract = make_temporal_carrier_contract(
        policy=VDN_TEMPORAL_CARRIER_DESTINATION,
        flow_semantic_digest=flow_digest,
        diagnostic_mode=VDN_PARTITIONED_LINEAR_DIAGNOSTIC_NORMAL,
        short_conv_spec=spec,
    )

    policy, validated, resolved_spec = _resolve_temporal_carrier_policy(
        {VDN_TEMPORAL_CARRIER_KEY: contract},
        plan,
        VDN_PARTITIONED_LINEAR_DIAGNOSTIC_NORMAL,
        branch,
    )
    assert policy == VDN_TEMPORAL_CARRIER_DESTINATION
    assert validated == contract
    assert resolved_spec == spec
    assert len(contract["numerical_digest"]) == 64

    native, absent, _ = _resolve_temporal_carrier_policy(
        {},
        plan,
        VDN_PARTITIONED_LINEAR_DIAGNOSTIC_NORMAL,
        branch,
    )
    assert native == VDN_TEMPORAL_CARRIER_NATIVE
    assert absent is None

    tampered = dict(contract)
    tampered["numerical_digest"] = "0" * 64
    import pytest

    with pytest.raises(RuntimeError, match="numerical policy"):
        _resolve_temporal_carrier_policy(
            {VDN_TEMPORAL_CARRIER_KEY: tampered},
            plan,
            VDN_PARTITIONED_LINEAR_DIAGNOSTIC_NORMAL,
            branch,
        )
    with pytest.raises(RuntimeError, match="requires vdn_linear_diagnostic='normal'"):
        suppression_contract = make_temporal_carrier_contract(
            policy=VDN_TEMPORAL_CARRIER_DESTINATION,
            flow_semantic_digest=flow_digest,
            diagnostic_mode=VDN_PARTITIONED_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
            short_conv_spec=spec,
        )
        _resolve_temporal_carrier_policy(
            {VDN_TEMPORAL_CARRIER_KEY: suppression_contract},
            plan,
            VDN_PARTITIONED_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
            branch,
        )


def test_partitioned_softmax_policy_keeps_boundary_suffix_group_dense_and_later_suffix_sparse():
    assert _partitioned_softmax_diagnostic_mode({}) == VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL
    selected = {
        VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY:
            VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_DENSE_SUFFIX,
    }
    assert (
        _partitioned_softmax_diagnostic_mode(selected)
        == VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_DENSE_SUFFIX
    )

    prefix_t = 12
    prefix = SimpleNamespace(query_prefix_domain=True, query_frames=(10, 11))
    boundary_suffix = SimpleNamespace(
        query_prefix_domain=False,
        query_frames=(12, 13, 14),
    )
    later_suffix = SimpleNamespace(
        query_prefix_domain=False,
        query_frames=(15, 16, 17, 18, 19),
    )

    assert _partitioned_local_force_dense(
        prefix,
        VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
        prefix_t=prefix_t,
    ) == (True, False, False)
    assert _partitioned_local_force_dense(
        boundary_suffix,
        VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
        prefix_t=prefix_t,
    ) == (True, False, True)
    assert _partitioned_local_force_dense(
        later_suffix,
        VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
        prefix_t=prefix_t,
    ) == (False, False, False)
    assert _partitioned_local_force_dense(
        boundary_suffix,
        VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_DENSE_SUFFIX,
        prefix_t=prefix_t,
    ) == (True, False, True)
    assert _partitioned_local_force_dense(
        later_suffix,
        VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_DENSE_SUFFIX,
        prefix_t=prefix_t,
    ) == (True, True, False)


def test_partitioned_softmax_diagnostic_rejects_unknown_mode():
    import pytest

    with pytest.raises(RuntimeError, match="partitioned VDN softmax diagnostic mode"):
        _partitioned_softmax_diagnostic_mode(
            {VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY: "invalid"}
        )


def test_high_keeps_free_target_band_queries_dense_without_changing_groups():
    mode = VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL
    for frames, expected in [((12, 13, 14), (True, False, True)),
                             ((15, 16, 17), (True, False, False)),
                             ((18, 19, 20), (False, False, False))]:
        group = SimpleNamespace(query_prefix_domain=False, query_frames=frames)
        assert _partitioned_local_force_dense(group, mode, prefix_t=12, attention_head_t=16) == expected
        assert group.query_frames == frames and group.query_prefix_domain is False


def test_partitioned_boundary_suffix_dense_records_explicit_flow_metrics_and_receipt():
    class Metrics:
        def __init__(self):
            self.values = {}
            self.events = []

        def increment(self, name, value=1):
            self.values[name] = self.values.get(name, 0) + value

        def event(self, kind, **fields):
            self.events.append((kind, fields))

    metrics = Metrics()
    options = {
        "h3_flow_partitioned_stage_v1": SimpleNamespace(metrics=metrics),
        "h3_flow_stage": "high",
    }
    group = SimpleNamespace(
        query_prefix_domain=False,
        query_frames=(12, 13, 14),
        q_rows=2850,
        kv_rows=15250,
    )
    _record_partitioned_boundary_suffix_dense(
        options,
        group=group,
        prefix_t=12,
        block_index=0,
        softmax_diagnostic_mode=VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
    )
    _record_partitioned_boundary_suffix_dense(
        options,
        group=group,
        prefix_t=12,
        block_index=1,
        softmax_diagnostic_mode=VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
    )

    assert metrics.values["partitioned_vdn_boundary_suffix_dense_calls"] == 2
    assert metrics.values["partitioned_vdn_boundary_suffix_dense_q_rows"] == 5700
    assert metrics.values["partitioned_vdn_boundary_suffix_dense_kv_rows"] == 30500
    assert metrics.values["partitioned_vdn_boundary_suffix_dense_query_frames"] == 6
    assert len(metrics.events) == 1
    kind, fields = metrics.events[0]
    assert kind == "partitioned_vdn_boundary_suffix_dense"
    assert fields["policy"] == VDN_PARTITIONED_BOUNDARY_QUERY_POLICY
    assert fields["stage"] == "high"
    assert fields["prefix_t"] == 12
    assert fields["query_frames"] == (12, 13, 14)
    assert fields["grouped_qkv_unchanged"] is True
    assert fields["prefix_measure_unchanged"] is True
    assert fields["softmax_diagnostic_mode"] == VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL
    assert fields["later_suffix_sparse"] is True
    assert fields["extra_model_calls"] == 0


def test_partitioned_dense_suffix_records_explicit_flow_metrics():
    class Metrics:
        def __init__(self):
            self.values = {}

        def increment(self, name, value=1):
            self.values[name] = self.values.get(name, 0) + value

    metrics = Metrics()
    options = {
        "h3_flow_partitioned_stage_v1": SimpleNamespace(metrics=metrics),
    }
    _record_partitioned_dense_suffix_same_domain(options, q_rows=13, kv_rows=31)
    _record_partitioned_dense_suffix_same_domain(options, q_rows=17, kv_rows=37)
    assert metrics.values["partitioned_vdn_dense_suffix_same_domain_calls"] == 2
    assert metrics.values["partitioned_vdn_dense_suffix_same_domain_q_rows"] == 30
    assert metrics.values["partitioned_vdn_dense_suffix_same_domain_kv_rows"] == 68


def test_partitioned_softmax_bridge_publishes_diagnostic_capability_api():
    base_branch = SimpleNamespace(
        short_conv=("k", "v"),
        delta_rule="vdn_solve",
        num_heads=56,
        head_dim=128,
        a_fp32=True,
    )
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
    assert VDN_PARTITIONED_BOUNDARY_QUERY_API == 1
    assert wrapped._vdn_partitioned_boundary_query_api == VDN_PARTITIONED_BOUNDARY_QUERY_API
    assert wrapped._vdn_partitioned_boundary_query_policy == VDN_PARTITIONED_BOUNDARY_QUERY_POLICY
    assert VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_API == 1
    assert (
        wrapped._vdn_partitioned_softmax_diagnostic_api
        == VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_API
    )
    assert tuple(wrapped._vdn_partitioned_softmax_diagnostic_modes) == VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_OPTIONS
    assert VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_DENSE_SUFFIX in wrapped._vdn_partitioned_softmax_diagnostic_modes
