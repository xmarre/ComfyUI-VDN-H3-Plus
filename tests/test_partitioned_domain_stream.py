"""Flow domain-uniform streams: VDN derives each stream's window layout from its contract."""
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

import comfy.quant_ops
from vdn_h3 import partitioned_linear, partitioned_runtime
from vdn_h3.hybrid import VDNLayout, VDNState
from vdn_h3.partitioned_sequence import (
    PARTITIONED_PREFIX_KEY,
    PartitionedSequence,
    make_vdn_partitioned_external_contract,
)

CFG = {"radius": 1, "chunk": 1, "anchor_frames": "rows", "enable_softmax_gate": False, "linear_enabled": True}
TEXT_START, TEXT_LEN, VIDEO_START = 1, 2, 7
NATIVE_FRAMES, NATIVE_ROWS_PER_FRAME = 6, 12


def _native_layout():
    return VDNLayout(
        video_start=VIDEO_START,
        video_end=VIDEO_START + NATIVE_FRAMES * NATIVE_ROWS_PER_FRAME,
        num_frames=NATIVE_FRAMES,
        tokens_per_frame=NATIVE_ROWS_PER_FRAME,
        frame_size=(3, 4),
        text_start=TEXT_START,
        text_len=TEXT_LEN,
        seq_len=VIDEO_START + NATIVE_FRAMES * NATIVE_ROWS_PER_FRAME,
        radius=1,
        chunk=1,
        anchor_frames="rows",
    )


def _stream(name):
    if name == "target":
        # Head clip: 4 target-grid frames, shorter conditioning audio.
        plan = PartitionedSequence(
            video_start=5, temporal=4, prefix_t=2,
            source_grid_h=3, source_grid_w=4, target_grid_h=3, target_grid_w=4,
        )
    else:
        plan = PartitionedSequence(
            video_start=VIDEO_START, temporal=NATIVE_FRAMES, prefix_t=2,
            source_grid_h=2, source_grid_w=3, target_grid_h=2, target_grid_w=3,
        )
    contract = plan.canonical_contract()
    leaf = {
        "api": 1,
        "policy": "domain_uniform_v1",
        "stream": name,
        "flow_semantic_digest": contract["semantic_digest"],
        "native_sequence_rows": _native_layout().seq_len,
        "native_video_start": VIDEO_START,
    }
    options = {
        PARTITIONED_PREFIX_KEY: contract,
        partitioned_runtime.VDN_EXTERNAL_SEQUENCE_KEY: make_vdn_partitioned_external_contract(plan),
        partitioned_runtime.FLOW_DOMAIN_STREAM_KEY: leaf,
    }
    return plan, options


@pytest.mark.parametrize("name", ["target", "source"])
def test_stream_layout_is_derived_from_the_equal_grid_contract(name):
    plan, options = _stream(name)
    native = _native_layout()
    layout, stream = partitioned_runtime.resolve_domain_stream_layout(options, native, CFG, plan.sequence_rows)
    assert stream == name
    assert (layout.video_start, layout.video_end, layout.seq_len) == (
        plan.video_start, plan.sequence_rows, plan.sequence_rows,
    )
    assert (layout.num_frames, layout.tokens_per_frame) == (plan.temporal, plan.target_rows)
    assert layout.frame_size == (plan.target_grid_h, plan.target_grid_w)
    assert (layout.text_start, layout.text_len) == (TEXT_START, TEXT_LEN)
    reference = VDNLayout(
        plan.video_start, plan.sequence_rows, plan.temporal, plan.target_rows,
        (plan.target_grid_h, plan.target_grid_w), TEXT_START, TEXT_LEN, plan.sequence_rows, 1, 1, "rows",
    )
    assert tuple(layout.bounds) == tuple(reference.bounds) and layout.full_cover == reference.full_cover
    # The released native-carrier validation accepts the stream only with its own layout.
    partitioned_runtime.validate_partitioned_external_execution(
        options, layout, plan.sequence_rows, torch.zeros(1, plan.sequence_rows, 1, 1)
    )
    with pytest.raises(RuntimeError, match="native carrier layout does not match"):
        partitioned_runtime.validate_partitioned_external_execution(
            options, native, plan.sequence_rows, torch.zeros(1, plan.sequence_rows, 1, 1)
        )


def test_absent_leaf_keeps_the_native_layout():
    native = _native_layout()
    layout, stream = partitioned_runtime.resolve_domain_stream_layout({}, native, CFG, 1)
    assert layout is native and stream is None


@pytest.mark.parametrize(
    "tamper",
    ["digest", "policy", "api", "stream", "native_rows", "native_start", "extra", "heterogeneous", "rows"],
)
def test_malformed_stream_contracts_fail_closed(tamper):
    plan, options = _stream("source")
    leaf = dict(options[partitioned_runtime.FLOW_DOMAIN_STREAM_KEY])
    rows = plan.sequence_rows
    if tamper == "digest":
        leaf["flow_semantic_digest"] = "0" * 64
    elif tamper == "policy":
        leaf["policy"] = "mixed_grid"
    elif tamper == "api":
        leaf["api"] = 2
    elif tamper == "stream":
        leaf["stream"] = "tail"
    elif tamper == "native_rows":
        leaf["native_sequence_rows"] += 1
    elif tamper == "native_start":
        leaf["native_video_start"] += 1
    elif tamper == "extra":
        leaf["unexpected"] = True
    elif tamper == "heterogeneous":
        hetero = PartitionedSequence(
            video_start=VIDEO_START, temporal=NATIVE_FRAMES, prefix_t=2,
            source_grid_h=2, source_grid_w=3, target_grid_h=3, target_grid_w=4,
        )
        options[PARTITIONED_PREFIX_KEY] = hetero.canonical_contract()
        leaf["flow_semantic_digest"] = hetero.canonical_contract()["semantic_digest"]
        rows = hetero.sequence_rows
    elif tamper == "rows":
        rows += 1
    options[partitioned_runtime.FLOW_DOMAIN_STREAM_KEY] = leaf
    with pytest.raises(RuntimeError):
        partitioned_runtime.resolve_domain_stream_layout(options, _native_layout(), CFG, rows)


def test_bridge_advertises_domain_stream_capability():
    base_branch = SimpleNamespace(short_conv=(), delta_rule="vdn_solve", num_heads=1, head_dim=8, a_fp32=True)
    block_index = 0
    cfg = {}
    head_dim = 8
    heads = 1
    k_norm = q_norm = out_proj = qkv_proj = state = SimpleNamespace()

    def current(x, rope_freqs=None, transformer_options=None):
        _ = (base_branch, block_index, cfg, head_dim, heads, k_norm, out_proj, q_norm, qkv_proj, state)
        return x

    current._vdn_forward = True
    wrapped = partitioned_runtime._wrap_vdn_forward(current)
    assert wrapped._vdn_partitioned_domain_stream_api == partitioned_runtime.FLOW_DOMAIN_STREAM_API == 1


@pytest.mark.parametrize("name", ["target", "source"])
def test_partitioned_forward_groups_only_the_stream_rows(monkeypatch, name):
    torch.manual_seed(5)
    plan, options = _stream(name)
    branch = SimpleNamespace(enable_text_state=True)
    state = VDNState("domain-stream-test", CFG, [branch], 1, 2, retain_buffers=False)
    qkv_proj = nn.Linear(2, 6, bias=False).requires_grad_(False)
    x = torch.randn(plan.sequence_rows, 2)
    grouped_layouts = []
    native_grouped = partitioned_runtime._grouped_plan

    def capture(plan_arg, layout, *, semantic_digest):
        grouped_layouts.append(layout)
        return native_grouped(plan_arg, layout, semantic_digest=semantic_digest)

    calls = []

    def attention(q, _k, _v, **kwargs):
        calls.append((kwargs["kind"], int(q.shape[0]), int(_k.shape[0])))
        return q.clone()

    readouts = []

    def linear_readout(_branch, _weights, xv, q_raw, k_raw, v_raw, **kwargs):
        readouts.append(True)
        assert int(xv.shape[0]) == plan.sequence_rows - plan.video_start
        assert tuple(kwargs["frame_sizes"]) == ((plan.target_grid_h, plan.target_grid_w),) * plan.temporal
        return q_raw.reshape(-1, 2)

    sol_package = ModuleType("sol_h3")
    sol_request = ModuleType("sol_h3.partitioned_request")
    sol_request.partitioned_request_attention = attention
    sol_request.PARTITIONED_SINK_MEASURE_API = 1
    monkeypatch.setitem(sys.modules, "sol_h3", sol_package)
    monkeypatch.setitem(sys.modules, "sol_h3.partitioned_request", sol_request)
    monkeypatch.setattr(comfy.quant_ops.ck, "rms_rope_split_half_", lambda *_a, **_k: None)
    monkeypatch.setattr("vdn_h3.softmax_provider.preprocess", lambda _o, q, k, v, _h: (q, k, v))
    monkeypatch.setattr(partitioned_linear, "partitioned_linear_readout", linear_readout)
    monkeypatch.setattr(partitioned_runtime, "_grouped_plan", capture)
    state.weights_on = lambda *_a, **_k: {"to_out_linear.weight": torch.eye(2)}
    metrics = SimpleNamespace(counters={}, increment=lambda key, value=1: metrics.counters.__setitem__(
        key, metrics.counters.get(key, 0) + value))
    options[partitioned_runtime.FLOW_PARTITIONED_STAGE_KEY] = SimpleNamespace(metrics=metrics, attention_head_t=None)
    values = {
        "state": state, "base_branch": branch, "qkv_proj": qkv_proj,
        "out_proj": lambda flat: flat.clone(), "q_norm": nn.RMSNorm(2), "k_norm": nn.RMSNorm(2),
        "heads": 1, "head_dim": 2, "block_index": 0, "cfg": CFG,
    }
    with state.runtime.execution():
        token = state._layout.set(_native_layout())
        try:
            with torch.no_grad():
                out = partitioned_runtime._partitioned_vdn_forward(
                    None, values, x, torch.zeros(1, plan.sequence_rows, 1, 1), options,
                )
        finally:
            state._layout.reset(token)
    assert out.shape == x.shape
    assert readouts == [True]
    assert len(grouped_layouts) == 1 and grouped_layouts[0].num_frames == plan.temporal
    assert grouped_layouts[0].tokens_per_frame == plan.target_rows
    # Every query and key domain lies inside this stream's own rows.
    assert calls and all(q_rows <= plan.sequence_rows and kv_rows <= plan.sequence_rows for _, q_rows, kv_rows in calls)
    assert metrics.counters[f"partitioned_vdn_domain_stream_{name}_calls"] == 1
