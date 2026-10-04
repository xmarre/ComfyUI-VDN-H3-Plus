"""Attention temporaries must die before partitioned linear workspace begins."""
import sys
from types import ModuleType, SimpleNamespace
import weakref

import pytest
import torch
from torch import nn

import comfy.quant_ops
from vdn_h3 import partitioned_linear, partitioned_runtime
from vdn_h3.hybrid import VDNLayout, VDNState
from vdn_h3 import branch as B
from vdn_h3.retained import RuntimeLinearBranch
from vdn_h3.apply import _PostForwardLoRA
from vdn_h3.boundary_witness import WITNESS_KEY
from vdn_h3.partitioned_sequence import (
    PARTITIONED_PREFIX_KEY, PartitionedSequence, make_vdn_partitioned_external_contract,
)


@pytest.mark.parametrize("gate_enabled,retain", [(False, False), (False, True), (True, True)])
def test_partitioned_linear_keeps_raw_copies_without_attention_temporaries(monkeypatch, gate_enabled, retain):
    torch.manual_seed(62)
    plan = PartitionedSequence(
        video_start=7, temporal=5, prefix_t=2,
        source_grid_h=2, source_grid_w=3, target_grid_h=3, target_grid_w=4,
    )
    cfg = {"radius": 1, "chunk": 1, "anchor_frames": "none",
           "enable_softmax_gate": gate_enabled, "linear_enabled": True}
    branch = SimpleNamespace(enable_text_state=True)
    state = VDNState("partitioned-lifetime-test", cfg, [branch], 1, 2, retain_buffers=retain)
    layout = VDNLayout(
        video_start=7, video_end=37, num_frames=5, tokens_per_frame=6,
        frame_size=(2, 3), text_start=1, text_len=2, seq_len=37,
        radius=1, chunk=1, anchor_frames="none",
    )
    qkv_proj = nn.Linear(2, 6, bias=False).requires_grad_(False)
    x = torch.randn(plan.sequence_rows, 2)
    with torch.no_grad():
        raw_q, raw_k, raw_v = tuple(t.clone() for t in qkv_proj(x).split(2, dim=-1))
    references = {}
    events = []

    def qkv_hook(_module, _args, result):
        references["qkv_storage_owner"] = weakref.ref(result)

    qkv_proj.register_forward_hook(qkv_hook)

    def rope(q4, k4, *_args, **_kwargs):
        references["q4"] = weakref.ref(q4)
        references["k4"] = weakref.ref(k4)
        q4.add_(1.0)
        k4.add_(2.0)

    def preprocess(_options, q, k, v, _heads):
        references.update({name: weakref.ref(t) for name, t in (("q", q), ("k", k), ("v", v))})
        return q, k, v

    def attention(q, _k, _v, **_kwargs):
        return q.clone()

    def weights_on(*_args, **_kwargs):
        assert all(ref() is None for ref in references.values())
        events.append("weights")
        return {"to_out_linear.weight": torch.eye(2),
                "softmax_gate.up.weight": torch.zeros(1, 2),
                "softmax_gate.up.bias": torch.zeros(1)}

    def out_proj(flat):
        references["softmax_flat"] = weakref.ref(flat)
        if flat._base is not None:
            references["softmax_storage_owner"] = weakref.ref(flat._base)
        return flat.clone()

    def linear_readout(_branch, _weights, _xv, q_raw, k_raw, v_raw, **kwargs):
        assert all(ref() is None for ref in references.values())
        events.append("linear")
        for actual, original in zip((q_raw, k_raw, v_raw), (raw_q, raw_k, raw_v)):
            assert torch.equal(actual.reshape(-1, 2), original[plan.video_start:])
        assert torch.equal(kwargs["text_k_raw"].reshape(-1, 2), raw_k[1:3])
        assert torch.equal(kwargs["text_v_raw"].reshape(-1, 2), raw_v[1:3])
        return q_raw.reshape(-1, 2)

    sol_package = ModuleType("sol_h3")
    sol_request = ModuleType("sol_h3.partitioned_request")
    sol_request.partitioned_request_attention = attention
    monkeypatch.setitem(sys.modules, "sol_h3", sol_package)
    monkeypatch.setitem(sys.modules, "sol_h3.partitioned_request", sol_request)
    monkeypatch.setattr(comfy.quant_ops.ck, "rms_rope_split_half_", rope)
    monkeypatch.setattr("vdn_h3.softmax_provider.preprocess", preprocess)
    monkeypatch.setattr(partitioned_linear, "partitioned_linear_readout", linear_readout)
    state.weights_on = weights_on
    values = {
        "state": state, "base_branch": branch, "qkv_proj": qkv_proj,
        "out_proj": out_proj, "q_norm": nn.RMSNorm(2), "k_norm": nn.RMSNorm(2),
        "heads": 1, "head_dim": 2, "block_index": 0, "cfg": cfg,
    }
    options = {PARTITIONED_PREFIX_KEY: plan.canonical_contract(),
               partitioned_runtime.VDN_EXTERNAL_SEQUENCE_KEY: make_vdn_partitioned_external_contract(plan)}
    with state.runtime.execution():
        token = state._layout.set(layout)
        try:
            with torch.no_grad():
                got = partitioned_runtime._partitioned_vdn_forward(
                    None, values, x, torch.zeros(1, plan.sequence_rows, 1, 1), options,
                )
        finally:
            state._layout.reset(token)

    expected = (raw_q + 1.0) * (0.5 if gate_enabled else 1.0)
    expected[plan.video_start:] += raw_q[plan.video_start:]
    assert torch.equal(got, expected)
    assert events == ["weights", "linear"]


@pytest.mark.parametrize("fast_kernels", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_same_grid_forward_executes_uniform_linear_and_publishes_receipt(monkeypatch, fast_kernels, dtype, strength):
    torch.manual_seed(774)
    plan = PartitionedSequence(
        video_start=7, temporal=5, prefix_t=2,
        source_grid_h=2, source_grid_w=3, target_grid_h=2, target_grid_w=3,
    )
    cfg = {"radius": 1, "chunk": 1, "anchor_frames": "none",
           "enable_softmax_gate": True, "linear_enabled": True}
    weights = {
        "beta_proj.weight": torch.randn(1, 2) * 0.1,
        "alpha.down.weight": torch.randn(2, 2) * 0.1,
        "alpha.up.weight": torch.randn(2, 2) * 0.1,
        "alpha.dt_bias": torch.randn(2) * 0.1,
        "alpha.A_log": torch.randn(1) * 0.1,
        "output_gate.down.weight": torch.randn(2, 2) * 0.1,
        "output_gate.up.weight": torch.randn(2, 2) * 0.1,
        "output_gate.up.bias": torch.randn(2) * 0.1,
        "norm.weight": torch.ones(2), "to_out_linear.weight": torch.eye(2),
        "softmax_gate.up.weight": torch.zeros(1, 2), "softmax_gate.up.bias": torch.zeros(1),
    }
    weights = {name: value.to(dtype) for name, value in weights.items()}
    branch = RuntimeLinearBranch(weights, 1, 2, short_conv=(), enable_text_state=True)
    branch.fuse_epilogue = fast_kernels
    state = VDNState("uniform-linear-test", cfg, [branch], 1, 2, retain_buffers=True)
    layout = VDNLayout(
        video_start=7, video_end=37, num_frames=5, tokens_per_frame=6,
        frame_size=(2, 3), text_start=1, text_len=2, seq_len=37,
        radius=1, chunk=1, anchor_frames="none",
    )
    values = {
        "state": state, "base_branch": branch, "qkv_proj": nn.Linear(2, 6, bias=False, dtype=dtype),
        "out_proj": lambda flat: flat.clone(), "q_norm": nn.RMSNorm(2), "k_norm": nn.RMSNorm(2),
        "heads": 1, "head_dim": 2, "block_index": 0, "cfg": cfg,
    }
    values["qkv_proj"].register_forward_hook(_PostForwardLoRA([
        (torch.randn(1, 2).to(dtype), torch.randn(6, 1).to(dtype), strength),
    ]))
    x = torch.randn(plan.sequence_rows, 2).to(dtype)
    counters = {}
    events = []
    weights_calls = []

    def increment(name, value=1):
        counters[name] = counters.get(name, 0) + value

    options = {
        PARTITIONED_PREFIX_KEY: plan.canonical_contract(),
        partitioned_runtime.VDN_EXTERNAL_SEQUENCE_KEY: make_vdn_partitioned_external_contract(plan),
        partitioned_runtime.FLOW_PARTITIONED_STAGE_KEY: SimpleNamespace(metrics=SimpleNamespace(increment=increment)),
    }
    sol_package = ModuleType("sol_h3")
    sol_request = ModuleType("sol_h3.partitioned_request")
    def attention(q, _k, _v, **kwargs):
        events.append("softmax")
        return q.clone()

    sol_request.partitioned_request_attention = attention
    monkeypatch.setitem(sys.modules, "sol_h3", sol_package)
    monkeypatch.setitem(sys.modules, "sol_h3.partitioned_request", sol_request)
    def rope(q4, k4, *args, **kwargs):
        events.append("rope")
        q4.add_(1.0)
        k4.add_(2.0)

    monkeypatch.setattr(comfy.quant_ops.ck, "rms_rope_split_half_", rope)
    monkeypatch.setattr("vdn_h3.softmax_provider.preprocess", lambda _options, q, k, v, _heads: (q, k, v))
    monkeypatch.setattr(B, "_run_compiled", lambda _key, body, *args, **kwargs: body(*args, **kwargs))
    epilogue = B.linear_epilogue
    monkeypatch.setattr(B, "linear_epilogue", lambda *args, **kwargs: epilogue(*args, fuse=False))
    def weights_on(*args, **kwargs):
        weights_calls.append(kwargs.get("prefetch_next", True))
        return weights

    state.weights_on = weights_on
    state.prefetch_next_weights = lambda *args: events.append("prefetch")
    readout = partitioned_linear.partitioned_linear_readout

    def observed_readout(*args, **kwargs):
        events.append("linear")
        return readout(*args, **kwargs)

    monkeypatch.setattr(partitioned_linear, "partitioned_linear_readout", observed_readout)

    def execute():
        with state.runtime.execution(), torch.no_grad():
            token = state._layout.set(layout)
            try:
                result = partitioned_runtime._partitioned_vdn_forward(
                    None, values, x, torch.zeros(1, plan.sequence_rows, 1, 1), options,
                )
                if WITNESS_KEY not in options:
                    assert not state.runtime.current()._activations
                return result
            finally:
                state._layout.reset(token)

    native = execute()
    expected_boundary_counters = {
        "partitioned_vdn_boundary_suffix_dense_calls": 1,
        "partitioned_vdn_boundary_suffix_dense_q_rows": 6,
        "partitioned_vdn_boundary_suffix_dense_kv_rows": 25,
        "partitioned_vdn_boundary_suffix_dense_query_frames": 1,
    }
    assert counters == {
        "partitioned_vdn_uniform_linear_calls": 1,
        "partitioned_vdn_uniform_pre_rope_calls": 1,
        **({"partitioned_vdn_uniform_fast_requested_calls": 1} if fast_kernels else {}),
        **expected_boundary_counters,
    }
    assert weights_calls == [False]
    assert events.index("linear") < events.index("rope") < events.index("softmax") < events.index("prefetch")
    counters.clear()
    weights_calls.clear()
    events.clear()

    def general(*args, **kwargs):
        kwargs["diagnostic_stats"] = {}
        return readout(*args, **kwargs)

    # Witness ownership retains the late/copy path, even when no capture is claimed.
    options[WITNESS_KEY] = SimpleNamespace(api=1, claim=lambda _context: False)
    monkeypatch.setattr(partitioned_linear, "partitioned_linear_readout", general)
    reference = execute()
    assert counters == expected_boundary_counters
    assert weights_calls == [True]
    tolerance = 2e-5 if dtype == torch.float32 else 2e-2
    assert torch.allclose(native.float(), reference.float(), rtol=tolerance, atol=tolerance)
