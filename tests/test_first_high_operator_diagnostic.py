from __future__ import annotations

import torch
import pytest

from vdn_h3 import first_high_operator_diagnostic as w
from vdn_h3 import window


def _request(mode="native_window"):
    return (
        ("api", 1),
        ("capture_id", "capture-w"),
        ("mode", mode),
        ("stage", "high"),
        ("logical_call_limit", 1),
        ("sigma", 0.8780487775802612),
        ("target_shapes_digest", "a" * 64),
        ("source_contract_digest", "b" * 64),
    )


def _options(mode="native_window"):
    return {w.REQUEST_KEY: _request(mode), "h3_flow_stage": "high"}


def test_request_parser_is_noop_when_absent_and_strict_when_present():
    assert w.parse_request({}) is None
    parsed = w.parse_request(_options())
    assert parsed is not None
    assert parsed["mode"] == "native_window"

    with pytest.raises(RuntimeError, match="immutable tuple"):
        w.parse_request({w.REQUEST_KEY: dict(_request())})

    wrong_order = list(_request())
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
    with pytest.raises(RuntimeError, match="fields/order"):
        w.parse_request({w.REQUEST_KEY: tuple(wrong_order)})

    bad_digest = list(_request())
    bad_digest[-1] = ("source_contract_digest", "ABC")
    with pytest.raises(RuntimeError, match="SHA-256"):
        w.parse_request({w.REQUEST_KEY: tuple(bad_digest)})


def test_full_support_proxy_disables_only_linear_branch_without_mutating_state():
    class State:
        def __init__(self):
            self.cfg = {"linear_enabled": True, "enable_softmax_gate": True, "other": 7}
            self.layout = object()
            self.retain_buffers = True
            self.marker = object()

        def weights_on(self, index, device, dtype):
            return index, device, dtype

    state = State()
    original_cfg = dict(state.cfg)
    proxy = w._FullSupportStateProxy(state)

    assert proxy.cfg == {**original_cfg, "linear_enabled": False}
    assert state.cfg == original_cfg
    assert proxy.layout is state.layout
    assert proxy.retain_buffers is True
    assert proxy.marker is state.marker
    assert proxy.weights_on(3, "cpu", torch.float32) == (3, "cpu", torch.float32)


def test_absent_request_delegates_to_original_window_runtime(monkeypatch):
    sentinel = torch.randn(2, 1, 4)
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(w, "_ORIGINAL_WINDOW_SOFTMAX", original)
    q = torch.randn(2, 1, 4)
    result = w._diagnostic_window_softmax(
        q,
        q,
        q,
        0,
        2,
        2,
        1,
        [(0, 1), (0, 1)],
        0.5,
        transformer_options={},
    )
    assert result is sentinel
    assert len(calls) == 1


def test_full_support_local_groups_consume_canonical_kv_once(monkeypatch):
    sdpa_calls = []

    def fake_sdpa(q, k, v, scale, mask):
        _ = v, scale, mask
        sdpa_calls.append((int(q.shape[0]), int(k.shape[0])))
        return torch.zeros_like(q)

    monkeypatch.setattr(window, "_sdpa", fake_sdpa)
    monkeypatch.setattr(w, "_append_backend_receipt", lambda *_args, **_kwargs: None)

    seq = 9
    query = torch.randn(seq, 1, 4)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    receipts = []
    options = {
        **_options("native_full_support"),
        w.RECEIPTS_KEY: receipts,
        w._BLOCK_KEY: 0,
        w._COMPLEMENT_KEY: False,
        w._GATE_FP_KEY: "gate",
        w._ADAPTER_FP_KEY: "adapter",
    }
    result = w._diagnostic_window_softmax(
        query,
        key,
        value,
        video_start=2,
        video_end=8,
        num_frames=6,
        tokens_per_frame=1,
        bounds=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)],
        scale=0.5,
        anchor_frames="both",
        transformer_options=options,
    )

    assert result.shape == query.shape
    locals_ = [item for item in receipts if item["kind"] == "local"]
    assert locals_
    assert all(item["support_mode"] == "canonical_full" for item in locals_)
    assert all(item["canonical_full_kv"] is True for item in locals_)
    assert all(item["kv_rows"] == seq for item in locals_)
    assert all(item["complement_executed"] is False for item in locals_)
    assert all(k_rows == seq for _q_rows, k_rows in sdpa_calls)


def test_receipt_sink_is_bounded():
    options = {w.RECEIPTS_KEY: []}
    w._append_receipt(options, kind="local")
    assert options[w.RECEIPTS_KEY] == [{"kind": "local"}]

    options[w.RECEIPTS_KEY] = [{} for _ in range(800)]
    with pytest.raises(RuntimeError, match="bound exceeded"):
        w._append_receipt(options, kind="local")
