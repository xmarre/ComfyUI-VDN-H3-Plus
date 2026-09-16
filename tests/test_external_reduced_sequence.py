from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vdn_h3.hybrid import (
    VDN_EXTERNAL_SEQUENCE_API_VERSION,
    VDN_EXTERNAL_SEQUENCE_KEY,
    VDN_EXTERNAL_SEQUENCE_MODE,
    VDNLayout,
    VDNState,
    _base_attention,
    make_vdn_forward,
)


class _TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.head_dim = 2
        self.qkv_proj = nn.Linear(4, 12, bias=False)
        self.q_norm = nn.RMSNorm(2, eps=1e-6)
        self.k_norm = nn.RMSNorm(2, eps=1e-6)
        self.out_proj = nn.Identity()


def _state(*, enable_softmax_gate: bool = False):
    cfg = {
        "radius": 1,
        "chunk": 1,
        "anchor_frames": "both",
        "enable_softmax_gate": enable_softmax_gate,
        "linear_enabled": True,
    }
    branch = SimpleNamespace(w={}, enable_text_state=False)
    state = VDNState("external-reduced-test", cfg, [branch], 2, 2)
    layout = VDNLayout(
        video_start=2,
        video_end=8,
        audio_start=2,
        audio_end=2,
        num_frames=2,
        tokens_per_frame=3,
        frame_size=(1, 3),
        text_start=0,
        text_len=2,
        seq_len=8,
        radius=1,
        chunk=1,
        anchor_frames="both",
    )
    return state, layout


def _contract(*, full_rows=8, reduced_rows=5):
    return {
        VDN_EXTERNAL_SEQUENCE_KEY: {
            "api": 1,
            "mode": VDN_EXTERNAL_SEQUENCE_MODE,
            "full_sequence_rows": full_rows,
            "reduced_sequence_rows": reduced_rows,
        }
    }


def test_reduced_sequence_fails_closed_without_explicit_contract():
    torch.manual_seed(11)
    attn = _TinyAttention()
    state, layout = _state()
    x = torch.randn(5, 4)
    token = state._layout.set(layout)
    try:
        with pytest.raises(RuntimeError, match="without an explicit external-sequence contract"):
            make_vdn_forward(attn, state, 0)(x, transformer_options={})
    finally:
        state._layout.reset(token)


def test_valid_external_reduced_sequence_uses_dense_attention_without_linear_geometry():
    torch.manual_seed(12)
    attn = _TinyAttention()
    state, layout = _state(enable_softmax_gate=False)
    x = torch.randn(5, 4)
    options = _contract()

    token = state._layout.set(layout)
    try:
        got = make_vdn_forward(attn, state, 0)(x, transformer_options=options)
        want = _base_attention(attn, x, None, options)
    finally:
        state._layout.reset(token)

    assert got.shape == (5, 4)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)


def test_external_reduced_sequence_preserves_vdn_softmax_gate(monkeypatch):
    torch.manual_seed(13)
    attn = _TinyAttention()
    state, layout = _state(enable_softmax_gate=True)
    x = torch.randn(5, 4)
    options = _contract()
    gate_weight = torch.randn(2, 4)
    gate_bias = torch.randn(2)
    monkeypatch.setattr(
        state,
        "weights_on",
        lambda *_args, **_kwargs: {
            "softmax_gate.up.weight": gate_weight,
            "softmax_gate.up.bias": gate_bias,
        },
    )

    token = state._layout.set(layout)
    try:
        dense = _base_attention(attn, x, None, options)
        got = make_vdn_forward(attn, state, 0)(x, transformer_options=options)
    finally:
        state._layout.reset(token)

    gate = torch.sigmoid(torch.nn.functional.linear(x, gate_weight, gate_bias))
    expected = dense * gate.view(x.shape[0], 2, 1).expand(-1, -1, 2).reshape(x.shape[0], 4)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)


def test_external_reduced_sequence_rejects_mismatched_rope_rows():
    attn = _TinyAttention()
    state, layout = _state()
    x = torch.randn(5, 4)
    rope = torch.randn(1, 4, 1, 1, 1, 1)
    token = state._layout.set(layout)
    try:
        with pytest.raises(RuntimeError, match="RoPE rows matching the reduced hidden stream"):
            make_vdn_forward(attn, state, 0)(x, rope_freqs=rope, transformer_options=_contract())
    finally:
        state._layout.reset(token)


@pytest.mark.parametrize(
    "payload, match",
    [
        (_contract(full_rows=9), "full-row count"),
        (_contract(reduced_rows=4), "row count does not match"),
        ({VDN_EXTERNAL_SEQUENCE_KEY: {"api": 99, "mode": VDN_EXTERNAL_SEQUENCE_MODE}}, "API is unsupported"),
        ({VDN_EXTERNAL_SEQUENCE_KEY: {"api": 1, "mode": "unknown"}}, "mode is unsupported"),
    ],
)
def test_external_reduced_sequence_rejects_stale_or_invalid_contracts(payload, match):
    attn = _TinyAttention()
    state, layout = _state()
    x = torch.randn(5, 4)
    token = state._layout.set(layout)
    try:
        with pytest.raises(RuntimeError, match=match):
            make_vdn_forward(attn, state, 0)(x, transformer_options=payload)
    finally:
        state._layout.reset(token)


def test_vdn_forward_advertises_external_sequence_capability():
    attn = _TinyAttention()
    state, _layout = _state()
    forward = make_vdn_forward(attn, state, 0)
    assert forward._vdn_forward is True
    assert forward._vdn_external_sequence_api == VDN_EXTERNAL_SEQUENCE_API_VERSION


def _mixed_contract():
    return {"api": 2, "mode": VDN_EXTERNAL_SEQUENCE_MODE, "topology": "mixed_grid_low_suffix",
            "native_sequence_rows": 8, "sequence_rows": 11, "video_start": 2,
            "temporal": 2, "prefix_t": 1, "source_rows_per_frame": 3, "prefix_rows_per_frame": 6}


def test_mixed_expansion_is_explicit_and_native_mode_resumes():
    from vdn_h3.hybrid import _external_reduced_sequence_active

    _, layout = _state()
    rope = torch.zeros(1, 11, 1, 1, 2, 2)
    assert _external_reduced_sequence_active({VDN_EXTERNAL_SEQUENCE_KEY: _mixed_contract()}, layout, 11, rope)
    assert not _external_reduced_sequence_active({}, layout, 8, None)


def test_mixed_expansion_preserves_learned_dense_gate(monkeypatch):
    torch.manual_seed(31)
    attn = _TinyAttention().requires_grad_(False)
    state, layout = _state(enable_softmax_gate=True)
    x = torch.randn(11, 4)
    options = {VDN_EXTERNAL_SEQUENCE_KEY: _mixed_contract()}
    rope = torch.eye(2).reshape(1, 1, 1, 1, 2, 2).expand(1, 11, 1, 1, 2, 2).contiguous()
    weight, bias = torch.randn(2, 4), torch.randn(2)
    monkeypatch.setattr(state, "weights_on", lambda *_a, **_k: {
        "softmax_gate.up.weight": weight, "softmax_gate.up.bias": bias})
    token = state._layout.set(layout)
    try:
        with torch.no_grad():
            dense = _base_attention(attn, x, rope, options)
            got = make_vdn_forward(attn, state, 0)(x, rope_freqs=rope, transformer_options=options)
    finally:
        state._layout.reset(token)
    gate = torch.sigmoid(torch.nn.functional.linear(x, weight, bias)).repeat_interleave(2, dim=1)
    assert torch.allclose(got, dense * gate, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("field", list(_mixed_contract()))
def test_mixed_contract_rejects_each_stale_field(field):
    from vdn_h3.hybrid import _external_reduced_sequence_active

    _, layout = _state()
    contract = _mixed_contract()
    contract[field] = 99 if isinstance(contract[field], int) else "invalid"
    with pytest.raises(RuntimeError):
        _external_reduced_sequence_active({VDN_EXTERNAL_SEQUENCE_KEY: contract}, layout, 11,
                                          torch.zeros(1, 11, 1, 1, 2, 2))


def test_mixed_requires_rope_and_rejects_stale_full_stream_contract():
    from vdn_h3.hybrid import _external_reduced_sequence_active

    _, layout = _state()
    with pytest.raises(RuntimeError, match="RoPE"):
        _external_reduced_sequence_active({VDN_EXTERNAL_SEQUENCE_KEY: _mixed_contract()}, layout, 11, None)
    with pytest.raises(RuntimeError, match="stale"):
        _external_reduced_sequence_active(_contract(), layout, 8, None)
