from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from vdn_h3.hybrid import (
    VDNLayout,
    VDNState,
    make_vdn_forward,
)


class _TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 1
        self.head_dim = 2
        self.qkv_proj = nn.Linear(2, 6, bias=False)
        self.q_norm = nn.RMSNorm(2, eps=1e-6)
        self.k_norm = nn.RMSNorm(2, eps=1e-6)
        self.out_proj = nn.Identity()


class _RawQReadoutBranch:
    enable_text_state = False
    w = {}

    def readout(
        self,
        _weights,
        _xv,
        q_raw,
        _k_raw,
        _v_raw,
        _num_frames,
        _tokens_per_frame,
        _bounds,
        **_kwargs,
    ):
        return q_raw.reshape(q_raw.shape[0], -1)


def test_native_linear_complement_consumes_raw_views_without_activation_scratch(monkeypatch):
    torch.manual_seed(901)
    attn = _TinyAttention().requires_grad_(False)
    cfg = {
        "radius": 0,
        "chunk": 1,
        "anchor_frames": "none",
        "enable_softmax_gate": False,
        "linear_enabled": True,
    }
    state = VDNState(
        "native-lifetime-test",
        cfg,
        [_RawQReadoutBranch()],
        1,
        2,
        retain_buffers=True,
    )
    layout = VDNLayout(
        video_start=0,
        video_end=4,
        num_frames=4,
        tokens_per_frame=1,
        frame_size=(1, 1),
        text_start=0,
        text_len=0,
        seq_len=4,
        radius=0,
        chunk=1,
        anchor_frames="none",
    )
    events = []

    def weights_on(*_args, **kwargs):
        assert kwargs.get("prefetch_next") is False
        events.append("weights")
        return {"to_out_linear.weight": torch.eye(2)}

    def softmax(query, *_args, **_kwargs):
        events.append("softmax")
        return torch.zeros_like(query)

    state.weights_on = weights_on
    state.prefetch_next_weights = lambda *_args, **_kwargs: events.append("prefetch")
    monkeypatch.setattr(
        "vdn_h3.retained.window_softmax_grouped_runtime",
        softmax,
    )

    x = torch.randn(4, 2)
    with torch.no_grad():
        expected_q = attn.qkv_proj(x).split(2, dim=-1)[0].clone()

    with state.runtime.execution() as resources:
        # Simulate the retained raw-QKV payload left by the preceding partitioned
        # low/probe stage. Native high-grid entry must release it before qkv_proj
        # allocates the new target-grid projection.
        resources.activation_scratch(
            video_rows=3,
            text_rows=1,
            heads=1,
            head_dim=2,
            device=x.device,
            dtype=x.dtype,
        )
        assert resources.retained_counts()["activations"] == 1

        def forbidden_activation_scratch(*_args, **_kwargs):
            raise AssertionError(
                "native VDN forward must not preserve raw Q/K/V in activation scratch"
            )

        qkv_entry_activation_counts = []
        hook = attn.qkv_proj.register_forward_pre_hook(
            lambda *_args: qkv_entry_activation_counts.append(
                resources.retained_counts()["activations"]
            )
        )
        monkeypatch.setattr(resources, "activation_scratch", forbidden_activation_scratch)
        token = state._layout.set(layout)
        try:
            with torch.no_grad():
                got = make_vdn_forward(attn, state, 0)(x, transformer_options={})
        finally:
            state._layout.reset(token)
            hook.remove()

        assert qkv_entry_activation_counts == [0]
        assert resources.retained_counts()["activations"] == 0

    assert torch.equal(got, expected_q)
    assert events == ["weights", "softmax", "prefetch"]


def test_bounded_projection_preserves_native_vdn_softmax_and_linear_outputs(monkeypatch):
    from vdn_h3.softmax_provider import QKV_PROJECTION_KEY

    torch.manual_seed(2026)
    attn = _TinyAttention().requires_grad_(False)
    cfg = {
        "radius": 0, "chunk": 1, "anchor_frames": "none",
        "enable_softmax_gate": False, "linear_enabled": True,
    }
    state = VDNState("bounded-projection-contract", cfg, [_RawQReadoutBranch()], 1, 2)
    layout = VDNLayout(0, 4, 4, 1, (1, 1), 0, 0, 4, 0, 1, "none")
    state.weights_on = lambda *_a, **_kw: {"to_out_linear.weight": torch.eye(2)}
    state.prefetch_next_weights = lambda *_a, **_kw: None
    from vdn_h3.retained import window_softmax_grouped_runtime
    monkeypatch.setattr("vdn_h3.retained.window_softmax_grouped_runtime",
                        lambda *a, **kw: window_softmax_grouped_runtime(*a, **kw))
    calls = []

    def projection(linear, x, heads, dim):
        assert heads == 1 and dim == 2
        carriers = tuple(torch.empty((len(x), heads, dim), dtype=x.dtype) for _ in range(3))
        for start in range(0, len(x), 2):
            stop = min(start + 2, len(x))
            packed = linear(x[start:stop])
            for carrier, piece in zip(carriers, packed.split(heads * dim, dim=-1)):
                carrier[start:stop].copy_(piece.view(stop - start, heads, dim))
            calls.append((start, stop))
        return carriers

    x = torch.randn(4, 2)
    token = state._layout.set(layout)
    try:
        with torch.no_grad(), state.runtime.execution():
            reference = make_vdn_forward(attn, state, 0)(x, transformer_options={})
            candidate = make_vdn_forward(attn, state, 0)(
                x, transformer_options={QKV_PROJECTION_KEY: projection})
    finally:
        state._layout.reset(token)
    assert calls == [(0, 2), (2, 4)]
    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)


def test_bounded_projection_rejects_shared_or_malformed_carriers():
    import pytest
    from vdn_h3.softmax_provider import QKV_PROJECTION_KEY

    attn = _TinyAttention().requires_grad_(False)
    cfg = {"radius": 0, "chunk": 1, "anchor_frames": "none",
           "enable_softmax_gate": False, "linear_enabled": False}
    state = VDNState("carrier-validation", cfg, [_RawQReadoutBranch()], 1, 2)
    layout = VDNLayout(0, 4, 4, 1, (1, 1), 0, 0, 4, 0, 1, "none")
    forward = make_vdn_forward(attn, state, 0)
    token = state._layout.set(layout)
    x = torch.randn(4, 2)
    try:
        with torch.no_grad(), state.runtime.execution():
            for malformed in (
                lambda *_: (x, x, x),
                lambda *_: tuple(torch.empty(4, 1, 3) for _ in range(3)),
            ):
                with pytest.raises(RuntimeError, match="independent BF16 carrier"):
                    forward(x, transformer_options={QKV_PROJECTION_KEY: malformed})
    finally:
        state._layout.reset(token)
