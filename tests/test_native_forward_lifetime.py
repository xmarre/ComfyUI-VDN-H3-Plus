from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from vdn_h3.hybrid import VDNLayout, VDNState, make_vdn_forward


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
    state.weights_on = lambda *_args, **_kwargs: {
        "to_out_linear.weight": torch.eye(2),
    }
    monkeypatch.setattr(
        "vdn_h3.retained.window_softmax_grouped_runtime",
        lambda query, *_args, **_kwargs: torch.zeros_like(query),
    )

    x = torch.randn(4, 2)
    with torch.no_grad():
        expected_q = attn.qkv_proj(x).split(2, dim=-1)[0].clone()

    with state.runtime.execution() as resources:
        def forbidden_activation_scratch(*_args, **_kwargs):
            raise AssertionError(
                "native VDN forward must not preserve raw Q/K/V in activation scratch"
            )

        monkeypatch.setattr(resources, "activation_scratch", forbidden_activation_scratch)
        token = state._layout.set(layout)
        try:
            with torch.no_grad():
                got = make_vdn_forward(attn, state, 0)(x, transformer_options={})
        finally:
            state._layout.reset(token)

        assert resources.retained_counts()["activations"] == 0

    assert torch.equal(got, expected_q)
