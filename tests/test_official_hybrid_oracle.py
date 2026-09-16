"""Complete CPU oracle for the OpenVDN HybridAttention orchestration.

The lower-level oracle suite already compares every numerical branch primitive. This
module additionally executes the released HybridAttention class itself and compares
its full local-softmax + recurrent-linear fusion against the ComfyUI port using the
same synthetic parameters and packed layout.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn

from vdn_h3.branch import LinearBranch
from vdn_h3.hybrid import VDNLayout, VDNState, make_vdn_forward


ROOT = Path(os.environ.get("OPENVDN_ROOT", ""))
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENVDN_ROOT") or not ROOT.is_dir(), reason="OPENVDN_ROOT checkout is required for official oracle")


def _pkg(name):
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod
    return mod


def _load(name, relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_official_hybrid():
    for name in (
        "src", "src.models", "src.models.ops", "src.models.linear_attention",
        "src.models.softmax_attention", "src.checkpoints",
        "diffusers", "diffusers.models", "diffusers.models.transformers",
    ):
        _pkg(name)

    minimax = types.ModuleType("diffusers.models.transformers.transformer_minimax_h3")
    minimax._apply_rotary_emb = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("RoPE helper must not run when rotary_emb=None"))
    sys.modules[minimax.__name__] = minimax

    sequence = _load("src.models.sequence_layout", "src/models/sequence_layout.py")
    window = _load("official_hybrid_window", "src/models/softmax_attention/window.py")
    delta = _load("src.models.linear_attention.delta_rule",
                  "src/models/linear_attention/delta_rule.py")
    _load("src.models.linear_attention.scan", "src/models/linear_attention/scan.py")

    temporal = types.ModuleType("src.models.ops.temporal_conv")
    temporal.temporal_conv_activate = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("CUDA temporal kernel must not run in CPU hybrid oracle"))
    sys.modules[temporal.__name__] = temporal
    features = _load("src.models.linear_attention.features",
                     "src/models/linear_attention/features.py")
    _load("src.models.ops.rms_norm", "src/models/ops/rms_norm.py")
    gates = _load("src.models.attention_gates", "src/models/attention_gates.py")
    _load("src.models.linear_attention.kernels",
          "src/models/linear_attention/kernels.py")
    _load("src.models.linear_attention.layers", "src/models/linear_attention/layers.py")

    key_mapping = types.ModuleType("src.checkpoints.key_mapping")
    key_mapping.SHORT_CONV_TARGETS = ("q", "k", "v")
    key_mapping.ANCHOR_FRAME_MODES = ("none", "rows", "columns", "both")
    sys.modules[key_mapping.__name__] = key_mapping

    linear_pkg = sys.modules["src.models.linear_attention"]
    linear_pkg.DELTA_BACKENDS = delta.DELTA_BACKENDS
    branch = _load("official_hybrid_branch", "src/models/linear_attention/branch.py")
    linear_pkg.BidirectionalLinearBranch = branch.BidirectionalLinearBranch

    softmax_kernels = _load(
        "src.models.softmax_attention.kernels",
        "src/models/softmax_attention/kernels.py",
    )
    softmax_pkg = sys.modules["src.models.softmax_attention"]
    softmax_pkg.apply_softmax_gate = softmax_kernels.apply_softmax_gate
    softmax_pkg.window_bounds = window.window_bounds
    softmax_pkg.window_softmax_reference = window.window_softmax_reference
    softmax_pkg.build_window_block_mask = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("FlexAttention mask must not run in CPU reference oracle"))
    softmax_pkg.window_softmax_flex = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("FlexAttention must not run in CPU reference oracle"))

    attention_dispatch = types.ModuleType("diffusers.models.attention_dispatch")
    attention_dispatch.dispatch_attention_fn = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("dense diffusers dispatch must not run in windowed oracle"))
    sys.modules[attention_dispatch.__name__] = attention_dispatch

    fp8 = types.ModuleType("src.models.ops.fp8_linear")

    class Fp8Linear(nn.Module):
        pass

    fp8.Fp8Linear = Fp8Linear
    fp8.quantize_activation = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("FP8 path must not run in CPU oracle"))
    sys.modules[fp8.__name__] = fp8

    hybrid = _load("official_vdn_hybrid_attention", "src/models/hybrid_attention.py")
    return types.SimpleNamespace(
        sequence=sequence, features=features, gates=gates, branch=branch,
        hybrid=hybrid)


@pytest.fixture(scope="module")
def official_hybrid():
    return _load_official_hybrid()


class _OfficialOrigAttention(nn.Module):
    def __init__(self, hidden, heads, head_dim):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(hidden, inner, bias=False)
        self.to_k = nn.Linear(hidden, inner, bias=False)
        self.to_v = nn.Linear(hidden, inner, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.to_out = nn.ModuleList([nn.Linear(inner, hidden, bias=False), nn.Identity()])
        self.processor = object()


class _PortAttention(nn.Module):
    def __init__(self, hidden, heads, head_dim):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.out_proj = nn.Linear(inner, hidden, bias=False)


def _copy_base_attention(orig, port):
    with torch.no_grad():
        port.qkv_proj.weight.copy_(torch.cat(
            [orig.to_q.weight, orig.to_k.weight, orig.to_v.weight], dim=0))
        port.q_norm.weight.copy_(orig.norm_q.weight)
        port.k_norm.weight.copy_(orig.norm_k.weight)
        port.out_proj.weight.copy_(orig.to_out[0].weight)


def test_complete_hybrid_attention_direct_official(official_hybrid):
    torch.manual_seed(1701)
    hidden, heads, head_dim = 16, 2, 4
    text_len = 4
    frames, tokens_per_frame = 5, 3
    video_start = text_len
    video_end = video_start + frames * tokens_per_frame
    seq_len = video_end

    orig = _OfficialOrigAttention(hidden, heads, head_dim)
    off = official_hybrid.hybrid.HybridAttention(
        orig,
        hidden_size=hidden,
        delta_rule="vdn_solve",
        radius=1,
        chunk=2,
        enable_softmax_gate=True,
        linear_head_dim=head_dim,
        softmax_impl="ref",
        anchor_frames="both",
        enable_text_state=True,
        bridge="alpha",
        a_fp32=True,
        short_conv=(),
    )
    off.layout = official_hybrid.sequence.SequenceLayout(
        seq_len=seq_len,
        video_start=video_start,
        num_frames=frames,
        tokens_per_frame=tokens_per_frame,
        text_start=0,
        text_len=text_len,
    )

    port_attn = _PortAttention(hidden, heads, head_dim)
    _copy_base_attention(orig, port_attn)

    weights = {
        name: tensor.detach().clone()
        for name, tensor in off.linear_attention.state_dict().items()
    }
    weights["to_out_linear.weight"] = off.to_out_linear.weight.detach().clone()
    weights["softmax_gate.up.weight"] = off.softmax_gate.up.weight.detach().clone()
    weights["softmax_gate.up.bias"] = off.softmax_gate.up.bias.detach().clone()

    port_linear = LinearBranch(
        weights,
        heads,
        head_dim,
        delta_rule="vdn_solve",
        bridge="alpha",
        a_fp32=True,
        short_conv=(),
        enable_text_state=True,
    )
    cfg = {
        "radius": 1,
        "chunk": 2,
        "anchor_frames": "both",
        "enable_softmax_gate": True,
        "linear_enabled": True,
        "delta_rule": "vdn_solve",
        "bridge": "alpha",
        "a_fp32": True,
        "linear_head_dim": head_dim,
        "short_conv": (),
        "enable_text_state": True,
    }
    state = VDNState("official-hybrid-oracle", cfg, [port_linear], heads, head_dim)
    state.softmax_backend = "grouped"
    layout = VDNLayout(
        video_start=video_start,
        video_end=video_end,
        audio_start=video_start,
        audio_end=video_start,
        num_frames=frames,
        tokens_per_frame=tokens_per_frame,
        frame_size=(1, tokens_per_frame),
        text_start=0,
        text_len=text_len,
        seq_len=seq_len,
        radius=1,
        chunk=2,
        anchor_frames="both",
    )

    x = torch.randn(seq_len, hidden)
    with torch.no_grad():
        want = off(x.unsqueeze(0), rotary_emb=None, attention_mask=None)[0]
        token = state._layout.set(layout)
        try:
            got = make_vdn_forward(port_attn, state, 0)(
                x, rope_freqs=None, transformer_options={})
        finally:
            state._layout.reset(token)

    assert got.shape == want.shape == (seq_len, hidden)
    assert torch.allclose(got, want, atol=3e-5, rtol=3e-5), (
        f"complete hybrid mismatch: max={(got - want).abs().max().item():.3e}")
