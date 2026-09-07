from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from comfy.ldm.minimax.model import PackedLayout, pack_audio
from vdn_h3.adapters import convert_adapter
from vdn_h3.audio_fix_train import (
    TrainableAudioFixBank,
    differentiable_run_scans,
)
from vdn_h3 import hybrid
from vdn_h3 import window as window_mod


class _DummyAttn(nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.qkv_proj = nn.Linear(hidden, hidden * 3, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)
        self.q_norm = SimpleNamespace(weight=nn.Parameter(torch.ones(hidden)), eps=1e-6)
        self.k_norm = SimpleNamespace(weight=nn.Parameter(torch.ones(hidden)), eps=1e-6)
        self.heads = 1
        self.head_dim = hidden


class _DummyMLP(nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 2, bias=False)


class _DummyBlock(nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.attn = _DummyAttn(hidden)
        self.mlp = _DummyMLP(hidden)


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock()])


def _effective(pair):
    a, b, scale = pair
    return b.float() @ a.float() * float(scale)


def test_minimax_audio_training_geometry_is_stereo_and_matches_packed_layout():
    audio_t = 32
    latent = torch.randn(1, 32, 2, audio_t)
    rows = pack_audio(latent)
    assert rows.shape == (2 * audio_t, 32)

    # Keep this positional so the regression remains tied to PackedLayout's semantic
    # argument order rather than to a parameter-name spelling that changed during
    # MiniMax-H3 development: (text_len, latent_t, latent_h, latent_w, audio_t).
    layout = PackedLayout(8, 8, 8, 8, audio_t)
    aa, ab, _ = next(seg for seg in layout.segments if seg[2] == "audio")
    assert ab - aa == rows.shape[0] == 2 * audio_t


def test_audio_fix_bank_does_not_register_or_serialize_frozen_h3():
    model = _DummyModel()
    bank = TrainableAudioFixBank(model, rank=2, alpha=2)

    assert bank.model is model
    assert "model" not in dict(bank.named_children())
    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    bank_parameter_ids = {id(parameter) for parameter in bank.parameters()}
    assert bank_parameter_ids
    assert bank_parameter_ids.isdisjoint(model_parameter_ids)
    assert bank.parameter_count == sum(
        parameter.numel() for pair in bank.pairs for parameter in pair.parameters()
    )
    state = bank.state_dict()
    assert state
    assert all(key.startswith("pairs.") for key in state)
    assert not any(key.startswith("model.") for key in state)


def test_audio_fix_export_roundtrips_through_runtime_converter_exactly():
    torch.manual_seed(7)
    model = _DummyModel()
    bank = TrainableAudioFixBank(model, rank=2, alpha=2)
    with torch.no_grad():
        for index, pair in enumerate(bank.pairs):
            pair.lora_A.copy_(
                torch.arange(pair.lora_A.numel(), dtype=torch.float32).reshape_as(pair.lora_A)
                / (11.0 + index))
            pair.lora_B.copy_(
                torch.arange(pair.lora_B.numel(), dtype=torch.float32).reshape_as(pair.lora_B)
                / (17.0 + index))

    state, cfg = bank.export_peft(dtype=torch.float32)
    converted = convert_adapter(state, {"config": cfg})

    assert cfg["source_stack"] == "comfy_int8_convrot_vdn_stage_b_turbo"
    for path, train_pair in zip(bank.targets, bank.pairs):
        expected = train_pair.lora_B @ train_pair.lora_A * train_pair.scale
        actual = _effective(converted[path])
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6), path

    qkv = converted["blocks.0.attn.qkv_proj"]
    # Exporting one fused rank-r pair as Q/K/V source pairs produces rank 3r after
    # exact block-diagonal re-fusion; the effective delta must nevertheless be exact.
    assert qkv[0].shape[0] == bank.rank * 3


def test_grouped_window_reference_keeps_gradients(monkeypatch):
    def differentiable_sdpa(q, k, v, scale, transformer_options=None):
        del transformer_options
        scores = torch.einsum("qhd,khd->hqk", q, k) * float(scale)
        probs = scores.softmax(dim=-1)
        return torch.einsum("hqk,khd->qhd", probs, v)

    monkeypatch.setattr(window_mod, "_sdpa", differentiable_sdpa)
    q = torch.randn(8, 2, 3, requires_grad=True)
    k = torch.randn(8, 2, 3, requires_grad=True)
    v = torch.randn(8, 2, 3, requires_grad=True)
    out = window_mod.window_softmax_grouped(
        q, k, v,
        video_start=2,
        video_end=8,
        num_frames=3,
        tokens_per_frame=2,
        bounds=[(-1, 1), (0, 2), (1, 3)],
        scale=3 ** -0.5,
        anchor_frames="both",
        transformer_options=None,
    )
    out.square().mean().backward()
    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0


class _IdentityDeltaBackend:
    def factor_apply(self, alpha, a_raw, b_raw):
        del a_raw
        dim = b_raw.shape[-1]
        eye = torch.eye(dim, device=b_raw.device, dtype=b_raw.dtype)
        transition = alpha.unsqueeze(-1) * eye
        return transition, b_raw


def test_differentiable_vdn_scan_propagates_gradients():
    alpha_source = torch.randn(3, 1, 2, requires_grad=True)
    alpha = torch.sigmoid(alpha_source)
    a = torch.randn(3, 1, 2, 2, requires_grad=True)
    b = torch.randn(3, 1, 2, 2, requires_grad=True)
    prefix, suffix = differentiable_run_scans(_IdentityDeltaBackend(), alpha, a, b)
    (prefix.square().mean() + suffix.square().mean()).backward()
    # This test backend intentionally ignores A; alpha and B are the recurrent inputs.
    assert alpha_source.grad is not None and torch.count_nonzero(alpha_source.grad) > 0
    assert b.grad is not None and torch.count_nonzero(b.grad) > 0
    assert a.grad is None


def test_vdn_main_rope_uses_functional_kernel_while_training(monkeypatch):
    calls = {"functional": 0, "inplace": 0}

    def functional(q, k, rope, qw, kw, **kwargs):
        del rope, qw, kw, kwargs
        calls["functional"] += 1
        return q * 1.25, k * 0.75

    def inplace(*args, **kwargs):
        del args, kwargs
        calls["inplace"] += 1
        raise AssertionError("in-place RMS/RoPE must not execute in training")

    monkeypatch.setattr(hybrid.comfy.model_management, "in_training", True)
    monkeypatch.setattr(hybrid.comfy.quant_ops.ck, "rms_rope_split_half", functional)
    monkeypatch.setattr(hybrid.comfy.quant_ops.ck, "rms_rope_split_half_", inplace)
    monkeypatch.setattr(
        hybrid,
        "_dense_subset_attention",
        lambda q, k, v, heads, head_dim, transformer_options: q + k + v,
    )

    attn = _DummyAttn(hidden=4)
    branch = SimpleNamespace(_backend=None, _backend_key=None, enable_text_state=False)
    layout = SimpleNamespace(
        seq_len=4,
        full_cover=True,
        video_start=3,
        video_end=4,
        audio_start=2,
        audio_end=3,
        num_frames=1,
        tokens_per_frame=1,
        bounds=[(0, 0)],
        frame_size=(1, 1),
        text_start=0,
        text_len=1,
    )
    state = SimpleNamespace(
        layout=layout,
        branches=[branch],
        cfg={
            "linear_enabled": False,
            "enable_softmax_gate": False,
            "anchor_frames": "none",
            "conditioning_video_context_strength": 1.0,
            "audio_video_context_strength": 1.0,
        },
        softmax_backend="grouped",
        runtime=SimpleNamespace(current=lambda: None),
        weights_on=lambda *args, **kwargs: {},
    )
    forward = hybrid.make_vdn_forward(attn, state, 0)
    x = torch.randn(4, 4, requires_grad=True)
    rope = torch.zeros(1, 4, 1, 2, 2)
    out = forward(x, rope_freqs=rope, transformer_options={})
    out.sum().backward()

    assert calls == {"functional": 1, "inplace": 0}
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
