from __future__ import annotations

import pytest
import torch

from vdn_h3.audio_fused_fc2 import _FusedFc2Plan
from vdn_h3.audio_node import ApplyVDNH3AdvancedAudioSafe
from vdn_h3.audio_scope import (
    AudioAdapterScope,
    enter_scope,
    exit_scope,
    scale_adapter_delta,
)
from vdn_h3.hybrid import _blend_audio_context


def _with_scope(scope):
    class ScopeContext:
        def __enter__(self):
            self.token = enter_scope(scope)
            return scope

        def __exit__(self, exc_type, exc, tb):
            exit_scope(self.token)
    return ScopeContext()


def _scope(conditioning_end=3, audio_start=3, audio_end=5, audio_mod_rows=(1,)):
    return AudioAdapterScope(
        conditioning_end=conditioning_end,
        audio_start=audio_start,
        audio_end=audio_end,
        audio_mod_rows=audio_mod_rows,
    )


def test_sequence_adapter_scope_changes_only_generated_audio_rows():
    delta = torch.ones(8, 4)
    x = torch.zeros_like(delta)
    with _with_scope(_scope()):
        out = scale_adapter_delta(
            delta.clone(), x, "blocks.0.attn.qkv_proj", 0.25, 1.0)
    assert torch.equal(out[:3], torch.ones(3, 4))
    assert torch.equal(out[3:5], torch.full((2, 4), 0.25))
    assert torch.equal(out[5:], torch.ones(3, 4))


def test_sequence_adapter_scope_can_isolate_conditioning_without_touching_audio_or_video():
    delta = torch.ones(8, 4)
    x = torch.zeros_like(delta)
    with _with_scope(_scope()):
        out = scale_adapter_delta(
            delta.clone(), x, "blocks.0.mlp.fc1", 0.25, 0.0)
    assert torch.equal(out[:3], torch.zeros(3, 4))
    assert torch.equal(out[3:5], torch.full((2, 4), 0.25))
    assert torch.equal(out[5:], torch.ones(3, 4))


def test_preprocessing_adapter_keeps_full_strength_without_main_model_scope():
    delta = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    x = torch.zeros_like(delta)
    out = scale_adapter_delta(
        delta.clone(), x, "token_refiner.blocks.0.attn.qkv_proj", 0.0, 0.0)
    assert torch.equal(out, delta)


def test_main_packed_adapter_still_requires_scope():
    with pytest.raises(RuntimeError, match="outside its model-call scope"):
        scale_adapter_delta(
            torch.ones(8, 4), torch.zeros(8, 4),
            "blocks.0.attn.qkv_proj", 0.0, 0.0,
        )


def test_block_adaln_scope_changes_only_audio_modality_on_target_audio_time_rows():
    delta = torch.ones(2, 12)
    x = torch.zeros(2, 3)
    with _with_scope(_scope()):
        out = scale_adapter_delta(
            delta.clone(), x, "blocks.0.adaln_proj.linear", 0.0, 0.0)
    assert torch.equal(out[0], torch.ones(12))
    assert torch.equal(out[1, :8], torch.ones(8))
    assert torch.equal(out[1, 8:], torch.zeros(4))


def test_final_adaln_scope_retains_video_time_rows():
    delta = torch.ones(3, 6)
    x = torch.zeros(3, 2)
    with _with_scope(_scope()):
        out = scale_adapter_delta(
            delta.clone(), x, "final_layer.adaln_proj.linear", 0.0, 0.0)
    assert torch.equal(out[0], torch.ones(6))
    assert torch.equal(out[1], torch.zeros(6))
    assert torch.equal(out[2], torch.ones(6))


def test_fused_fc2_runtime_scopes_conditioning_and_audio_without_recomputing_base_gemms():
    down = torch.tensor([[1.0, 1.0]])
    up = torch.tensor([[1.0], [1.0]])
    plan = _FusedFc2Plan(
        "blocks.0.mlp.fc2", [(down, up, 1.0)],
        audio_strength=0.0, conditioning_strength=0.0)
    fc1 = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ])
    base = torch.zeros(3, 2)
    scope = _scope(conditioning_end=1, audio_start=1, audio_end=2, audio_mod_rows=(0,))
    with _with_scope(scope):
        plan.capture_fc1(None, (), fc1)
        out = plan.apply_mlp(None, (), base)
    assert torch.equal(out[0], torch.zeros(2))
    assert torch.equal(out[1], torch.zeros(2))
    assert torch.count_nonzero(out[2]) > 0


def test_context_blend_endpoints_and_interpolation():
    full = torch.tensor([[10.0, 6.0]])
    isolated = torch.tensor([[2.0, 2.0]])
    assert torch.equal(_blend_audio_context(full, isolated, 1.0), full)
    assert torch.equal(_blend_audio_context(full, isolated, 0.0), isolated)
    assert torch.equal(
        _blend_audio_context(full, isolated, 0.25),
        torch.tensor([[4.0, 3.0]]),
    )
    with pytest.raises(ValueError, match="context strength"):
        _blend_audio_context(full, isolated, -0.1)


def test_advanced_node_exposes_audio_fidelity_controls_without_changing_defaults():
    optional = ApplyVDNH3AdvancedAudioSafe.INPUT_TYPES()["optional"]
    adapter = optional["audio_adapter_strength"]
    conditioning = optional["conditioning_adapter_strength"]
    audio_context = optional["audio_video_context_strength"]
    conditioning_context = optional["conditioning_video_context_strength"]
    for field in (adapter, conditioning, audio_context, conditioning_context):
        assert field[1]["default"] == 1.0
        assert field[1]["min"] == 0.0
        assert field[1]["max"] == 1.0


def test_audio_fidelity_controls_require_bypass():
    node = ApplyVDNH3AdvancedAudioSafe()
    for kwargs in (
        {"audio_adapter_strength": 0.0},
        {"conditioning_adapter_strength": 0.0},
        {"audio_video_context_strength": 0.0},
        {"conditioning_video_context_strength": 0.0},
    ):
        with pytest.raises(ValueError, match="require lora_mode='bypass'"):
            node.apply(
                object(), "stage", True, 1.0, 0.5, "merge", "auto", "auto",
                "grouped", False, **kwargs)
