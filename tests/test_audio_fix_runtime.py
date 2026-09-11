from __future__ import annotations

import pytest
import torch

from vdn_h3.audio_fix_runtime import _AudioOnlyPostForwardLoRA, _validate_paths
from vdn_h3.audio_node import ApplyVDNH3AdvancedAudioSafe, _validate_audio_fix_config
from vdn_h3.audio_scope import AudioAdapterScope, enter_scope, exit_scope


def _scope():
    return AudioAdapterScope(
        conditioning_end=2,
        audio_start=2,
        audio_end=4,
        audio_mod_rows=(1,),
    )


def test_audio_fix_residual_changes_only_generated_audio_rows():
    down = torch.tensor([[1.0, 0.0, 0.0]])
    up = torch.tensor([[1.0], [2.0]])
    plan = _AudioOnlyPostForwardLoRA(
        "blocks.0.attn.out_proj", [(down, up, 1.0)])
    x = torch.tensor([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ])
    base = torch.zeros(5, 2)
    token = enter_scope(_scope())
    try:
        out = plan(None, (x,), base.clone())
    finally:
        exit_scope(token)
    assert torch.equal(out[:2], torch.zeros(2, 2))
    assert torch.equal(out[2], torch.tensor([3.0, 6.0]))
    assert torch.equal(out[3], torch.tensor([4.0, 8.0]))
    assert torch.equal(out[4:], torch.zeros(1, 2))


def test_audio_fix_residual_supports_batched_packed_rows():
    down = torch.tensor([[1.0, 0.0]])
    up = torch.tensor([[1.0]])
    plan = _AudioOnlyPostForwardLoRA(
        "blocks.0.mlp.fc1", [(down, up, 1.0)])
    x = torch.arange(20, dtype=torch.float32).reshape(2, 5, 2)
    base = torch.zeros(2, 5, 1)
    token = enter_scope(_scope())
    try:
        out = plan(None, (x,), base.clone())
    finally:
        exit_scope(token)
    assert torch.count_nonzero(out[:, :2]) == 0
    assert torch.count_nonzero(out[:, 2:4]) > 0
    assert torch.count_nonzero(out[:, 4:]) == 0


def test_audio_fix_fails_closed_without_scope():
    plan = _AudioOnlyPostForwardLoRA(
        "blocks.0.attn.out_proj",
        [(torch.ones(1, 2), torch.ones(2, 1), 1.0)],
    )
    with pytest.raises(RuntimeError, match="outside the packed model-call scope"):
        plan(None, (torch.ones(4, 2),), torch.zeros(4, 2))


def test_audio_fix_accepts_only_portable_comfy_targets():
    _validate_paths({
        "blocks.0.attn.qkv_proj": object(),
        "blocks.0.attn.out_proj": object(),
        "blocks.0.mlp.fc1": object(),
    })
    for bad in (
        "blocks.0.mlp.fc2",
        "blocks.0.adaln_proj.linear",
        "token_refiner.blocks.0.attn.qkv_proj",
        "final_layer.adaln_proj.linear",
    ):
        with pytest.raises(ValueError, match="unsupported Comfy targets"):
            _validate_paths({bad: object()})


def test_audio_fix_checkpoint_contract_is_strict():
    good = {"type": "lora", "version": 1, "config": {
        "scope": "generated_audio",
        "target_policy": "portable_sequence_linear",
        "exact_targets": True,
    }}
    assert _validate_audio_fix_config(good)["scope"] == "generated_audio"
    for key, value in (
        ("scope", "global"),
        ("target_policy", "sequence_linear"),
        ("exact_targets", False),
    ):
        bad = {"type": "lora", "version": 1, "config": dict(good["config"])}
        bad["config"][key] = value
        with pytest.raises(ValueError):
            _validate_audio_fix_config(bad)


def test_advanced_node_exposes_audio_fix_strength_at_native_default():
    optional = ApplyVDNH3AdvancedAudioSafe.INPUT_TYPES()["optional"]
    field = optional["audio_fix_strength"]
    assert field[1]["default"] == 1.0
    assert field[1]["min"] == 0.0
    assert field[1]["max"] == 2.0
