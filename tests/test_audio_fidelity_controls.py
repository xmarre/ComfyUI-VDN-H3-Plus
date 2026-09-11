from __future__ import annotations

import pytest
import torch

from vdn_h3.hybrid import _scope_softmax_gate
from vdn_h3.nodes import ApplyVDNH3Advanced, _apply_adapter_ablation


def test_video_only_gate_preserves_global_rows_and_checkpoint_video_gate():
    gate = torch.tensor([
        [0.10, 0.20],
        [0.30, 0.40],
        [0.50, 0.60],
        [0.70, 0.80],
    ])
    original = gate.clone()

    scoped = _scope_softmax_gate(gate, video_start=2, mode="video_only")

    assert torch.equal(gate, original), "audio-fidelity scoping must not mutate the learned gate"
    assert torch.equal(scoped[:2], torch.ones_like(scoped[:2]))
    assert torch.equal(scoped[2:], original[2:])
    assert _scope_softmax_gate(gate, video_start=2, mode="checkpoint") is gate


def test_global_gate_mode_rejects_unknown_scope():
    with pytest.raises(ValueError, match="global_gate_mode"):
        _scope_softmax_gate(torch.ones(2, 1), video_start=1, mode="audio_only")


def _converted_fixture():
    return {
        "default": {
            "blocks.0.attn.qkv_proj": "stage-b-dit-qkv",
            "blocks.0.attn.out_proj": "stage-b-dit-out",
            "token_refiner.blocks.0.attn.qkv_proj": "stage-b-refiner-qkv",
        },
        "turbo": {
            "blocks.0.attn.qkv_proj": "turbo-attn",
            "blocks.0.mlp.fc1": "turbo-mlp",
            "blocks.0.adaln_proj.linear": "turbo-adaln",
            "final_layer.adaln_proj.linear": "turbo-final-adaln",
            "token_refiner.blocks.0.attn.qkv_proj": "turbo-refiner",
        },
    }


@pytest.mark.parametrize(
    ("ablation", "adapter", "removed_paths"),
    [
        (
            "stage_b_dit_off",
            "default",
            {"blocks.0.attn.qkv_proj", "blocks.0.attn.out_proj"},
        ),
        (
            "stage_b_refiner_off",
            "default",
            {"token_refiner.blocks.0.attn.qkv_proj"},
        ),
        (
            "turbo_attention_off",
            "turbo",
            {"blocks.0.attn.qkv_proj"},
        ),
        (
            "turbo_mlp_off",
            "turbo",
            {"blocks.0.mlp.fc1"},
        ),
        (
            "turbo_adaln_off",
            "turbo",
            {"blocks.0.adaln_proj.linear", "final_layer.adaln_proj.linear"},
        ),
        (
            "turbo_refiner_off",
            "turbo",
            {"token_refiner.blocks.0.attn.qkv_proj"},
        ),
    ],
)
def test_adapter_ablation_removes_only_requested_target_class(
    ablation, adapter, removed_paths
):
    converted = _converted_fixture()
    before_other = {
        name: dict(modules) for name, modules in converted.items() if name != adapter
    }

    filtered, removed = _apply_adapter_ablation(converted, ablation)

    assert removed == {adapter: len(removed_paths)}
    assert removed_paths.isdisjoint(filtered[adapter])
    expected_kept = set(converted[adapter]) - removed_paths
    assert set(filtered[adapter]) == expected_kept
    for name, modules in before_other.items():
        assert filtered[name] == modules


def test_none_ablation_is_exact_noop():
    converted = _converted_fixture()
    filtered, removed = _apply_adapter_ablation(converted, "none")
    assert filtered is converted
    assert removed == {}


def test_turbo_ablation_requires_turbo_enabled():
    converted = {"default": _converted_fixture()["default"]}
    with pytest.raises(ValueError, match="requires adapter 'turbo'"):
        _apply_adapter_ablation(converted, "turbo_adaln_off")


def test_advanced_node_exposes_opt_in_audio_fidelity_controls():
    optional = ApplyVDNH3Advanced.INPUT_TYPES()["optional"]
    assert optional["global_gate_mode"][0] == ["checkpoint", "video_only"]
    assert optional["global_gate_mode"][1]["default"] == "checkpoint"
    assert optional["adapter_ablation"][0][0] == "none"
    assert optional["adapter_ablation"][1]["default"] == "none"
