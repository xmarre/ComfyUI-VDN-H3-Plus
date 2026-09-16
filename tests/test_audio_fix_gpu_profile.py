from __future__ import annotations

import ast
from pathlib import Path


GPU_TRAINER = Path(__file__).resolve().parents[1] / "tools" / "audio_fix_int8_train_gpu.py"


def _module_constants():
    tree = ast.parse(GPU_TRAINER.read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                values[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def test_gpu_trainer_uses_released_turbo_8_step_grid():
    assert _module_constants()["CANONICAL_SAMPLER_STEPS"] == 8


def test_gpu_trainer_rejects_10_step_canonical_profile_textually():
    source = GPU_TRAINER.read_text(encoding="utf-8")
    assert "Canonical Turbo-1.0 audio-fix training requires --sampler-steps 8" in source
    assert '"sampler_steps": CANONICAL_SAMPLER_STEPS' in source


def test_gpu_trainer_records_factor_gradient_and_prodigy_telemetry():
    source = GPU_TRAINER.read_text(encoding="utf-8")
    for field in (
        '"nonzero_lora_a_grad_tensors"',
        '"nonzero_lora_b_grad_tensors"',
        '"prodigy_d"',
        '"prodigy_d_prev"',
    ):
        assert field in source
    assert 'optimizer_group = optimizer.param_groups[0]' in source
    assert 'prodigy_d = float(optimizer_group["d"])' in source
    assert 'prodigy_d_prev = float(optimizer_group["d_prev"])' in source
