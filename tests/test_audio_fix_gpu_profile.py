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
