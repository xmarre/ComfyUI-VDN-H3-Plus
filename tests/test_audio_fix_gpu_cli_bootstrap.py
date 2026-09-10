from __future__ import annotations

import os
import pathlib
import subprocess
import sys


def test_gpu_trainer_can_bootstrap_comfy_from_cli_before_vdn_import():
    """Regression for importing vdn_h3 progress helpers before Comfy was on sys.path."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    comfy_root = os.environ.get("COMFYUI_ROOT")
    if not comfy_root:
        raise RuntimeError("COMFYUI_ROOT is required for the standalone CLI bootstrap test")

    env = os.environ.copy()
    env.pop("COMFYUI_ROOT", None)
    # Deliberately do not put the Comfy checkout on PYTHONPATH. The executable must
    # honor its own --comfy-root before importing Comfy-dependent vdn_h3 modules.
    env["PYTHONPATH"] = str(repo)

    completed = subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "audio_fix_int8_train_gpu.py"),
            "--comfy-root",
            str(pathlib.Path(comfy_root).resolve()),
            "--help",
        ],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout
    assert "--sampler-steps" in completed.stdout
    assert "--turbo-strength" in completed.stdout
