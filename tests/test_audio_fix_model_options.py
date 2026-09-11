from types import SimpleNamespace

import pytest
import torch

import comfy.patcher_extension
from comfy.patcher_extension import WrappersMP

from vdn_h3.audio_fix_model_options import build_student_transformer_options


def _wrapper(*args, **kwargs):
    del args, kwargs


def test_student_transformer_options_merge_required_modelpatcher_wrappers():
    existing = lambda *args, **kwargs: None
    vdn = lambda *args, **kwargs: None
    scope = lambda *args, **kwargs: None
    callback = lambda *args, **kwargs: None
    original_model_options = {
        "transformer_options": {
            "sentinel": "preserved",
            "wrappers": {
                WrappersMP.DIFFUSION_MODEL: {"existing": [existing]},
            },
        },
    }
    patcher = SimpleNamespace(
        model_options=original_model_options,
        wrappers={
            WrappersMP.DIFFUSION_MODEL: {
                "vdn_h3": [vdn],
                "vdn_h3_audio_adapter_scope": [scope],
            },
        },
        callbacks={"test_callback_type": {"test": [callback]}},
    )
    sigmas = torch.tensor([1.0, 0.5, 0.0])

    options = build_student_transformer_options(
        patcher,
        sigmas,
        video_shift=12.0,
        audio_shift=3.0,
    )

    assert options["sentinel"] == "preserved"
    assert options["sample_sigmas"] is sigmas
    assert options["minimax_h3_sigma_shift_video"] == 12.0
    assert options["minimax_h3_sigma_shift_audio"] == 3.0
    diffusion = options["wrappers"][WrappersMP.DIFFUSION_MODEL]
    assert diffusion["existing"] == [existing]
    assert diffusion["vdn_h3"] == [vdn]
    assert diffusion["vdn_h3_audio_adapter_scope"] == [scope]
    assert options["callbacks"]["test_callback_type"]["test"] == [callback]

    # The helper must not smuggle the student wrappers back into the persistent
    # ModelPatcher model_options object while preparing a direct training call.
    assert "vdn_h3" not in original_model_options["transformer_options"]["wrappers"][
        WrappersMP.DIFFUSION_MODEL
    ]

    visible = comfy.patcher_extension.get_all_wrappers(
        WrappersMP.DIFFUSION_MODEL, options)
    assert visible == [existing, vdn, scope]


def test_student_transformer_options_fail_closed_without_audio_scope_wrapper():
    patcher = SimpleNamespace(
        model_options={"transformer_options": {}},
        wrappers={
            WrappersMP.DIFFUSION_MODEL: {
                "vdn_h3": [_wrapper],
            },
        },
        callbacks={},
    )

    with pytest.raises(RuntimeError, match="vdn_h3_audio_adapter_scope"):
        build_student_transformer_options(
            patcher,
            torch.tensor([1.0, 0.0]),
            video_shift=12.0,
            audio_shift=3.0,
        )
