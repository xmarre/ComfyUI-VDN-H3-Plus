"""Model-option plumbing for direct production-stack audio-fix training calls.

ComfyUI normally copies ModelPatcher wrappers/callbacks into the per-sampling
``transformer_options`` before the diffusion model is invoked.  The standalone
INT8 trainer calls MiniMax-H3 directly, so it must perform that merge explicitly;
otherwise VDN's layout wrapper and the generated-audio adapter-scope wrapper are
silently absent from the student forward.
"""
from __future__ import annotations


REQUIRED_STUDENT_DIFFUSION_WRAPPERS = (
    "vdn_h3",
    "vdn_h3_audio_adapter_scope",
)


def build_student_transformer_options(
    model_patcher,
    sigmas,
    *,
    video_shift: float,
    audio_shift: float,
):
    """Build the transformer-options object used by direct student H3 calls.

    This mirrors the wrapper/callback merge performed by
    ``comfy.sampler_helpers.prepare_model_patcher`` without invoking sampling or
    model-loading machinery.  The returned object is detached from
    ``model_patcher.model_options`` but intentionally persists for one training
    rollout so normal per-run H3 layout caching has the same lifetime as sampling.
    """
    import comfy.model_patcher
    import comfy.patcher_extension
    from comfy.patcher_extension import WrappersMP

    model_options = comfy.model_patcher.create_model_options_clone(
        model_patcher.model_options)
    transformer_options = model_options.setdefault("transformer_options", {})

    comfy.patcher_extension.merge_nested_dicts(
        transformer_options.setdefault("wrappers", {}),
        model_patcher.wrappers,
        copy_dict1=False,
    )
    comfy.patcher_extension.merge_nested_dicts(
        transformer_options.setdefault("callbacks", {}),
        model_patcher.callbacks,
        copy_dict1=False,
    )

    diffusion_groups = transformer_options.get("wrappers", {}).get(
        WrappersMP.DIFFUSION_MODEL, {})
    missing = [
        key for key in REQUIRED_STUDENT_DIFFUSION_WRAPPERS
        if not diffusion_groups.get(key)
    ]
    if missing:
        raise RuntimeError(
            "audio-fix student is missing required DIFFUSION_MODEL wrappers: "
            + ", ".join(missing)
        )

    transformer_options["minimax_h3_sigma_shift_video"] = float(video_shift)
    transformer_options["minimax_h3_sigma_shift_audio"] = float(audio_shift)
    transformer_options["sample_sigmas"] = sigmas
    return transformer_options


__all__ = [
    "REQUIRED_STUDENT_DIFFUSION_WRAPPERS",
    "build_student_transformer_options",
]
