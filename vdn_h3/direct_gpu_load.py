"""Direct-CUDA checkpoint loading for standalone audio-fix tools.

ComfyUI's public ``load_diffusion_model`` and ``load_clip`` helpers intentionally load
safetensors state dictionaries on CPU first.  That is a poor fit for machines with
large VRAM but comparatively small host RAM.  These helpers keep the large checkpoint
state dictionaries on CUDA from the first safetensors read and pin both load/offload
devices to the same GPU for the lifetime of the standalone process.

This module is training-tool-only; normal ComfyUI nodes continue to use Comfy's own
model-management policy.
"""
from __future__ import annotations

import gc

import torch

import comfy.model_management
import comfy.sd
import comfy.utils


def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("audio-fix direct-GPU tools require CUDA")
    return comfy.model_management.get_torch_device()


def load_diffusion_model_direct_gpu(path: str):
    """Load a diffusion checkpoint without staging its state dict in host RAM."""
    device = cuda_device()
    state, metadata = comfy.utils.load_torch_file(
        path,
        safe_load=True,
        device=device,
        return_metadata=True,
    )
    try:
        model = comfy.sd.load_diffusion_model_state_dict(
            state,
            model_options={
                "load_device": device,
                "offload_device": device,
            },
            metadata=metadata,
        )
    finally:
        del state
        gc.collect()
    if model is None:
        raise RuntimeError(f"Comfy could not detect diffusion model type for {path}")
    # Keep the frozen production model resident in VRAM.  The standalone trainer has
    # no reason to evict it to CPU between teacher/student calls.
    comfy.model_management.load_models_gpu([model], force_full_load=True)
    return model


def load_minimax_clip_direct_gpu(path: str):
    """Load the MiniMax-H3 Qwen3-VL text encoder directly into CUDA memory."""
    device = cuda_device()
    state = comfy.utils.load_torch_file(
        path,
        safe_load=True,
        device=device,
    )
    try:
        clip = comfy.sd.load_text_encoder_state_dicts(
            [state],
            embedding_directory=None,
            clip_type=comfy.sd.CLIPType.MINIMAX,
            model_options={
                "load_device": device,
                "offload_device": device,
                "initial_device": device,
            },
        )
    finally:
        del state
        gc.collect()
    comfy.model_management.load_models_gpu([clip.patcher], force_full_load=True)
    return clip
