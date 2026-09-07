"""Direct-CUDA checkpoint loading for standalone audio-fix tools.

ComfyUI's public ``load_diffusion_model`` and ``load_clip`` helpers intentionally load
safetensors state dictionaries on CPU first.  In addition, current Comfy's AIMDO mmap
path can ignore a requested CUDA device in ``load_torch_file`` and still return
mmap-backed CPU tensors.  That is a poor fit for machines with large VRAM but
comparatively small host RAM.

These helpers bypass Comfy's checkpoint reader for the large standalone safetensors
files and read tensors with ``safetensors.safe_open(..., device="cuda")`` directly.
Load and offload devices are then pinned to the same GPU for the lifetime of the
standalone process.

This module is training-tool-only; normal ComfyUI nodes continue to use Comfy's own
model-management policy.
"""
from __future__ import annotations

import gc

import torch
from safetensors import safe_open

import comfy.model_management
import comfy.sd


def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("audio-fix direct-GPU tools require CUDA")
    device = comfy.model_management.get_torch_device()
    if torch.device(device).type != "cuda":
        raise RuntimeError(f"audio-fix direct-GPU tools expected CUDA, got {device}")
    return torch.device(device)


def _load_safetensors_cuda(path: str, device: torch.device, *, metadata: bool):
    """Own a CUDA state dict without creating a full CPU/mmap checkpoint copy."""
    if not path.lower().endswith((".safetensors", ".sft")):
        raise RuntimeError(
            f"direct-GPU loader only accepts safetensors checkpoints, got {path}")
    state = {}
    with torch.cuda.device(device):
        with safe_open(path, framework="pt", device="cuda") as handle:
            for key in handle.keys():
                state[key] = handle.get_tensor(key)
            file_metadata = handle.metadata() if metadata else None
    return (state, file_metadata) if metadata else state


def load_diffusion_model_direct_gpu(path: str):
    """Load a diffusion checkpoint without staging its state dict in host RAM."""
    device = cuda_device()
    state, metadata = _load_safetensors_cuda(path, device, metadata=True)
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
    # Keep the frozen production model resident in VRAM. The standalone trainer has
    # no reason to evict it to CPU between teacher/student calls.
    comfy.model_management.load_models_gpu([model], force_full_load=True)
    return model


def load_minimax_clip_direct_gpu(path: str):
    """Load the MiniMax-H3 Qwen3-VL text encoder directly into CUDA memory."""
    device = cuda_device()
    state = _load_safetensors_cuda(path, device, metadata=False)
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
