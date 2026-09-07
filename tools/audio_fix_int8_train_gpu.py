#!/usr/bin/env python3
"""GPU-resident entrypoint for the direct INT8/ConvRot audio-fix trainer.

The underlying trainer logic remains in ``audio_fix_int8_train.py``. This entrypoint
changes only standalone loading policy:

- the 24+ GiB production H3 checkpoint is read straight into CUDA instead of first
  materializing a full CPU state dictionary;
- VDN branch policy is changed from the implementation's conservative hardcoded
  ``stream`` mode to ``auto`` so a large-VRAM workstation can keep the branch resident
  when the existing policy says it fits.

Normal ComfyUI node behavior is untouched.
"""
from __future__ import annotations

import importlib.util
import os
import sys


def _load_impl():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    path = os.path.join(here, "audio_fix_int8_train.py")
    spec = importlib.util.spec_from_file_location("audio_fix_int8_train_impl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trainer implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    impl = _load_impl()
    from vdn_h3.direct_gpu_load import load_diffusion_model_direct_gpu

    # Avoid Comfy's ordinary CPU-first state-dict staging for the production base.
    impl.comfy.sd.load_diffusion_model = load_diffusion_model_direct_gpu

    # The implementation deliberately used "stream" while the training graph was
    # being brought up. For this GPU-resident entrypoint, hand the decision back to
    # the existing VRAM-aware VDN policy. On a 96 GiB card it may select resident BF16;
    # if that does not fit it safely falls back to the policy's streamed representation.
    apply_vdn = impl._apply_vdn_audio_safe

    def apply_vdn_gpu_auto(model, vdn_checkpoint, strength, branch_weights,
                           attention_backend, verbose, **kwargs):
        return apply_vdn(
            model,
            vdn_checkpoint,
            strength,
            "auto",
            attention_backend,
            verbose,
            **kwargs,
        )

    impl._apply_vdn_audio_safe = apply_vdn_gpu_auto
    impl.main()


if __name__ == "__main__":
    main()
