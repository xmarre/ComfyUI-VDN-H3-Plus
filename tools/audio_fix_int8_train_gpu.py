#!/usr/bin/env python3
"""Low-host-RAM GPU entrypoint for the direct INT8/ConvRot audio-fix trainer.

The underlying trainer logic remains in ``audio_fix_int8_train.py``. This entrypoint
changes only standalone loading/runtime policy for a workstation with substantially
more VRAM than host RAM:

- the 24+ GiB production H3 checkpoint is read straight into CUDA instead of first
  materializing a full CPU state dictionary;
- the VDN linear branch is forced to bounded streaming, preferring the released
  INT8/ConvRot branch when present, so the ordinary resident-BF16 path cannot first
  clone the complete branch into host RAM;
- retained CUDA scratch/prefetch buffers stay enabled so one-block-ahead branch I/O
  can overlap execution without turning host RAM into a model-weight cache.

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


def _training_branch_policy(vdn_policy, path, free_bytes):
    """Use bounded streaming and prefer the native INT8 branch when available."""
    plain, quant = vdn_policy.branch_candidates(path)
    have_quant = os.path.isfile(quant)
    selected = quant if have_quant else plain
    selected_size = os.path.getsize(selected) if os.path.isfile(selected) else 0
    print(
        "[audio-fix GPU] VDN branch policy: "
        f"{'int8_convrot' if have_quant else 'bf16'} / stream "
        f"({free_bytes / (1 << 30):.1f} GiB free; "
        f"selected {selected_size / (1 << 30):.2f} GiB); "
        "full branch will not be cloned into host RAM",
        flush=True,
    )
    return "stream", have_quant


def main():
    impl = _load_impl()
    from vdn_h3.direct_gpu_load import load_diffusion_model_direct_gpu
    import vdn_h3.policy as vdn_policy

    # Avoid Comfy's ordinary CPU-first state-dict staging for the production base.
    impl.comfy.sd.load_diffusion_model = load_diffusion_model_direct_gpu

    # The ordinary resident VDN path intentionally materializes BF16 branch tensors as
    # CPU Parameters before Comfy migrates them. That is correct for normal Comfy model
    # management but wrong for this 96-GiB-VRAM / low-host-RAM standalone trainer.
    # Keep branch weights bounded/streamed and prefer the checkpoint's native INT8
    # representation. Retained CUDA scratch enables one-block lookahead without
    # retaining checkpoint weights in host RAM.
    apply_vdn = impl._apply_vdn_audio_safe

    def apply_vdn_gpu_low_host_ram(model, vdn_checkpoint, strength, branch_weights,
                                   attention_backend, verbose, **kwargs):
        del branch_weights
        original_auto_policy = vdn_policy.auto_branch_policy
        vdn_policy.auto_branch_policy = (
            lambda path, free: _training_branch_policy(vdn_policy, path, free)
        )
        kwargs = dict(kwargs)
        kwargs["retain_buffers"] = "on"
        try:
            return apply_vdn(
                model,
                vdn_checkpoint,
                strength,
                "auto",
                attention_backend,
                verbose,
                **kwargs,
            )
        finally:
            vdn_policy.auto_branch_policy = original_auto_policy

    impl._apply_vdn_audio_safe = apply_vdn_gpu_low_host_ram
    impl.main()


if __name__ == "__main__":
    main()
