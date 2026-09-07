#!/usr/bin/env python3
"""GPU-resident entrypoint for the direct INT8/ConvRot audio-fix trainer.

The underlying trainer logic remains in ``audio_fix_int8_train.py``.  This entrypoint
replaces only its base-model loader so the 24+ GiB production checkpoint is read
straight into CUDA instead of first materializing a full CPU state dictionary.
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

    # The implementation calls this one public loader exactly once for the production
    # H3 base.  Replace it before main() so no giant CPU checkpoint staging occurs.
    impl.comfy.sd.load_diffusion_model = load_diffusion_model_direct_gpu
    impl.main()


if __name__ == "__main__":
    main()
