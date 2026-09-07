#!/usr/bin/env python3
"""Probe the already-installed production H3 before starting audio-fix training.

This downloads nothing. It verifies that Comfy's quantized Linear backward propagates
through the user's actual INT8/ConvRot qkv projection, then verifies the zero-initialized
sidecar receives the expected first-step gradient while the frozen base receives none.
"""
from __future__ import annotations

import argparse
import os
import sys


def _bootstrap():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--comfy-root", default=os.environ.get("COMFYUI_ROOT", "/home/toor/ComfyUI"))
    known, _ = parser.parse_known_args()
    root = os.path.abspath(os.path.expanduser(known.comfy_root))
    if not os.path.isfile(os.path.join(root, "comfy", "sd.py")):
        raise SystemExit(f"Not a ComfyUI checkout: {root}")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    sys.path.insert(0, repo)
    return root


COMFY_ROOT = _bootstrap()

import torch  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.sd  # noqa: E402

from vdn_h3.audio_fix_train import (  # noqa: E402
    TrainableAudioFixBank,
    comfy_quant_training_mode,
    validate_production_quant_targets,
)


def _resolve_base(path):
    path = os.path.expanduser(path)
    if os.path.isabs(path) and os.path.isfile(path):
        return os.path.realpath(path)
    candidate = os.path.join(COMFY_ROOT, "models", "diffusion_models", path)
    if os.path.isfile(candidate):
        return os.path.realpath(candidate)
    raise FileNotFoundError(
        f"Base model {path!r} not found as an absolute file or under "
        f"{os.path.join(COMFY_ROOT, 'models', 'diffusion_models')}")


def _grad_max(parameter):
    return 0.0 if parameter.grad is None else float(parameter.grad.detach().abs().max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", default=COMFY_ROOT)
    parser.add_argument("--base-model", required=True,
                        help="Existing production INT8/ConvRot MiniMax-H3 safetensors")
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()
    if args.rows < 4:
        raise SystemExit("--rows must be >= 4")

    base_path = _resolve_base(args.base_model)
    base = comfy.sd.load_diffusion_model(base_path, model_options={})
    if base is None:
        raise RuntimeError(f"Comfy could not load {base_path}")
    device = base.load_device
    comfy.model_management.load_models_gpu([base], force_full_load=True)
    dm = base.get_model_object("diffusion_model")
    targets = validate_production_quant_targets(dm)
    target = targets[0]
    module = dm.get_submodule(target)
    for parameter in dm.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    print(f"base: {base_path}")
    print(f"probe target: {target} ({module.in_features} -> {module.out_features})")

    with comfy_quant_training_mode():
        # Probe 1: the frozen quantized op must provide d(output)/d(input).
        x = torch.randn(args.rows, module.in_features, device=device, dtype=torch.bfloat16,
                        requires_grad=True)
        y = module(x)
        loss = y.float().square().mean()
        loss.backward()
        if x.grad is None or not torch.isfinite(x.grad).all() or float(x.grad.abs().max()) == 0.0:
            raise RuntimeError("INT8/ConvRot Linear did not propagate a finite nonzero input gradient")
        if any(parameter.grad is not None for parameter in dm.parameters()):
            raise RuntimeError("frozen base unexpectedly accumulated gradients")
        print(f"quantized input gradient: OK (max={float(x.grad.abs().max()):.6g})")

        # Probe 2: zero-init LoRA must be an exact no-op. On its first backward B gets
        # gradient while A is mathematically zero because B starts at zero.
        bank = TrainableAudioFixBank(dm, rank=4, alpha=4, targets=(target,)).to(device)
        bank.install()
        try:
            x2 = torch.randn(args.rows, module.in_features, device=device, dtype=torch.bfloat16)
            with bank.scope(1, args.rows - 1, enabled=True):
                base_out = None
                with bank.disabled():
                    base_out = module(x2).detach()
                out = module(x2)
                if not torch.equal(out, base_out):
                    raise RuntimeError("zero-initialized audio_fix changed the forward output")
                out.float().square().mean().backward()
            pair = bank.pairs[0]
            b_grad = _grad_max(pair.lora_B)
            a_grad = _grad_max(pair.lora_A)
            if not b_grad or not math_isfinite(b_grad):
                raise RuntimeError("zero-init audio_fix B received no finite first-step gradient")
            if a_grad != 0.0:
                raise RuntimeError(
                    f"zero-init audio_fix A gradient should be zero before B moves, got {a_grad}")
            if any(parameter.grad is not None for parameter in dm.parameters()):
                raise RuntimeError("frozen base accumulated gradients during sidecar probe")
            print(f"zero-init sidecar B gradient: OK (max={b_grad:.6g}); A=0 as expected")

            # Move B by one tiny gradient step and prove the A path becomes trainable.
            with torch.no_grad():
                pair.lora_B.add_(pair.lora_B.grad, alpha=-1e-4)
            bank.zero_grad(set_to_none=True)
            with bank.scope(1, args.rows - 1, enabled=True):
                module(x2).float().square().mean().backward()
            a_grad = _grad_max(pair.lora_A)
            if not a_grad or not math_isfinite(a_grad):
                raise RuntimeError("audio_fix A received no finite gradient after B moved off zero")
            print(f"sidecar A gradient after one B update: OK (max={a_grad:.6g})")
        finally:
            bank.uninstall()

    print("PROBE PASSED: existing production INT8/ConvRot H3 is usable for sidecar training")


def math_isfinite(value):
    import math
    return math.isfinite(float(value))


if __name__ == "__main__":
    main()
