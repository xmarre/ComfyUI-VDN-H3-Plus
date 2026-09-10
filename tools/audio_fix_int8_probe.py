#!/usr/bin/env python3
"""Probe the installed INT8/ConvRot H3 before starting audio-fix training.

This downloads nothing. The cheap probe verifies that Comfy's quantized Linear
backward propagates through the actual INT8/ConvRot qkv projection and that the
zero-initialized sidecar receives the expected first-step gradient while the frozen
base receives none.

With ``--vdn-checkpoint`` it additionally performs a small direct MiniMax-H3 forward
through the exact VDN/Stage-B/Turbo ModelPatcher wrapper stack used by the standalone
trainer. It also compares ordinary inference execution against the autograd-safe
training execution for both dense H3 and the finished VDN/Turbo student. A material
training/inference mismatch is a hard stop before optimization.
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager


def _bootstrap():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--comfy-root",
        default=os.environ.get("COMFYUI_ROOT", "/home/toor/ComfyUI"),
    )
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
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402

from vdn_h3.audio_fix_model_options import build_student_transformer_options  # noqa: E402
from vdn_h3.audio_fix_train import (  # noqa: E402
    TrainableAudioFixBank,
    comfy_quant_training_mode,
    validate_production_quant_targets,
)
from vdn_h3.audio_node import _apply_vdn_audio_safe  # noqa: E402


VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
AUDIO_STREAMS = 2
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
AUDIO_SCALE = VIDEO_SHIFT / AUDIO_SHIFT


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


def _finite_delta(left, right):
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise RuntimeError("full-stack probe produced a non-finite H3 output")
    return float((left.float() - right.float()).abs().max())


def _relative_rms(reference, candidate):
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise RuntimeError("training/inference parity probe produced a non-finite H3 output")
    ref = reference.float()
    cand = candidate.float()
    numerator = (cand - ref).square().mean().sqrt()
    denominator = ref.square().mean().sqrt().clamp_min(1e-8)
    return float(numerator / denominator), float((cand - ref).abs().max())


def _assert_pair_parity(label, inference, training, limit):
    video_rms, video_max = _relative_rms(inference[0], training[0])
    audio_rms, audio_max = _relative_rms(inference[1], training[1])
    print(
        f"{label} training/inference parity: "
        f"video_rel_rms={video_rms:.6g} video_max={video_max:.6g} "
        f"audio_rel_rms={audio_rms:.6g} audio_max={audio_max:.6g}",
        flush=True,
    )
    worst = max(video_rms, audio_rms)
    if worst > float(limit):
        raise RuntimeError(
            f"{label} training/inference relative RMS mismatch {worst:.6g} exceeds "
            f"the probe limit {float(limit):.6g}; do not train against a materially "
            "different execution path"
        )


@contextmanager
def _student_role(student, device):
    student.patch_model(device_to=device, load_weights=False)
    student.pre_run()
    try:
        yield student.get_model_object("diffusion_model")
    finally:
        student.cleanup()
        student.unpatch_model(device_to=device, unpatch_weights=False)


def _direct_h3_output(dm, video, audio, sigma, context, transformer_options):
    timestep = torch.tensor([float(sigma) * 1000.0], device=video.device, dtype=torch.float32)
    output = dm(
        [video, audio],
        timestep,
        context,
        transformer_options,
        minimax_payload={"audio_scale": AUDIO_SCALE},
    )
    if not isinstance(output, (list, tuple)) or len(output) != 2:
        raise RuntimeError("MiniMax H3 did not return [video, audio]")
    return output[0].detach(), output[1].detach()


def _full_stack_probe(base, base_dm, device, args):
    if not args.vdn_checkpoint:
        return

    print("full-stack direct-call probe: building VDN/Stage-B/Turbo student", flush=True)
    student = _apply_vdn_audio_safe(
        base,
        args.vdn_checkpoint,
        {"default": args.stage_b_strength, "turbo": args.turbo_strength},
        "stream",
        "grouped",
        True,
        audio_adapter_strength=1.0,
        conditioning_adapter_strength=1.0,
        audio_video_context_strength=1.0,
        conditioning_video_context_strength=1.0,
        audio_fix_strength=0.0,
        apply_turbo_adapter=True,
        retain_buffers="off",
        global_gate_mode=args.global_gate_mode,
        adapter_ablation="none",
        fast_kernels=False,
    )[0]

    sigmas = comfy.samplers.calculate_sigmas(
        base.model.model_sampling, "simple", 10
    ).to(device=device, dtype=torch.float32)
    if sigmas.numel() != 11 or float(sigmas[-1]) != 0.0:
        raise RuntimeError(f"unexpected 10-step simple sigma table: {sigmas.tolist()}")
    transformer_options = build_student_transformer_options(
        student,
        sigmas,
        video_shift=VIDEO_SHIFT,
        audio_shift=AUDIO_SHIFT,
    )
    dense_options = {
        "minimax_h3_sigma_shift_video": VIDEO_SHIFT,
        "minimax_h3_sigma_shift_audio": AUDIO_SHIFT,
        "sample_sigmas": sigmas,
    }

    generator = torch.Generator(device=device).manual_seed(args.seed + 4049)
    video = torch.randn(
        1,
        VIDEO_CHANNELS,
        args.full_stack_video_frames,
        args.full_stack_latent_height,
        args.full_stack_latent_width,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    # MiniMax-H3 audio latents are stereo: [B, 32, 2, T]. PackedLayout reserves
    # 2*T target-audio rows, so a mono [B, 32, 1, T] probe cannot represent H3 geometry.
    audio = torch.randn(
        1,
        AUDIO_CHANNELS,
        AUDIO_STREAMS,
        args.full_stack_audio_frames,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    # Hidden-width context skips the token-refiner and keeps the probe focused on the
    # packed H3 + VDN + released adapter graph. Real prompt embeddings are exercised by
    # the subsequent one-step trainer smoke run.
    context = torch.randn(
        1,
        args.full_stack_text_tokens,
        int(base_dm.hidden_size),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    sigma = float(sigmas[min(4, sigmas.numel() - 2)])

    # First execute exactly as deployment inference does. Then execute the same graph
    # under the trainer's autograd-safe global mode. The latter swaps in functional
    # RMS/RoPE and differentiable VDN recurrence implementations, which must remain
    # numerically equivalent enough to train a correction intended for inference.
    with torch.no_grad():
        dense_inference = _direct_h3_output(
            base_dm, video, audio, sigma, context, dense_options)
        with _student_role(student, device) as student_dm:
            student_inference = _direct_h3_output(
                student_dm, video, audio, sigma, context, transformer_options)

    with torch.no_grad(), comfy_quant_training_mode():
        dense_training = _direct_h3_output(
            base_dm, video, audio, sigma, context, dense_options)
        with _student_role(student, device) as student_dm:
            student_training = _direct_h3_output(
                student_dm, video, audio, sigma, context, transformer_options)

    _assert_pair_parity(
        "dense H3",
        dense_inference,
        dense_training,
        args.max_training_parity_rel_rms,
    )
    _assert_pair_parity(
        "VDN/Turbo student",
        student_inference,
        student_training,
        args.max_training_parity_rel_rms,
    )

    video_delta = _finite_delta(student_training[0], dense_training[0])
    audio_delta = _finite_delta(student_training[1], dense_training[1])
    print(
        "full-stack direct-call delta: "
        f"video_max={video_delta:.6g} audio_max={audio_delta:.6g}",
        flush=True,
    )
    if video_delta == 0.0 and audio_delta == 0.0:
        raise RuntimeError(
            "VDN/Stage-B/Turbo student was bit-identical to dense H3; the direct-call "
            "wrapper/adapter stack did not affect the model and training must not start"
        )

    # build_student_transformer_options proves the required wrapper keys are present;
    # MiniMaxH3Model.forward consumes exactly that wrapper collection. The nonzero
    # student/base delta additionally proves the finished patched student is active.
    print(
        "full-stack direct-call execution: OK "
        f"(Stage-B={args.stage_b_strength:g}, Turbo={args.turbo_strength:g}, "
        f"gate={args.global_gate_mode})",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", default=COMFY_ROOT)
    parser.add_argument(
        "--base-model",
        required=True,
        help="Existing production INT8/ConvRot MiniMax-H3 safetensors",
    )
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument(
        "--vdn-checkpoint",
        default=None,
        help="Optional released VDN stage; enables the direct full-stack wrapper probe",
    )
    parser.add_argument("--stage-b-strength", type=float, default=1.0)
    parser.add_argument("--turbo-strength", type=float, default=0.75)
    parser.add_argument(
        "--global-gate-mode",
        choices=("checkpoint", "video_only"),
        default="video_only",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full-stack-video-frames", type=int, default=8)
    parser.add_argument("--full-stack-audio-frames", type=int, default=32)
    parser.add_argument("--full-stack-latent-height", type=int, default=8)
    parser.add_argument("--full-stack-latent-width", type=int, default=8)
    parser.add_argument("--full-stack-text-tokens", type=int, default=8)
    parser.add_argument(
        "--max-training-parity-rel-rms",
        type=float,
        default=0.02,
        help="Hard relative-RMS limit for inference vs autograd-safe full-stack output",
    )
    args = parser.parse_args()

    if args.rows < 4:
        raise SystemExit("--rows must be >= 4")
    if not 0.0 <= args.stage_b_strength <= 2.0:
        raise SystemExit("--stage-b-strength must be in [0, 2]")
    if not 0.0 < args.turbo_strength <= 2.0:
        raise SystemExit("--turbo-strength must be in (0, 2]; Turbo-off is not this probe path")
    if args.full_stack_video_frames < 2 or args.full_stack_audio_frames < 2:
        raise SystemExit("full-stack video/audio frame counts must be >= 2")
    if (
        args.full_stack_latent_height < 2
        or args.full_stack_latent_width < 2
        or args.full_stack_latent_height % 2
        or args.full_stack_latent_width % 2
    ):
        raise SystemExit("full-stack latent height/width must be positive even values >= 2")
    if args.full_stack_text_tokens < 1:
        raise SystemExit("--full-stack-text-tokens must be >= 1")
    if not 0.0 < args.max_training_parity_rel_rms <= 1.0:
        raise SystemExit("--max-training-parity-rel-rms must be in (0, 1]")

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
        x = torch.randn(
            args.rows,
            module.in_features,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
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
                    f"zero-init audio_fix A gradient should be zero before B moves, got {a_grad}"
                )
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

    print("INT8/ConvRot sidecar gradient probe: PASSED", flush=True)
    _full_stack_probe(base, dm, device, args)
    print("PROBE PASSED: production H3 is structurally usable for audio-fix training", flush=True)


def math_isfinite(value):
    import math

    return math.isfinite(float(value))


if __name__ == "__main__":
    main()
