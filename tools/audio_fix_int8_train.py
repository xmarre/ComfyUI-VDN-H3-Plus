#!/usr/bin/env python3
"""Train ``audio_fix`` directly on the user's installed Comfy INT8/ConvRot H3.

No H3 or text-encoder weights are downloaded by this program. Prompt embeddings must
first be cached with ``tools/audio_fix_cache_prompts.py`` using the existing Comfy text
encoder. The frozen teacher is the same quantized H3 base without VDN/adapters; the
student is VDN + Stage-B 1.0 + Turbo 1.0 plus the trainable audio-only sidecar.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import sys
import time
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
import torch.nn.functional as F  # noqa: E402
import comfy.cli_args  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree  # noqa: E402

from vdn_h3.audio_fix_train import (  # noqa: E402
    TrainableAudioFixBank,
    comfy_quant_training_mode,
    generated_audio_span,
    install_block_checkpointing,
    res_multistep_update,
    restore_block_checkpointing,
    validate_production_quant_targets,
)
from vdn_h3.audio_node import _apply_vdn_audio_safe  # noqa: E402


AUDIO_CHANNELS = 32
VIDEO_CHANNELS = 24
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


def _prompt_files(directory):
    files = sorted(glob.glob(os.path.join(os.path.abspath(directory), "*.pt")))
    if not files:
        raise FileNotFoundError(f"No cached prompt .pt files under {directory}")
    return files


def _load_prompt(path, device):
    row = torch.load(path, map_location="cpu", weights_only=True)
    if row.get("format") != 1:
        raise RuntimeError(f"Unsupported prompt cache format in {path}")
    context = row.get("context")
    tags = row.get("text_token_tags")
    if not isinstance(context, torch.Tensor) or context.ndim != 3 or context.shape[0] != 1:
        raise RuntimeError(f"Invalid H3 context in {path}")
    if not isinstance(tags, torch.Tensor) or tags.numel() != context.shape[1]:
        raise RuntimeError(f"Invalid H3 token tags in {path}")
    return (
        context.to(device=device, dtype=torch.bfloat16, non_blocking=True),
        tags.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1),
        str(row.get("prompt", "")),
    )


def _optimizer(bank):
    return ProdigyPlusScheduleFree(
        bank.parameters(),
        lr=1.0,
        betas=(0.95, 0.99),
        weight_decay=0.0,
        d0=1e-6,
        d_coef=1.0,
        d_limiter=True,
        prodigy_steps=0,
        schedulefree_c=0.0,
        eps=1e-8,
        factored=True,
        factored_fp32=True,
        use_stableadamw=True,
        use_schedulefree=True,
        split_groups=False,
        use_speed=False,
        stochastic_rounding=False,
        fused_back_pass=False,
        use_cautious=False,
        use_grams=False,
        use_adopt=False,
        use_orthograd=False,
        use_focus=False,
    )


def _tree_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _tree_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_tree_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_tree_cpu(v) for v in value)
    return value


def _save_state(path, step, bank, optimizer, generator, prompt_cursor):
    tmp = path + ".tmp"
    payload = {
        "format": 1,
        "stage": "audio_fix_int8",
        "step": int(step),
        "bank": _tree_cpu(bank.state_dict()),
        "optimizer": _tree_cpu(optimizer.state_dict()),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "noise_rng": generator.get_state().cpu(),
        "prompt_cursor": int(prompt_cursor),
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _resume(path, bank, optimizer, generator):
    if not os.path.isfile(path):
        return 0, 0
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != 1 or payload.get("stage") != "audio_fix_int8":
        raise RuntimeError(f"Unsupported train state {path}")
    bank.load_state_dict(payload["bank"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if hasattr(optimizer, "train"):
        optimizer.train()
    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state(payload["cuda_rng"])
    generator.set_state(payload["noise_rng"])
    return int(payload["step"]), int(payload.get("prompt_cursor", 0))


@contextmanager
def _student_role(student, device):
    """Install only the student's object patches/injections; keep base weights resident."""
    student.patch_model(device_to=device, load_weights=False)
    student.pre_run()
    try:
        yield student.get_model_object("diffusion_model")
    finally:
        student.cleanup()
        student.unpatch_model(device_to=device, unpatch_weights=False)


def _model_output(dm, video, audio, sigma, context, tags, sigmas):
    timestep = torch.as_tensor([float(sigma) * 1000.0], device=video.device,
                               dtype=torch.float32)
    transformer_options = {
        "minimax_h3_sigma_shift_video": VIDEO_SHIFT,
        "minimax_h3_sigma_shift_audio": AUDIO_SHIFT,
        "sample_sigmas": sigmas,
    }
    payload = {
        "audio_scale": AUDIO_SCALE,
        "text_token_tags": tags,
    }
    output = dm(
        [video, audio], timestep, context, transformer_options,
        minimax_payload=payload,
    )
    if not isinstance(output, (list, tuple)) or len(output) != 2:
        raise RuntimeError("MiniMax H3 did not return [video, audio] model outputs")
    return output[0].float(), output[1].float()


def _x0(video, audio, out_v, out_a, sigma):
    sigma = torch.as_tensor(sigma, device=video.device, dtype=torch.float32)
    # MiniMax H3 uses Comfy's CONST parameterisation: denoised = input - output*sigma.
    return video.float() - out_v * sigma, audio.float() - out_a * sigma


def _rollout_student(student, bank, video, audio, upto, context, tags, sigmas,
                     audio_span, device):
    old_v = old_a = old_down = old_sigma = None
    with torch.no_grad(), _student_role(student, device), bank.scope(
            *audio_span, enabled=True, checkpoint_blocks=False):
        dm = student.get_model_object("diffusion_model")
        for index in range(upto):
            out_v, out_a = _model_output(
                dm, video, audio, sigmas[index], context, tags, sigmas)
            den_v, den_a = _x0(video, audio, out_v, out_a, sigmas[index])
            next_v = res_multistep_update(
                video, den_v, sigmas[index], sigmas[index + 1],
                previous_denoised=old_v, previous_sigma_down=old_down,
                previous_sigma=old_sigma)
            next_a = res_multistep_update(
                audio, den_a, sigmas[index], sigmas[index + 1],
                previous_denoised=old_a, previous_sigma_down=old_down,
                previous_sigma=old_sigma)
            old_v, old_a = den_v.detach(), den_a.detach()
            old_down, old_sigma = sigmas[index + 1], sigmas[index]
            video, audio = next_v.detach(), next_a.detach()
    return video, audio


def _teacher_target(base_dm, bank, video, audio, index, context, tags, sigmas):
    with torch.no_grad(), bank.disabled():
        out_v, out_a = _model_output(
            base_dm, video, audio, sigmas[index], context, tags, sigmas)
        return tuple(x.detach() for x in _x0(video, audio, out_v, out_a, sigmas[index]))


def _frozen_student_target(student, bank, video, audio, index, context, tags, sigmas,
                           device):
    with torch.no_grad(), _student_role(student, device), bank.disabled():
        dm = student.get_model_object("diffusion_model")
        out_v, out_a = _model_output(
            dm, video, audio, sigmas[index], context, tags, sigmas)
        return tuple(x.detach() for x in _x0(video, audio, out_v, out_a, sigmas[index]))


def _save_adapter(output_root, step, bank, base_path, vdn_checkpoint, latent_shape,
                  audio_t):
    path = os.path.join(output_root, f"audio_fix_step_{step:06d}")
    if os.path.exists(path):
        shutil.rmtree(path)
    bank.save_adapter(
        path,
        step=step,
        metadata={
            "base_model": os.path.realpath(base_path),
            "vdn_checkpoint": vdn_checkpoint,
            "sampler": "res_multistep",
            "sigma_schedule": "simple",
            "sampler_steps": 10,
            "video_shift": VIDEO_SHIFT,
            "audio_shift": AUDIO_SHIFT,
            "video_latent_shape": list(latent_shape),
            "audio_latent_frames": int(audio_t),
            "teacher": "same production INT8/ConvRot H3 with VDN/adapters disabled",
        },
    )
    print(f"adapter checkpoint -> {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", default=COMFY_ROOT)
    parser.add_argument("--base-model", required=True,
                        help="Existing production INT8/ConvRot MiniMax-H3 safetensors")
    parser.add_argument("--vdn-checkpoint", required=True,
                        help="Existing VDN stage name under Comfy models/vdn")
    parser.add_argument("--prompt-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latent-height", type=int, required=True,
                        help="Production VIDEO latent height (not pixel height)")
    parser.add_argument("--latent-width", type=int, required=True,
                        help="Production VIDEO latent width (not pixel width)")
    parser.add_argument("--video-latent-frames", type=int, default=52,
                        help="7-second H3 chunk at 24 fps -> 52 video latent frames")
    parser.add_argument("--audio-latent-frames", type=int, default=292,
                        help="7-second H3 chunk -> 292 audio latent frames")
    parser.add_argument("--train-steps", type=int, default=250)
    parser.add_argument("--sampler-steps", type=int, default=10)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-preserve-weight", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Force one optimizer step and save step 0/1")
    args = parser.parse_args()

    if args.sampler_steps != 10:
        raise SystemExit("Production audio-fix contract requires --sampler-steps 10")
    if args.video_latent_frames != 52 or args.audio_latent_frames != 292:
        raise SystemExit(
            "Production 7-second contract requires --video-latent-frames 52 and "
            "--audio-latent-frames 292")
    if args.latent_height < 2 or args.latent_width < 2:
        raise SystemExit("latent height/width must be >= 2")
    if args.latent_height % 2 or args.latent_width % 2:
        raise SystemExit("MiniMax H3 latent height/width must be even")
    if args.save_every < 1:
        raise SystemExit("--save-every must be >= 1")

    # Never let the training process silently enter Comfy compiler/inference-only paths.
    comfy.cli_args.args.disable_comfy_compiler = True
    base_path = _resolve_base(args.base_model)
    prompt_files = _prompt_files(args.prompt_cache_dir)
    output_root = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_root, exist_ok=True)
    state_path = os.path.join(output_root, "train_state.pt")
    metrics_path = os.path.join(output_root, "metrics.jsonl")

    max_steps = 1 if args.smoke else int(args.train_steps)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    base = comfy.sd.load_diffusion_model(base_path, model_options={})
    if base is None:
        raise RuntimeError(f"Comfy could not load {base_path}")
    device = base.load_device
    comfy.model_management.load_models_gpu([base], force_full_load=True)
    base_dm = base.get_model_object("diffusion_model")
    validate_production_quant_targets(base_dm)
    for parameter in base_dm.parameters():
        parameter.requires_grad_(False)

    # Build the exact released student path in bypass mode. No audio_fix is read from
    # the VDN stage during training; the new bank below is the only trainable adapter.
    student = _apply_vdn_audio_safe(
        base,
        args.vdn_checkpoint,
        {"default": 1.0, "turbo": 1.0},
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
        global_gate_mode="checkpoint",
        adapter_ablation="none",
        fast_kernels=False,
    )[0]

    bank = TrainableAudioFixBank(base_dm, rank=args.rank, alpha=args.alpha).to(device)
    bank.install()
    checkpoint_originals = install_block_checkpointing(base_dm, bank)
    optimizer = _optimizer(bank)
    if hasattr(optimizer, "train"):
        optimizer.train()
    generator = torch.Generator(device=device).manual_seed(args.seed + 1009)

    step = prompt_cursor = 0
    if not args.no_resume and not args.smoke:
        step, prompt_cursor = _resume(state_path, bank, optimizer, generator)
        if step:
            print(f"resumed step {step} from {state_path}", flush=True)

    sigmas = comfy.samplers.calculate_sigmas(
        base.model.model_sampling, "simple", args.sampler_steps).to(
            device=device, dtype=torch.float32)
    if sigmas.numel() != 11 or float(sigmas[-1]) != 0.0:
        raise RuntimeError(f"Unexpected production simple sigma table: {sigmas.tolist()}")

    latent_shape = (
        1, VIDEO_CHANNELS, args.video_latent_frames,
        args.latent_height, args.latent_width,
    )
    early = {0, 1, 2, 4, 8, 16, 32}

    try:
        with comfy_quant_training_mode():
            if step == 0 and (args.smoke or 0 in early):
                if hasattr(optimizer, "eval"):
                    optimizer.eval()
                _save_adapter(output_root, 0, bank, base_path, args.vdn_checkpoint,
                              latent_shape, args.audio_latent_frames)
                if hasattr(optimizer, "train"):
                    optimizer.train()

            while step < max_steps:
                prompt_path = prompt_files[prompt_cursor % len(prompt_files)]
                prompt_cursor += 1
                context, tags, prompt = _load_prompt(prompt_path, device)
                audio_span = generated_audio_span(
                    context.shape[1], args.video_latent_frames,
                    args.latent_height, args.latent_width, args.audio_latent_frames)
                # Uniformly train every actual model-call location in the 10-NFE grid.
                train_index = random.Random(args.seed + step * 1000003).randrange(args.sampler_steps)
                video = torch.randn(latent_shape, generator=generator, device=device,
                                    dtype=torch.float32)
                audio = torch.randn(
                    1, AUDIO_CHANNELS, 1, args.audio_latent_frames,
                    generator=generator, device=device, dtype=torch.float32)

                started = time.time()
                video, audio = _rollout_student(
                    student, bank, video, audio, train_index, context, tags, sigmas,
                    audio_span, device)
                teacher_v, teacher_a = _teacher_target(
                    base_dm, bank, video, audio, train_index, context, tags, sigmas)
                frozen_v, _frozen_a = _frozen_student_target(
                    student, bank, video, audio, train_index, context, tags, sigmas, device)

                optimizer.zero_grad(set_to_none=True)
                with _student_role(student, device), bank.scope(
                        *audio_span, enabled=True, checkpoint_blocks=True):
                    dm = student.get_model_object("diffusion_model")
                    out_v, out_a = _model_output(
                        dm, video, audio, sigmas[train_index], context, tags, sigmas)
                    student_v, student_a = _x0(
                        video, audio, out_v, out_a, sigmas[train_index])
                    # Audio is carried at video_shift/audio_shift = 4. Compare native
                    # x0 units so loss magnitude is independent of that sampler carry.
                    audio_loss = F.mse_loss(student_a / AUDIO_SCALE, teacher_a / AUDIO_SCALE)
                    video_loss = F.mse_loss(student_v, frozen_v)
                    loss = audio_loss + float(args.video_preserve_weight) * video_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"non-finite loss at step {step + 1}, grid {train_index}")
                    loss.backward()

                grad_sq = 0.0
                nonzero = 0
                for parameter in bank.parameters():
                    if parameter.grad is None:
                        continue
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("non-finite audio-fix gradient")
                    maximum = float(parameter.grad.detach().abs().max())
                    if maximum > 0:
                        nonzero += 1
                    grad_sq += float(parameter.grad.detach().float().pow(2).sum())
                if nonzero == 0:
                    raise RuntimeError(
                        "audio_fix received no nonzero gradients through the full INT8/VDN graph")
                optimizer.step()
                step += 1
                elapsed = time.time() - started

                row = {
                    "step": step,
                    "prompt_file": os.path.basename(prompt_path),
                    "prompt": prompt,
                    "rollout_index": train_index,
                    "loss": float(loss.detach()),
                    "audio_teacher_loss": float(audio_loss.detach()),
                    "video_preserve_loss": float(video_loss.detach()),
                    "grad_norm": grad_sq ** 0.5,
                    "nonzero_grad_tensors": nonzero,
                    "seconds": elapsed,
                    "peak_gib": torch.cuda.max_memory_allocated(device) / (1 << 30),
                }
                print(json.dumps(row, ensure_ascii=False), flush=True)
                with open(metrics_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

                should_save = args.smoke or step in early or step % args.save_every == 0 \
                    or step == max_steps
                if should_save:
                    if hasattr(optimizer, "eval"):
                        optimizer.eval()
                    _save_adapter(
                        output_root, step, bank, base_path, args.vdn_checkpoint,
                        latent_shape, args.audio_latent_frames)
                    if hasattr(optimizer, "train"):
                        optimizer.train()
                    _save_state(
                        state_path, step, bank, optimizer, generator, prompt_cursor)

                del context, tags, video, audio, teacher_v, teacher_a, frozen_v
                del out_v, out_a, student_v, student_a, loss, audio_loss, video_loss
                torch.cuda.empty_cache()
    finally:
        restore_block_checkpointing(checkpoint_originals)
        bank.uninstall()
        try:
            student.cleanup()
            student.unpatch_model(unpatch_weights=False)
        except Exception:
            pass
        comfy.model_management.in_training = False

    print(f"audio-fix training complete at step {step}", flush=True)


if __name__ == "__main__":
    main()