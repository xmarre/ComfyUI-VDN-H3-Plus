#!/usr/bin/env python3
"""Low-host-RAM GPU entrypoint for the direct INT8/ConvRot audio-fix trainer.

This is the authoritative workstation path for the canonical released VDN/Turbo
training profile:

- Stage-B 1.0;
- Turbo 1.0;
- checkpoint global gate;
- 8-step ``res_multistep`` / Comfy ``simple`` sigma grid;
- 52 video latent frames and 292 stereo audio latent frames.

The released Turbo adapter is an 8-step adapter.  A 10-step deployment run at reduced
Turbo strength is a separate transfer/quality operating point and must not be used as
the canonical Turbo-1.0 training rollout.

Loading/runtime policy is specialized for a workstation with substantially more VRAM
than host RAM:

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

import argparse
import importlib.util
import json
import os
import random
import sys
import time


CANONICAL_SAMPLER_STEPS = 8


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


def _build_parser():
    """Build the CLI without importing Comfy/VDN runtime modules.

    ``--help`` must work on CPU-only hosts and before the supplied Comfy checkout has
    been added to ``sys.path``.  Runtime imports therefore happen only after argument
    parsing in :func:`main`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comfy-root",
        default=os.environ.get("COMFYUI_ROOT", "/home/toor/ComfyUI"),
    )
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
    parser.add_argument("--video-latent-frames", type=int, default=52)
    parser.add_argument("--audio-latent-frames", type=int, default=292)
    parser.add_argument("--train-steps", type=int, default=250)
    parser.add_argument("--sampler-steps", type=int, default=CANONICAL_SAMPLER_STEPS)
    parser.add_argument("--stage-b-strength", type=float, default=1.0)
    parser.add_argument("--turbo-strength", type=float, default=1.0)
    parser.add_argument(
        "--global-gate-mode",
        choices=("checkpoint", "video_only"),
        default="checkpoint",
    )
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-preserve-weight", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Force one optimizer step and save step 0/1")
    return parser


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


def _save_adapter_8(impl, output_root, step, bank, base_path, vdn_checkpoint,
                    latent_shape, audio_t, *, stage_b_strength, turbo_strength,
                    global_gate_mode):
    path = os.path.join(output_root, f"audio_fix_step_{step:06d}")
    if os.path.exists(path):
        impl.shutil.rmtree(path)
    bank.save_adapter(
        path,
        step=step,
        metadata={
            "base_model": os.path.realpath(base_path),
            "vdn_checkpoint": vdn_checkpoint,
            "sampler": "res_multistep",
            "sigma_schedule": "simple",
            "sampler_steps": CANONICAL_SAMPLER_STEPS,
            "video_shift": impl.VIDEO_SHIFT,
            "audio_shift": impl.AUDIO_SHIFT,
            "stage_b_strength": float(stage_b_strength),
            "turbo_strength": float(turbo_strength),
            "global_gate_mode": global_gate_mode,
            "audio_adapter_strength": 1.0,
            "conditioning_adapter_strength": 1.0,
            "audio_video_context_strength": 1.0,
            "conditioning_video_context_strength": 1.0,
            "rollout_profile": impl.ROLLOUT_PROFILE,
            "spectrum_forecasting_emulated": False,
            "progressive_handoff_emulated": False,
            "video_latent_shape": list(latent_shape),
            "audio_latent_frames": int(audio_t),
            "audio_streams": impl.AUDIO_STREAMS,
            "teacher": "same production INT8/ConvRot H3 with VDN/adapters disabled",
        },
    )
    return path


def main():
    # Parse first.  This lets argparse service --help without importing Comfy's CUDA
    # runtime on CPU-only CI hosts and gives standalone tools a root before vdn_h3 is
    # imported for the first time.
    args = _build_parser().parse_args()
    os.environ["COMFYUI_ROOT"] = os.path.abspath(os.path.expanduser(args.comfy_root))

    impl = _load_impl()
    from vdn_h3.audio_fix_progress import (
        TrainingProgress,
        install_scope_safe_block_checkpointing,
    )
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

    if args.sampler_steps != CANONICAL_SAMPLER_STEPS:
        raise SystemExit(
            "Canonical Turbo-1.0 audio-fix training requires --sampler-steps 8; "
            "10-step reduced-Turbo deployment runs are a separate transfer profile")
    if args.video_latent_frames != 52 or args.audio_latent_frames != 292:
        raise SystemExit(
            "Production 7-second contract requires --video-latent-frames 52 and "
            "--audio-latent-frames 292")
    if args.latent_height < 2 or args.latent_width < 2:
        raise SystemExit("latent height/width must be >= 2")
    if args.latent_height % 2 or args.latent_width % 2:
        raise SystemExit("MiniMax H3 latent height/width must be even")
    if not 0.0 <= args.stage_b_strength <= 2.0:
        raise SystemExit("--stage-b-strength must be in [0, 2]")
    if not 0.0 < args.turbo_strength <= 2.0:
        raise SystemExit("--turbo-strength must be in (0, 2]; Turbo-off is not this training path")
    if args.save_every < 1:
        raise SystemExit("--save-every must be >= 1")

    impl.comfy.cli_args.args.disable_comfy_compiler = True
    base_path = impl._resolve_base(args.base_model)
    prompt_files = impl._prompt_files(args.prompt_cache_dir)
    output_root = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_root, exist_ok=True)
    state_path = os.path.join(output_root, "train_state.pt")
    metrics_path = os.path.join(output_root, "metrics.jsonl")

    max_steps = 1 if args.smoke else int(args.train_steps)
    impl.torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("[audio-fix setup 1/5] loading production H3 directly into CUDA", flush=True)
    base = load_diffusion_model_direct_gpu(base_path, model_options={})
    if base is None:
        raise RuntimeError(f"Comfy could not load {base_path}")
    device = base.load_device
    impl.comfy.model_management.load_models_gpu([base], force_full_load=True)
    base_dm = base.get_model_object("diffusion_model")

    print("[audio-fix setup 2/5] validating INT8/ConvRot training targets", flush=True)
    impl.validate_production_quant_targets(base_dm)
    for parameter in base_dm.parameters():
        parameter.requires_grad_(False)

    print("[audio-fix setup 3/5] attaching VDN Stage-B/Turbo student", flush=True)
    student = apply_vdn_gpu_low_host_ram(
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
        retain_buffers="on",
        global_gate_mode=args.global_gate_mode,
        adapter_ablation="none",
        fast_kernels=False,
    )[0]

    print("[audio-fix setup 4/5] installing trainable audio-fix bank", flush=True)
    bank = impl.TrainableAudioFixBank(base_dm, rank=args.rank, alpha=args.alpha).to(device)
    bank.install()
    optimizer = impl._optimizer(bank)
    if hasattr(optimizer, "train"):
        optimizer.train()
    generator = impl.torch.Generator(device=device).manual_seed(args.seed + 1009)

    step = prompt_cursor = 0
    if not args.no_resume and not args.smoke:
        step, prompt_cursor = impl._resume(state_path, bank, optimizer, generator)
        if step:
            print(f"resumed step {step} from {state_path}", flush=True)

    print("[audio-fix setup 5/5] building canonical 8-step RES/simple schedule", flush=True)
    sigmas = impl.comfy.samplers.calculate_sigmas(
        base.model.model_sampling, "simple", CANONICAL_SAMPLER_STEPS).to(
            device=device, dtype=impl.torch.float32)
    if sigmas.numel() != CANONICAL_SAMPLER_STEPS + 1 or float(sigmas[-1]) != 0.0:
        raise RuntimeError(f"Unexpected canonical 8-step simple sigma table: {sigmas.tolist()}")

    student_transformer_options = impl.build_student_transformer_options(
        student,
        sigmas,
        video_shift=impl.VIDEO_SHIFT,
        audio_shift=impl.AUDIO_SHIFT,
    )

    latent_shape = (
        1, impl.VIDEO_CHANNELS, args.video_latent_frames,
        args.latent_height, args.latent_width,
    )
    early = {0, 1, 2, 4, 8, 16, 32}
    profile = {
        "stage_b_strength": args.stage_b_strength,
        "turbo_strength": args.turbo_strength,
        "global_gate_mode": args.global_gate_mode,
        "sampler": "res_multistep",
        "sampler_steps": CANONICAL_SAMPLER_STEPS,
        "audio_streams": impl.AUDIO_STREAMS,
        "rollout_profile": impl.ROLLOUT_PROFILE,
    }
    print("audio-fix training profile: " + json.dumps(profile, sort_keys=True), flush=True)

    progress = TrainingProgress(
        num_blocks=len(base_dm.blocks),
        total_steps=max_steps,
        initial_step=step,
    )
    checkpoint_originals = install_scope_safe_block_checkpointing(
        base_dm, bank, progress=progress
    )

    try:
        with impl.comfy_quant_training_mode():
            if step == 0 and (args.smoke or 0 in early):
                if hasattr(optimizer, "eval"):
                    optimizer.eval()
                saved = _save_adapter_8(
                    impl, output_root, 0, bank, base_path, args.vdn_checkpoint,
                    latent_shape, args.audio_latent_frames,
                    stage_b_strength=args.stage_b_strength,
                    turbo_strength=args.turbo_strength,
                    global_gate_mode=args.global_gate_mode,
                )
                progress.write(f"adapter checkpoint -> {saved}")
                if hasattr(optimizer, "train"):
                    optimizer.train()

            while step < max_steps:
                prompt_path = prompt_files[prompt_cursor % len(prompt_files)]
                prompt_cursor += 1
                context, tags, prompt = impl._load_prompt(prompt_path, device)
                audio_span = impl.generated_audio_span(
                    context.shape[1], args.video_latent_frames,
                    args.latent_height, args.latent_width, args.audio_latent_frames)
                train_index = random.Random(
                    args.seed + step * 1000003).randrange(CANONICAL_SAMPLER_STEPS)
                video = impl.torch.randn(
                    latent_shape, generator=generator, device=device,
                    dtype=impl.torch.float32)
                audio = impl.torch.randn(
                    1, impl.AUDIO_CHANNELS, impl.AUDIO_STREAMS,
                    args.audio_latent_frames,
                    generator=generator, device=device, dtype=impl.torch.float32)

                progress.start_step(step=step + 1, train_index=train_index)
                started = time.time()

                progress.phase(f"rollout 0→{train_index} ({train_index} H3 evals)")
                video, audio = impl._rollout_student(
                    student, bank, video, audio, train_index, context, tags, sigmas,
                    audio_span, device, student_transformer_options)

                progress.phase("dense teacher target")
                teacher_v, teacher_a = impl._teacher_target(
                    base_dm, bank, video, audio, train_index, context, tags, sigmas)

                progress.phase("frozen VDN/Turbo target")
                frozen_v, _frozen_a = impl._frozen_student_target(
                    student, bank, video, audio, train_index, context, tags, sigmas,
                    device, student_transformer_options)

                optimizer.zero_grad(set_to_none=True)
                progress.phase("train forward")
                with impl._student_role(student, device), bank.scope(
                        *audio_span, enabled=True, checkpoint_blocks=True):
                    dm = student.get_model_object("diffusion_model")
                    out_v, out_a = impl._model_output(
                        dm, video, audio, sigmas[train_index], context, tags, sigmas,
                        transformer_options=student_transformer_options)
                    student_v, student_a = impl._x0(
                        video, audio, out_v, out_a, sigmas[train_index])
                    audio_loss = impl.F.mse_loss(
                        student_a / impl.AUDIO_SCALE, teacher_a / impl.AUDIO_SCALE)
                    video_loss = impl.F.mse_loss(student_v, frozen_v)
                    loss = audio_loss + float(args.video_preserve_weight) * video_loss
                    if not impl.torch.isfinite(loss):
                        raise FloatingPointError(
                            f"non-finite loss at step {step + 1}, grid {train_index}")
                    progress.phase("backward / checkpoint recompute")
                    loss.backward()

                progress.phase("gradient validation + optimizer")
                grad_sq = 0.0
                nonzero = 0
                nonzero_a = 0
                nonzero_b = 0
                for pair in bank.pairs:
                    for side, parameter in (("A", pair.lora_A), ("B", pair.lora_B)):
                        if parameter.grad is None:
                            continue
                        if not impl.torch.isfinite(parameter.grad).all():
                            raise FloatingPointError("non-finite audio-fix gradient")
                        maximum = float(parameter.grad.detach().abs().max())
                        if maximum > 0:
                            nonzero += 1
                            if side == "A":
                                nonzero_a += 1
                            else:
                                nonzero_b += 1
                        grad_sq += float(parameter.grad.detach().float().pow(2).sum())
                if nonzero == 0:
                    raise RuntimeError(
                        "audio_fix received no nonzero gradients through the full INT8/VDN graph")
                optimizer.step()
                step += 1
                elapsed = time.time() - started

                # prodigy-plus-schedule-free is pinned to 2.0.1. With d_limiter=True,
                # this adaptive scale is the quantity that tells us whether a short
                # diagnostic run has actually escaped the deliberately tiny d0 regime.
                optimizer_group = optimizer.param_groups[0]
                prodigy_d = float(optimizer_group["d"])
                prodigy_d_prev = float(optimizer_group["d_prev"])

                peak_gib = impl.torch.cuda.max_memory_allocated(device) / (1 << 30)
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
                    "nonzero_lora_a_grad_tensors": nonzero_a,
                    "nonzero_lora_b_grad_tensors": nonzero_b,
                    "prodigy_d": prodigy_d,
                    "prodigy_d_prev": prodigy_d_prev,
                    "seconds": elapsed,
                    "peak_gib": peak_gib,
                    **profile,
                }
                progress.write(json.dumps(row, ensure_ascii=False))
                with open(metrics_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

                should_save = (
                    args.smoke or step in early or step % args.save_every == 0
                    or step == max_steps
                )
                if should_save:
                    progress.phase("saving checkpoint")
                    if hasattr(optimizer, "eval"):
                        optimizer.eval()
                    saved = _save_adapter_8(
                        impl, output_root, step, bank, base_path,
                        args.vdn_checkpoint, latent_shape, args.audio_latent_frames,
                        stage_b_strength=args.stage_b_strength,
                        turbo_strength=args.turbo_strength,
                        global_gate_mode=args.global_gate_mode,
                    )
                    progress.write(f"adapter checkpoint -> {saved}")
                    if hasattr(optimizer, "train"):
                        optimizer.train()
                    impl._save_state(
                        state_path, step, bank, optimizer, generator, prompt_cursor)

                progress.finish_step(
                    loss=float(row["loss"]), peak_gib=peak_gib, seconds=elapsed
                )

                del context, tags, video, audio, teacher_v, teacher_a, frozen_v
                del out_v, out_a, student_v, student_a, loss, audio_loss, video_loss
                impl.torch.cuda.empty_cache()
    finally:
        progress.close()
        impl.restore_block_checkpointing(checkpoint_originals)
        bank.uninstall()
        try:
            student.cleanup()
            student.unpatch_model(unpatch_weights=False)
        except Exception:
            pass
        impl.comfy.model_management.in_training = False

    print(f"audio-fix training complete at step {step}", flush=True)


if __name__ == "__main__":
    main()
