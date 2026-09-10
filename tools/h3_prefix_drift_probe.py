#!/usr/bin/env python3
"""Trace native-H3 versus finished VDN/Stage-B/Turbo drift by packed row class.

This is a diagnostic, not another attention backend and not a production control.
It uses the installed Comfy MiniMax-H3 INT8/ConvRot graph and the normal VDN wrapper
stack, then compares one tiny inference call against wrapper-free dense H3 at the
same top-level latent/timestep/context input.

The trace is deliberately small.  Native tensors are snapshotted to CPU only for the
synthetic probe geometry, then student tensors are compared and discarded.  It reports
where divergence appears across the exact PackedLayout groups:

    [ packed conditioning prefix | generated target audio | generated target video ]

Within every block it observes:
- block input hidden rows (cumulative drift entering the block),
- raw q/k/v after the finished projection hooks,
- the attention result immediately before out_proj (softmax + VDN gate),
- the attention module output after out_proj and the video-only VDN linear branch,
- MLP output,
- final block output.

For prefix/audio rows under global_gate_mode=video_only, VDN's softmax path is already
dense against every K/V row.  A remaining drift there therefore cannot be explained by
"audio was sparse"; it comes from the finished adapter/hidden trajectory or later block
math.  This distinction is the point of the probe.
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
import comfy.ldm.minimax.model as minimax_model  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402

from vdn_h3.audio_fix_model_options import build_student_transformer_options  # noqa: E402
from vdn_h3.audio_node import _apply_vdn_audio_safe  # noqa: E402
from vdn_h3.drift_diagnostics import (  # noqa: E402
    cpu_snapshot,
    packed_target_ranges,
    pair_metrics,
    split_qkv_rows,
)


VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
AUDIO_STREAMS = 2
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
AUDIO_SCALE = VIDEO_SHIFT / AUDIO_SHIFT
_COMPONENTS = (
    "block_in",
    "q_raw",
    "k_raw",
    "v_raw",
    "attn_pre_out",
    "attn_out",
    "mlp_out",
    "block_out",
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


class _DenseTrace:
    def __init__(self, ranges):
        self.ranges = dict(ranges)
        self.snapshots = {}

    def record(self, block_index, component, tensor):
        if component == "qkv":
            for name, part in zip(("q_raw", "k_raw", "v_raw"), split_qkv_rows(tensor)):
                self.record(block_index, name, part)
            return
        for segment, (start, stop) in self.ranges.items():
            if stop > start:
                self.snapshots[(block_index, component, segment)] = cpu_snapshot(
                    tensor[start:stop]
                )


class _StudentTrace:
    def __init__(self, reference):
        self.reference = reference
        self.ranges = reference.ranges
        self.metrics = {}

    def record(self, block_index, component, tensor):
        if component == "qkv":
            for name, part in zip(("q_raw", "k_raw", "v_raw"), split_qkv_rows(tensor)):
                self.record(block_index, name, part)
            return
        for segment, (start, stop) in self.ranges.items():
            if stop <= start:
                continue
            key = (block_index, component, segment)
            reference = self.reference.snapshots.get(key)
            if reference is None:
                raise RuntimeError(f"dense trace is missing {key}")
            candidate = cpu_snapshot(tensor[start:stop])
            self.metrics[key] = pair_metrics(reference, candidate)


@contextmanager
def _install_trace_hooks(dm, collector):
    """Install hooks after the current role is active, then remove them exactly once."""
    handles = []

    def output_hook(block_index, component):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(
                    f"drift trace expected Tensor from block {block_index} {component}, "
                    f"got {type(output).__name__}"
                )
            collector.record(block_index, component, output)
            return output
        return hook

    def input_hook(block_index, component):
        def hook(_module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError(
                    f"drift trace expected Tensor input at block {block_index} {component}"
                )
            collector.record(block_index, component, inputs[0])
            return None
        return hook

    try:
        for index, block in enumerate(dm.blocks):
            # Register after student.pre_run(): released adapter post-hooks therefore run
            # before our output observers, so qkv/MLP snapshots describe the finished
            # student rather than the underlying base module before its LoRA residual.
            handles.append(block.register_forward_pre_hook(input_hook(index, "block_in")))
            handles.append(block.attn.qkv_proj.register_forward_hook(output_hook(index, "qkv")))
            handles.append(block.attn.out_proj.register_forward_pre_hook(
                input_hook(index, "attn_pre_out")
            ))
            handles.append(block.attn.register_forward_hook(output_hook(index, "attn_out")))
            handles.append(block.mlp.register_forward_hook(output_hook(index, "mlp_out")))
            handles.append(block.register_forward_hook(output_hook(index, "block_out")))
        yield collector
    finally:
        for handle in reversed(handles):
            handle.remove()


def _fmt(metric):
    return f"r={metric.rel_rms:.4g} c={metric.cosine:.6f} m={metric.max_abs:.4g}"


def _print_trace(trace, num_blocks):
    print("\npacked-row drift trace: finished VDN/Stage-B/Turbo vs dense native H3", flush=True)
    print("metrics: r=relative RMS, c=cosine, m=max absolute delta", flush=True)
    print(
        "interpretation: block_in is cumulative drift; q/k/v are raw projections; "
        "attn_pre_out is after QK norm/RoPE + attention + VDN softmax gate but before "
        "out_proj; attn_out includes out_proj and the video-only VDN linear branch.",
        flush=True,
    )
    for index in range(num_blocks):
        for segment in ("prefix", "audio", "video"):
            entries = []
            for component in _COMPONENTS:
                metric = trace.metrics.get((index, component, segment))
                if metric is not None:
                    entries.append(f"{component}({_fmt(metric)})")
            if entries:
                print(f"b{index:02d} {segment:6s}: " + " ".join(entries), flush=True)

    print("\nfirst block crossing relative-RMS thresholds:", flush=True)
    for segment in ("prefix", "audio", "video"):
        for component in ("block_in", "q_raw", "attn_pre_out", "attn_out", "mlp_out", "block_out"):
            values = [
                (index, trace.metrics[(index, component, segment)].rel_rms)
                for index in range(num_blocks)
                if (index, component, segment) in trace.metrics
            ]
            crossings = []
            for threshold in (0.01, 0.05, 0.10):
                hit = next(((i, value) for i, value in values if value >= threshold), None)
                crossings.append("-" if hit is None else f"b{hit[0]:02d}:{hit[1]:.3g}")
            print(
                f"  {segment:6s} {component:12s} >=1%/5%/10%: " + "/".join(crossings),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", default=COMFY_ROOT)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--vdn-checkpoint", required=True)
    parser.add_argument("--stage-b-strength", type=float, default=1.0)
    parser.add_argument("--turbo-strength", type=float, default=0.75)
    parser.add_argument(
        "--global-gate-mode", choices=("checkpoint", "video_only"), default="video_only"
    )
    parser.add_argument("--audio-adapter-strength", type=float, default=1.0)
    parser.add_argument("--conditioning-adapter-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-frames", type=int, default=8)
    parser.add_argument("--audio-frames", type=int, default=32)
    parser.add_argument("--latent-height", type=int, default=8)
    parser.add_argument("--latent-width", type=int, default=8)
    parser.add_argument("--text-tokens", type=int, default=8)
    parser.add_argument(
        "--max-trace-rows",
        type=int,
        default=512,
        help="Fail closed instead of snapshotting a large packed sequence to host memory",
    )
    args = parser.parse_args()

    if not 0.0 <= args.stage_b_strength <= 2.0:
        raise SystemExit("--stage-b-strength must be in [0, 2]")
    if not 0.0 < args.turbo_strength <= 2.0:
        raise SystemExit("--turbo-strength must be in (0, 2]; Turbo-off is not this diagnostic")
    for name in ("audio_adapter_strength", "conditioning_adapter_strength"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.video_frames < 2 or args.audio_frames < 2 or args.text_tokens < 1:
        raise SystemExit("video/audio frames must be >= 2 and text tokens >= 1")
    if (
        args.latent_height < 2
        or args.latent_width < 2
        or args.latent_height % 2
        or args.latent_width % 2
    ):
        raise SystemExit("latent height/width must be positive even values >= 2")

    base_path = _resolve_base(args.base_model)
    base = comfy.sd.load_diffusion_model(base_path, model_options={})
    if base is None:
        raise RuntimeError(f"Comfy could not load {base_path}")
    device = base.load_device
    comfy.model_management.load_models_gpu([base], force_full_load=True)
    base_dm = base.get_model_object("diffusion_model")
    for parameter in base_dm.parameters():
        parameter.requires_grad_(False)

    student = _apply_vdn_audio_safe(
        base,
        args.vdn_checkpoint,
        {"default": args.stage_b_strength, "turbo": args.turbo_strength},
        "stream",
        "grouped",
        True,
        audio_adapter_strength=args.audio_adapter_strength,
        conditioning_adapter_strength=args.conditioning_adapter_strength,
        audio_video_context_strength=1.0,
        conditioning_video_context_strength=1.0,
        audio_fix_strength=0.0,
        apply_turbo_adapter=True,
        retain_buffers="off",
        global_gate_mode=args.global_gate_mode,
        adapter_ablation="none",
        fast_kernels=False,
    )[0]

    sigmas = comfy.samplers.calculate_sigmas(base.model.model_sampling, "simple", 10).to(
        device=device, dtype=torch.float32
    )
    transformer_options = build_student_transformer_options(
        student, sigmas, video_shift=VIDEO_SHIFT, audio_shift=AUDIO_SHIFT
    )
    dense_options = {
        "minimax_h3_sigma_shift_video": VIDEO_SHIFT,
        "minimax_h3_sigma_shift_audio": AUDIO_SHIFT,
        "sample_sigmas": sigmas,
    }

    layout = minimax_model.PackedLayout(
        args.text_tokens,
        args.video_frames,
        args.latent_height,
        args.latent_width,
        args.audio_frames,
    )
    ranges = packed_target_ranges(layout)
    if layout.seq_len > args.max_trace_rows:
        raise RuntimeError(
            f"diagnostic PackedLayout has {layout.seq_len} rows, exceeding "
            f"--max-trace-rows={args.max_trace_rows}; keep this a small structural probe"
        )
    print(f"base: {base_path}", flush=True)
    print(f"layout seq={layout.seq_len}: " + " ".join(
        f"{kind}[{a},{b})" for a, b, kind in layout.segments
    ), flush=True)
    print(
        "grouped ranges: " + " ".join(
            f"{name}[{a},{b})" for name, (a, b) in ranges.items()
        ),
        flush=True,
    )
    print(
        f"student profile: Stage-B={args.stage_b_strength:g} Turbo={args.turbo_strength:g} "
        f"gate={args.global_gate_mode} audio_adapter={args.audio_adapter_strength:g} "
        f"conditioning_adapter={args.conditioning_adapter_strength:g}",
        flush=True,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed + 9187)
    video = torch.randn(
        1, VIDEO_CHANNELS, args.video_frames, args.latent_height, args.latent_width,
        generator=generator, device=device, dtype=torch.float32,
    )
    audio = torch.randn(
        1, AUDIO_CHANNELS, AUDIO_STREAMS, args.audio_frames,
        generator=generator, device=device, dtype=torch.float32,
    )
    # Hidden-width context deliberately skips token-refiner.  The trace therefore
    # attributes the 50 packed DiT blocks; token-refiner adapter effects are catalogued
    # separately in the audit and exercised by real prompt-cache/trainer smoke tests.
    context = torch.randn(
        1, args.text_tokens, int(base_dm.hidden_size),
        generator=generator, device=device, dtype=torch.bfloat16,
    )
    sigma = float(sigmas[min(4, sigmas.numel() - 2)])

    dense_trace = _DenseTrace(ranges)
    with torch.no_grad(), _install_trace_hooks(base_dm, dense_trace):
        dense_out = _direct_h3_output(
            base_dm, video, audio, sigma, context, dense_options
        )

    student_trace = _StudentTrace(dense_trace)
    with torch.no_grad(), _student_role(student, device) as student_dm:
        # Hooks are installed only after pre_run so they observe the released adapter
        # post-forward hooks' final qkv/MLP outputs.
        with _install_trace_hooks(student_dm, student_trace):
            student_out = _direct_h3_output(
                student_dm, video, audio, sigma, context, transformer_options
            )

    _print_trace(student_trace, len(base_dm.blocks))
    final_video = pair_metrics(dense_out[0].cpu(), student_out[0].cpu())
    final_audio = pair_metrics(dense_out[1].cpu(), student_out[1].cpu())
    print("\nfinal native-vs-student output:", flush=True)
    print(f"  video {_fmt(final_video)}", flush=True)
    print(f"  audio {_fmt(final_audio)}", flush=True)
    print(
        "DRIFT PROBE COMPLETE. Tensor divergence is diagnostic evidence only; decoded "
        "media remains the acceptance test for yapping, dialogue and whisper behavior.",
        flush=True,
    )


if __name__ == "__main__":
    main()
