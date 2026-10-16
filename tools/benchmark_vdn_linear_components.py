#!/usr/bin/env python3
"""Model-free CUDA timing of VDN's learned linear complement at H3 geometry.

Runs synthetic BF16 activations and random branch weights (no checkpoint, no
H3 model) through the released readout, so it isolates the linear branch from
attention, MLP and model management. Two measurements:

1. ``temporal``: the eager shift-multiply-add temporal short-conv versus the
   single-pass kernel on one [F, S, C] feature volume, with a full-tensor
   bitwise comparison and incremental peak memory.
2. ``components``: CUDA-event time of each readout step (features, statistics,
   alpha gate, scans, output gate, gather, readout matmul, epilogue) and the
   end-to-end ``LinearBranch.readout`` with the fused temporal kernel on and off.

Example (inside the ComfyUI environment)::

    python tools/benchmark_vdn_linear_components.py --preset high --json vdn_linear.json

The numbers are per single block call. They are not sampler wall time.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT.parents[1]):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

PRESETS = {
    "low": {"frames": 52, "grid": (27, 39)},
    "high": {"frames": 62, "grid": (28, 62)},
}


def _timed(fn, rounds, warmup):
    for _ in range(warmup):
        fn()
    if not torch.cuda.is_available():
        samples = []
        for _ in range(rounds):
            started = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - started) * 1e3)
        return {"median_ms": round(statistics.median(samples), 4), "min_ms": round(min(samples), 4)}
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {"median_ms": round(statistics.median(samples), 4), "min_ms": round(min(samples), 4)}


def _peak_increment(fn):
    if not torch.cuda.is_available():
        fn()
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del result
    return round(peak / (1 << 20), 1)


def _weights(channels, heads, head_dim, hidden, rank, device, generator):
    def randn(shape, scale):
        return (torch.randn(shape, generator=generator, device=device) * scale).to(torch.bfloat16)

    return {
        "short_conv.k_sp.weight": randn((channels, 1, 5, 5), 0.05),
        "short_conv.k_tm.weight": randn((channels, 1, 5), 0.2),
        "short_conv.v_sp.weight": randn((channels, 1, 5, 5), 0.05),
        "short_conv.v_tm.weight": randn((channels, 1, 5), 0.2),
        "beta_proj.weight": randn((heads, hidden), 0.02),
        "alpha.down.weight": randn((rank, hidden), 0.02),
        "alpha.up.weight": randn((channels, rank), 0.02),
        "alpha.dt_bias": randn((channels,), 0.05),
        "alpha.A_log": randn((heads,), 0.05),
        "output_gate.down.weight": randn((rank, hidden), 0.02),
        "output_gate.up.weight": randn((channels, rank), 0.02),
        "output_gate.up.bias": randn((channels,), 0.05),
        "norm.weight": torch.ones(head_dim, device=device, dtype=torch.bfloat16),
    }


def run(args):
    from vdn_h3 import branch as B
    from vdn_h3 import temporal_shift as T
    from vdn_h3.window import window_bounds

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    frames, (grid_h, grid_w) = args.frames, args.grid
    per_frame = grid_h * grid_w
    heads, head_dim = args.heads, args.head_dim
    channels = heads * head_dim
    hidden = args.hidden or channels
    rows = frames * per_frame
    report = {
        "contract": "vdn-linear-components-benchmark-v1",
        "end_to_end": False,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch": torch.__version__,
        "geometry": {"frames": frames, "grid": [grid_h, grid_w], "rows": rows, "heads": heads,
                     "head_dim": head_dim, "hidden": hidden},
    }

    feature_volume = torch.randn(frames, per_frame, channels, generator=generator, device=device).to(torch.bfloat16)
    taps = (torch.randn(channels, 5, generator=generator, device=device) * 0.2).to(torch.bfloat16)
    eager = T.eager_temporal_shift(feature_volume, taps, 5)
    fused = T.fused_temporal_shift(feature_volume, taps)
    bitwise = bool(torch.equal(eager.view(torch.int16), fused.view(torch.int16)))
    mismatches = int((eager.view(torch.int16) != fused.view(torch.int16)).sum().item())
    del eager, fused
    eager_time = _timed(lambda: T.eager_temporal_shift(feature_volume, taps, 5), args.rounds, args.warmup)
    fused_time = _timed(lambda: T.fused_temporal_shift(feature_volume, taps), args.rounds, args.warmup)
    report["temporal"] = {
        "bitwise_equal": bitwise,
        "mismatched_elements": mismatches,
        "eager": {**eager_time, "incremental_peak_mib": _peak_increment(
            lambda: T.eager_temporal_shift(feature_volume, taps, 5))},
        "fused": {**fused_time, "incremental_peak_mib": _peak_increment(
            lambda: T.fused_temporal_shift(feature_volume, taps))},
        "volume_mib": round(feature_volume.numel() * 2 / (1 << 20), 1),
    }
    report["temporal"]["speedup"] = round(eager_time["median_ms"] / fused_time["median_ms"], 3)
    feature_volume = None

    weights = _weights(channels, heads, head_dim, hidden, args.rank, device, generator)
    xv = torch.randn(rows, hidden, generator=generator, device=device).to(torch.bfloat16)
    q_raw, k_raw, v_raw = (
        torch.randn(rows, heads, head_dim, generator=generator, device=device).to(torch.bfloat16)
        for _ in range(3)
    )
    grid = (grid_h, grid_w)
    bounds = window_bounds(frames, args.window_radius)
    branch = B.LinearBranch(weights, heads, head_dim, short_conv=("k", "v"), enable_text_state=False)
    shape = (frames, per_frame, heads, head_dim)
    state = {}

    def features_q():
        state["q"] = B._activate(q_raw, l2norm=True)

    def spatial(name, raw):
        def body():
            volume = raw.reshape(frames, grid_h, grid_w, channels).permute(0, 3, 1, 2)
            volume = F.conv2d(volume, weights[f"short_conv.{name}_sp.weight"], padding=2, groups=channels)
            state[f"{name}_sp"] = volume.permute(0, 2, 3, 1).reshape(frames, per_frame, channels)
        return body

    def temporal(name, use_fused):
        def body():
            os.environ[T.ENV_FUSED_TEMPORAL] = "1" if use_fused else "0"
            tm = weights[f"short_conv.{name}_tm.weight"].squeeze(1)
            state[f"{name}_tm"] = T.temporal_shift(state[f"{name}_sp"], tm.to(torch.bfloat16), 5)
        return body

    def activate(name, l2norm):
        def body():
            state[name] = B._activate(state[f"{name}_tm"].reshape(-1, heads, head_dim), l2norm)
        return body

    def statistics_step():
        key_by_frame = state["k"].view(shape).permute(0, 2, 1, 3)
        value_by_frame = state["v"].view(shape).permute(0, 2, 1, 3)
        beta = torch.sigmoid(F.linear(xv, weights["beta_proj.weight"]))
        beta = beta.view(frames, per_frame, heads).permute(0, 2, 1)
        state["a"], state["b"] = B.frame_statistics(key_by_frame, value_by_frame, beta, a_fp32=True)

    def alpha_step():
        mean = xv.view(frames, per_frame, -1).mean(dim=1, dtype=torch.float32)
        state["alpha"] = B.alpha_gate(mean, weights["alpha.down.weight"], weights["alpha.up.weight"],
                                      weights["alpha.dt_bias"], weights["alpha.A_log"], heads, head_dim)

    def scans_step():
        backend = branch._delta_backend(per_frame)
        state["prefix"], state["suffix"] = B.run_scans(backend, state["alpha"], state["a"], state["b"])

    def gate_step():
        state["gate"] = torch.sigmoid(F.linear(xv, weights["output_gate.down.weight"])
                                      @ weights["output_gate.up.weight"].T + weights["output_gate.up.bias"])

    def gather_step():
        state["linear_state"] = B.gather_linear_state(state["prefix"], state["suffix"], state["alpha"], bounds,
                                                      out_dtype=state["gate"].dtype)

    def readout_step():
        query = state["q"].view(shape).permute(0, 2, 1, 3)
        state["readout"] = torch.matmul(query, state["linear_state"].transpose(-1, -2))

    def epilogue_step():
        state["out"] = B.linear_epilogue(state["readout"], weights["norm.weight"], state["gate"],
                                         B.linear_norm_epsilon(weights["norm.weight"].dtype))

    steps = [
        ("q_activation", features_q),
        ("k_spatial_conv", spatial("k", k_raw)),
        ("k_temporal_eager", temporal("k", False)),
        ("k_temporal_fused", temporal("k", True)),
        ("k_activation", activate("k", True)),
        ("v_spatial_conv", spatial("v", v_raw)),
        ("v_temporal_eager", temporal("v", False)),
        ("v_temporal_fused", temporal("v", True)),
        ("v_activation", activate("v", False)),
        ("beta_and_frame_statistics", statistics_step),
        ("alpha_gate", alpha_step),
        ("scans", scans_step),
        ("output_gate", gate_step),
        ("gather_linear_state", gather_step),
        ("readout_matmul", readout_step),
        ("epilogue", epilogue_step),
    ]
    report["components"] = {name: _timed(body, args.rounds, args.warmup) for name, body in steps}

    def end_to_end(use_fused):
        def body():
            os.environ[T.ENV_FUSED_TEMPORAL] = "1" if use_fused else "0"
            return branch.readout(weights, xv, q_raw, k_raw, v_raw, frames, per_frame, bounds, frame_size=grid)
        return body

    eager_out = end_to_end(False)()
    fused_out = end_to_end(True)()
    report["readout"] = {
        "eager_temporal": _timed(end_to_end(False), args.rounds, args.warmup),
        "fused_temporal": _timed(end_to_end(True), args.rounds, args.warmup),
        "outputs_bitwise_equal": bool(torch.equal(eager_out.view(torch.int16), fused_out.view(torch.int16))),
        "eager_incremental_peak_mib": _peak_increment(end_to_end(False)),
        "fused_incremental_peak_mib": _peak_increment(end_to_end(True)),
    }
    report["fused_status"] = T.fused_temporal_status()
    os.environ.pop(T.ENV_FUSED_TEMPORAL, None)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="high")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--grid", type=int, nargs=2, metavar=("H", "W"), help="spatial token grid per frame")
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=0, help="hidden size (default heads*head_dim)")
    parser.add_argument("--rank", type=int, default=128, help="alpha/output-gate low-rank width")
    parser.add_argument("--window-radius", type=int, default=2, help="softmax window radius in frames")
    parser.add_argument("--device", default="cuda", help="cuda, or cpu for a smoke run (no timing meaning)")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    preset = PRESETS[args.preset]
    args.frames = args.frames or preset["frames"]
    args.grid = tuple(args.grid or preset["grid"])
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is required (use --device cpu only for a smoke run)")
    with torch.inference_mode():
        report = run(args)
    text = json.dumps(report, indent=2)
    if args.json is not None:
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    temporal = report["temporal"]
    if not temporal["bitwise_equal"] or not report["readout"]["outputs_bitwise_equal"]:
        raise SystemExit("fused temporal short-conv is not bitwise identical to the eager arithmetic")


if __name__ == "__main__":
    main()
