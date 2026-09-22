"""Variable-grid extension of VDN's learned linear complement.

The released linear branch assumes one fixed spatial token grid for every video
frame. Partitioned exact-prefix continuation deliberately violates that assumption:
protected prefix frames stay on the target grid while generated suffix frames stay
on the source grid.

This module extends only the released ``vdn_solve`` inference path. Per-frame
statistics and recurrence remain the released VDN arithmetic. Spatial short-conv
runs on each frame's native grid. The temporal depthwise short-conv uses the same
kernel and zero-padding as released VDN; only taps crossing a grid boundary are
mapped to the destination frame on MiniMax-H3's area-normalized spatial
coordinate lattice with FP32 bilinear sampling. When all frame grids are identical, no interpolation occurs
and the helper reduces to the released fixed-grid computation.

Target-prefix frame statistics are weighted by source_rows / target_rows so the
linear state uses the same physical per-frame carrier measure as the partitioned
softmax contract instead of overweighting the denser prefix representation.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
import math
import time
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from . import branch as B
from .retained import run_scans_runtime


def _frame_offsets(frame_sizes: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    offsets = []
    cursor = 0
    for grid_h, grid_w in frame_sizes:
        if type(grid_h) is not int or type(grid_w) is not int or grid_h <= 0 or grid_w <= 0:
            raise RuntimeError("partitioned VDN linear frame grids must use positive integers")
        rows = grid_h * grid_w
        offsets.append((cursor, cursor + rows))
        cursor += rows
    if not offsets:
        raise RuntimeError("partitioned VDN linear requires at least one video frame")
    return tuple(offsets)


def _validate_inputs(
    branch,
    x_video: torch.Tensor,
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    frame_sizes: Sequence[tuple[int, int]],
    bounds: Sequence[tuple[int, int]],
    measure_scales: Sequence[float],
) -> tuple[tuple[int, int], ...]:
    if getattr(branch, "delta_rule", None) != "vdn_solve":
        raise RuntimeError("partitioned VDN linear currently supports the released vdn_solve rule only")
    if len(frame_sizes) != len(bounds) or len(frame_sizes) != len(measure_scales):
        raise RuntimeError("partitioned VDN linear frame geometry/bounds/measure lengths differ")
    offsets = _frame_offsets(frame_sizes)
    rows = offsets[-1][1]
    heads = int(branch.num_heads)
    head_dim = int(branch.head_dim)
    if x_video.ndim != 2 or int(x_video.shape[0]) != rows:
        raise RuntimeError("partitioned VDN linear hidden rows do not match frame geometry")
    expected = (rows, heads, head_dim)
    if any(tuple(tensor.shape) != expected for tensor in (q_raw, k_raw, v_raw)):
        raise RuntimeError("partitioned VDN linear raw QKV rows do not match frame geometry")
    frame_count = len(frame_sizes)
    for index, ((lo, hi), scale) in enumerate(zip(bounds, measure_scales, strict=True)):
        if (
            type(lo) is not int
            or type(hi) is not int
            or lo > hi
            or lo > frame_count - 1
            or hi < 0
        ):
            raise RuntimeError(f"partitioned VDN linear bound {index} is invalid")
        scale = float(scale)
        if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
            raise RuntimeError("partitioned VDN linear physical measure must be finite inside (0, 1]")
    return offsets


def _spatial_conv_frame(tokens: torch.Tensor, weight: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    grid_h, grid_w = grid
    rows, heads, head_dim = tokens.shape
    if rows != grid_h * grid_w:
        raise RuntimeError("partitioned VDN spatial short-conv frame rows do not match its grid")
    channels = heads * head_dim
    volume = tokens.reshape(grid_h, grid_w, channels).permute(2, 0, 1).unsqueeze(0)
    return F.conv2d(volume, weight, padding=2, groups=channels)


def _h3_axis_geometry(grid_h: int, grid_w: int, axis: int) -> tuple[int, float, float]:
    """Return (length, start, step) for ComfyUI MiniMax-H3 spatial RoPE."""
    if grid_h <= 0 or grid_w <= 0 or axis not in (0, 1):
        raise RuntimeError("partitioned VDN physical grid requires positive H/W and axis 0/1")
    dim = grid_h if axis == 0 else grid_w
    sqrt_area = math.sqrt(float(grid_h * grid_w))
    ratio = float(dim) / sqrt_area
    return dim, (1.0 - ratio) * 16.0, 32.0 / sqrt_area


def _h3_axis_coordinates(
    grid_h: int,
    grid_w: int,
    axis: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return ComfyUI MiniMax-H3 spatial RoPE coordinates for one patch-grid axis."""
    length, start, step = _h3_axis_geometry(grid_h, grid_w, axis)
    return torch.arange(length, device=device, dtype=torch.float32) * step + start


def _physical_resample_grid(
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Build grid_sample coordinates that preserve H3's physical RoPE lattice."""
    source_h, source_w = map(int, source_hw)
    target_h, target_w = map(int, target_hw)
    source_y_len, source_y_start, source_step = _h3_axis_geometry(source_h, source_w, 0)
    source_x_len, source_x_start, source_step_x = _h3_axis_geometry(source_h, source_w, 1)
    target_y_len, target_y_start, target_step = _h3_axis_geometry(target_h, target_w, 0)
    target_x_len, target_x_start, target_step_x = _h3_axis_geometry(target_h, target_w, 1)

    def normalized(
        target_len: int,
        target_start: float,
        target_stride: float,
        source_len: int,
        source_start: float,
        source_stride: float,
    ) -> torch.Tensor:
        if source_len == 1:
            return torch.zeros(target_len, device=device, dtype=torch.float32)
        target_index = torch.arange(target_len, device=device, dtype=torch.float32)
        source_index = (target_start + target_index * target_stride - source_start) / source_stride
        return 2.0 * source_index / float(source_len - 1) - 1.0

    grid_y = normalized(
        target_y_len,
        target_y_start,
        target_step,
        source_y_len,
        source_y_start,
        source_step,
    )
    grid_x = normalized(
        target_x_len,
        target_x_start,
        target_step_x,
        source_x_len,
        source_x_start,
        source_step_x,
    )
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0)


def _map_temporal_neighbor(
    source: torch.Tensor,
    target_hw: tuple[int, int],
    *,
    grid_cache: dict[tuple[tuple[int, int], tuple[int, int], str], torch.Tensor] | None = None,
) -> torch.Tensor:
    if tuple(source.shape[-2:]) == tuple(target_hw):
        return source
    if source.ndim != 4 or int(source.shape[0]) != 1:
        raise RuntimeError("partitioned VDN temporal map expects 1xCxHxW features")
    source_hw = tuple(map(int, source.shape[-2:]))
    target_hw = tuple(map(int, target_hw))
    if min(*source_hw, *target_hw) <= 0:
        raise RuntimeError("partitioned VDN temporal grids must be positive")

    # H3 RoPE does not place different resolutions on PyTorch interpolate's
    # implicit half-pixel lattice. _frame_grid/_axis_from_sqrt_area use a
    # distinct area-normalized, endpoint-excluded physical lattice. Mapping
    # cross-grid temporal taps with F.interpolate(..., align_corners=False)
    # therefore introduces a systematic spatial phase offset at the boundary.
    cache_key = (source_hw, target_hw, str(source.device))
    grid = grid_cache.get(cache_key) if grid_cache is not None else None
    if grid is None:
        grid = _physical_resample_grid(source_hw, target_hw, source.device)
        if grid_cache is not None:
            grid_cache[cache_key] = grid

    # Only cross-grid taps use this path. FP32 interpolation preserves the
    # existing precision contract; border padding handles the small endpoint
    # extent mismatch caused by endpoint-excluded H3 lattices without injecting
    # zero-valued feature bands.
    return F.grid_sample(
        source.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).to(source.dtype)


def _heterogeneous_conv_features(
    tokens: torch.Tensor,
    spatial_weight: torch.Tensor,
    temporal_weight: torch.Tensor,
    frame_sizes: Sequence[tuple[int, int]],
    offsets: Sequence[tuple[int, int]],
    *,
    l2norm: bool,
    grid_cache: dict[tuple[tuple[int, int], tuple[int, int], str], torch.Tensor] | None = None,
    suppress_cross_grid_temporal_taps: bool = False,
    diagnostic_stats: dict[str, int] | None = None,
) -> torch.Tensor:
    maps = [
        _spatial_conv_frame(tokens[start:stop], spatial_weight, grid)
        for (start, stop), grid in zip(offsets, frame_sizes, strict=True)
    ]
    temporal = temporal_weight.squeeze(1)
    if temporal.ndim != 2 or temporal.shape[1] <= 0 or temporal.shape[1] % 2 == 0:
        raise RuntimeError("partitioned VDN temporal short-conv requires an odd depthwise kernel")
    kernel = int(temporal.shape[1])
    pad = kernel // 2
    outputs = []
    for frame, grid in enumerate(frame_sizes):
        target_h, target_w = grid
        mixed = None
        for tap in range(kernel):
            source_frame = frame + tap - pad
            if source_frame < 0 or source_frame >= len(maps):
                continue
            if (
                suppress_cross_grid_temporal_taps
                and tuple(frame_sizes[source_frame]) != tuple(grid)
            ):
                if diagnostic_stats is not None:
                    diagnostic_stats["suppressed_taps"] = diagnostic_stats.get("suppressed_taps", 0) + 1
                    diagnostic_stats["suppressed_rows"] = (
                        diagnostic_stats.get("suppressed_rows", 0) + target_h * target_w
                    )
                continue
            source = _map_temporal_neighbor(
                maps[source_frame],
                (target_h, target_w),
                grid_cache=grid_cache,
            )
            part = source * temporal[:, tap].to(source.dtype).view(1, -1, 1, 1)
            mixed = part if mixed is None else mixed + part
        if mixed is None:  # pragma: no cover - an odd kernel always includes the current frame
            mixed = maps[frame].new_zeros(maps[frame].shape)
        rows = target_h * target_w
        heads = int(tokens.shape[1])
        head_dim = int(tokens.shape[2])
        frame_tokens = mixed[0].permute(1, 2, 0).reshape(rows, heads, head_dim)
        outputs.append(B._activate(frame_tokens, l2norm=l2norm))
    return torch.cat(outputs, dim=0)


def _variable_features(
    branch,
    weights,
    q_raw,
    k_raw,
    v_raw,
    frame_sizes,
    offsets,
    *,
    suppress_cross_grid_temporal_taps: bool = False,
    diagnostic_stats: dict[str, int] | None = None,
):
    conv = tuple(getattr(branch, "short_conv", ()) or ())
    grid_cache: dict[tuple[tuple[int, int], tuple[int, int], str], torch.Tensor] = {}

    def feature(name: str, raw: torch.Tensor, *, l2norm: bool):
        if name not in conv:
            return B._activate(raw, l2norm=l2norm)
        return _heterogeneous_conv_features(
            raw,
            weights[f"short_conv.{name}_sp.weight"],
            weights[f"short_conv.{name}_tm.weight"],
            frame_sizes,
            offsets,
            l2norm=l2norm,
            grid_cache=grid_cache,
            suppress_cross_grid_temporal_taps=suppress_cross_grid_temporal_taps,
            diagnostic_stats=diagnostic_stats,
        )

    return (
        feature("q", q_raw, l2norm=True),
        feature("k", k_raw, l2norm=True),
        feature("v", v_raw, l2norm=False),
    )


def _core_readout(
    branch,
    weights,
    x_video,
    q_raw,
    k_raw,
    v_raw,
    frame_sizes,
    bounds,
    measure_scales,
    *,
    text_x=None,
    text_k_raw=None,
    text_v_raw=None,
    suppress_cross_grid_temporal_taps: bool = False,
    diagnostic_stats: dict[str, int] | None = None,
    record_component=None,
    cuda_span=None,
):
    offsets = _validate_inputs(
        branch,
        x_video,
        q_raw,
        k_raw,
        v_raw,
        frame_sizes,
        bounds,
        measure_scales,
    )
    heads = int(branch.num_heads)
    head_dim = int(branch.head_dim)
    def component_span(name):
        return nullcontext() if cuda_span is None else cuda_span(name)

    features_started = time.perf_counter()
    with component_span("vdn_linear_features"):
        query, key, value = _variable_features(
            branch,
            weights,
            q_raw,
            k_raw,
            v_raw,
            frame_sizes,
            offsets,
            suppress_cross_grid_temporal_taps=suppress_cross_grid_temporal_taps,
            diagnostic_stats=diagnostic_stats,
        )
    if record_component is not None:
        record_component(
            "vdn_linear_features_host_wall_s",
            time.perf_counter() - features_started,
        )
    statistics_started = time.perf_counter()
    with component_span("vdn_linear_statistics"):
        beta_rows = torch.sigmoid(F.linear(x_video, weights["beta_proj.weight"]))
        if tuple(beta_rows.shape) != (x_video.shape[0], heads):
            raise RuntimeError("partitioned VDN beta projection has unexpected geometry")

        a_frames = []
        b_frames = []
        means = []
        for (start, stop), measure in zip(offsets, measure_scales, strict=True):
            key_frame = key[start:stop].permute(1, 0, 2).unsqueeze(0)
            value_frame = value[start:stop].permute(1, 0, 2).unsqueeze(0)
            beta_frame = beta_rows[start:stop].transpose(0, 1).unsqueeze(0)
            beta_frame = beta_frame * float(measure)
            frame_a, frame_b = B.frame_statistics(
                key_frame,
                value_frame,
                beta_frame,
                a_fp32=branch.a_fp32,
            )
            a_frames.append(frame_a)
            b_frames.append(frame_b)
            means.append(x_video[start:stop].mean(dim=0, dtype=torch.float32))
        a_raw = torch.cat(a_frames, dim=0)
        b_raw = torch.cat(b_frames, dim=0)
        frame_mean = torch.stack(means, dim=0)
        alpha = B.alpha_gate(
            frame_mean,
            weights["alpha.down.weight"],
            weights["alpha.up.weight"],
            weights["alpha.dt_bias"],
            weights["alpha.A_log"],
            heads,
            head_dim,
        )
    if record_component is not None:
        record_component(
            "vdn_linear_statistics_host_wall_s",
            time.perf_counter() - statistics_started,
        )

    scans_started = time.perf_counter()
    with component_span("vdn_linear_scans"):
        text_state = branch._text_state(weights, text_x, text_k_raw, text_v_raw)
        # vdn_solve does not depend on tokens_per_frame. Use the smallest physical
        # frame size as a stable cache identity while preserving released arithmetic.
        backend = branch._delta_backend(min(stop - start for start, stop in offsets))
        prefix_states, suffix_states = run_scans_runtime(
            backend,
            alpha,
            a_raw,
            b_raw,
            text_state=text_state,
        )
    if record_component is not None:
        record_component(
            "vdn_linear_scans_host_wall_s",
            time.perf_counter() - scans_started,
        )
    gate_started = time.perf_counter()
    with component_span("vdn_linear_gate"):
        gate = torch.sigmoid(
            F.linear(x_video, weights["output_gate.down.weight"])
            @ weights["output_gate.up.weight"].T
            + weights["output_gate.up.bias"]
        )
    if record_component is not None:
        record_component(
            "vdn_linear_gate_host_wall_s",
            time.perf_counter() - gate_started,
        )
    gather_started = time.perf_counter()
    with component_span("vdn_linear_gather"):
        linear_state = B.gather_linear_state(
            prefix_states,
            suffix_states,
            alpha,
            bounds,
            bridge=branch.bridge,
            text_state=text_state,
            out_dtype=gate.dtype,
            fuse=False,
        )
    if record_component is not None:
        record_component(
            "vdn_linear_gather_host_wall_s",
            time.perf_counter() - gather_started,
        )

    outputs = []
    epsilon_started = time.perf_counter()
    with component_span("vdn_linear_epsilon_scalar"):
        eps = weights["norm.weight"].new_tensor(1e-6).item()
    if record_component is not None:
        record_component(
            "vdn_linear_epsilon_scalar_host_wall_s",
            time.perf_counter() - epsilon_started,
        )
    output_started = time.perf_counter()
    with component_span("vdn_linear_output"):
        for frame, (start, stop) in enumerate(offsets):
            query_frame = query[start:stop].permute(1, 0, 2)
            readout = torch.matmul(query_frame, linear_state[frame].transpose(-1, -2)).unsqueeze(0)
            outputs.append(
                B.linear_epilogue(
                    readout,
                    weights["norm.weight"],
                    gate[start:stop],
                    eps,
                    fuse=False,
                )
            )
        result = torch.cat(outputs, dim=0)
    if record_component is not None:
        record_component(
            "vdn_linear_output_host_wall_s",
            time.perf_counter() - output_started,
        )
    return result


def partitioned_linear_readout(
    branch: Any,
    weights: dict[str, torch.Tensor],
    x_video: torch.Tensor,
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    *,
    frame_sizes: Sequence[tuple[int, int]],
    bounds: Sequence[tuple[int, int]],
    measure_scales: Sequence[float],
    text_x: torch.Tensor | None = None,
    text_k_raw: torch.Tensor | None = None,
    text_v_raw: torch.Tensor | None = None,
    skip_ends: bool = False,
    suppress_cross_grid_temporal_taps: bool = False,
    diagnostic_stats: dict[str, int] | None = None,
    record_component=None,
    cuda_span=None,
) -> torch.Tensor:
    """Evaluate VDN's learned linear complement over heterogeneous frame grids."""
    total_started = time.perf_counter()
    local = copy.copy(branch)
    local._backend = None
    local._backend_key = None
    offsets = _validate_inputs(
        local,
        x_video,
        q_raw,
        k_raw,
        v_raw,
        frame_sizes,
        bounds,
        measure_scales,
    )
    heads = int(local.num_heads)
    head_dim = int(local.head_dim)

    if skip_ends:
        if len(frame_sizes) <= 2:
            result = x_video.new_zeros((x_video.shape[0], heads * head_dim))
            if record_component is not None:
                record_component(
                    "vdn_linear_api_host_wall_s",
                    time.perf_counter() - total_started,
                )
            return result
        first_stop = offsets[0][1]
        last_start = offsets[-1][0]
        inner_bounds = tuple((lo - 1, hi - 1) for lo, hi in bounds[1:-1])
        inner = _core_readout(
            local,
            weights,
            x_video[first_stop:last_start],
            q_raw[first_stop:last_start],
            k_raw[first_stop:last_start],
            v_raw[first_stop:last_start],
            tuple(frame_sizes[1:-1]),
            inner_bounds,
            tuple(measure_scales[1:-1]),
            text_x=text_x,
            text_k_raw=text_k_raw,
            text_v_raw=text_v_raw,
            suppress_cross_grid_temporal_taps=suppress_cross_grid_temporal_taps,
            diagnostic_stats=diagnostic_stats,
            record_component=record_component,
            cuda_span=cuda_span,
        )
        out = inner.new_zeros((x_video.shape[0], inner.shape[-1]))
        out[first_stop:last_start] = inner
        if record_component is not None:
            record_component(
                "vdn_linear_api_host_wall_s",
                time.perf_counter() - total_started,
            )
        return out

    result = _core_readout(
        local,
        weights,
        x_video,
        q_raw,
        k_raw,
        v_raw,
        tuple(frame_sizes),
        tuple(bounds),
        tuple(measure_scales),
        text_x=text_x,
        text_k_raw=text_k_raw,
        text_v_raw=text_v_raw,
        suppress_cross_grid_temporal_taps=suppress_cross_grid_temporal_taps,
        diagnostic_stats=diagnostic_stats,
        record_component=record_component,
        cuda_span=cuda_span,
    )
    if record_component is not None:
        record_component(
            "vdn_linear_api_host_wall_s",
            time.perf_counter() - total_started,
        )
    return result


def partitioned_frame_contract(plan) -> tuple[tuple[tuple[int, int], ...], tuple[float, ...]]:
    """Return per-frame grids and physical linear-state measure scales."""
    target = (int(plan.target_grid_h), int(plan.target_grid_w))
    source = (int(plan.source_grid_h), int(plan.source_grid_w))
    prefix_t = int(plan.prefix_t)
    temporal = int(plan.temporal)
    source_rows = int(plan.source_rows)
    target_rows = int(plan.target_rows)
    if not 0 < prefix_t < temporal or not 0 < source_rows < target_rows:
        raise RuntimeError("partitioned VDN linear received invalid Flow geometry")
    prefix_measure = source_rows / target_rows
    frame_sizes = (target,) * prefix_t + (source,) * (temporal - prefix_t)
    measures = (prefix_measure,) * prefix_t + (1.0,) * (temporal - prefix_t)
    return frame_sizes, measures


__all__ = [
    "partitioned_frame_contract",
    "partitioned_linear_readout",
]
