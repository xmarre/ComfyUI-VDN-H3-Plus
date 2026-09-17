"""Variable-grid extension of VDN's learned linear complement.

The released linear branch assumes one fixed spatial token grid for every video
frame. Partitioned exact-prefix continuation deliberately violates that assumption:
protected prefix frames stay on the target grid while generated suffix frames stay
on the source grid.

This module extends only the released ``vdn_solve`` inference path. Per-frame
statistics and recurrence remain the released VDN arithmetic. Spatial short-conv
runs on each frame's native grid. The temporal depthwise short-conv uses the same
kernel and zero-padding as released VDN; only taps crossing a grid boundary are
mapped to the destination frame's physical grid with center-aligned bilinear
interpolation in FP32. When all frame grids are identical, no interpolation occurs
and the helper reduces to the released fixed-grid computation.

Target-prefix frame statistics are weighted by source_rows / target_rows so the
linear state uses the same physical per-frame carrier measure as the partitioned
softmax contract instead of overweighting the denser prefix representation.
"""

from __future__ import annotations

import copy
import math
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
    for index, ((lo, hi), scale) in enumerate(zip(bounds, measure_scales, strict=True)):
        if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi < len(frame_sizes):
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


def _map_temporal_neighbor(source: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if tuple(source.shape[-2:]) == tuple(target_hw):
        return source
    # Interpolate only cross-grid temporal taps. FP32 keeps the coordinate mapping
    # deterministic across BF16 model execution before returning to branch dtype.
    return F.interpolate(
        source.float(),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    ).to(source.dtype)


def _heterogeneous_conv_features(
    tokens: torch.Tensor,
    spatial_weight: torch.Tensor,
    temporal_weight: torch.Tensor,
    frame_sizes: Sequence[tuple[int, int]],
    offsets: Sequence[tuple[int, int]],
    *,
    l2norm: bool,
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
            source = _map_temporal_neighbor(maps[source_frame], (target_h, target_w))
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


def _variable_features(branch, weights, q_raw, k_raw, v_raw, frame_sizes, offsets):
    conv = tuple(getattr(branch, "short_conv", ()) or ())

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
    query, key, value = _variable_features(
        branch,
        weights,
        q_raw,
        k_raw,
        v_raw,
        frame_sizes,
        offsets,
    )
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
    gate = torch.sigmoid(
        F.linear(x_video, weights["output_gate.down.weight"])
        @ weights["output_gate.up.weight"].T
        + weights["output_gate.up.bias"]
    )
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

    outputs = []
    eps = weights["norm.weight"].new_tensor(1e-6).item()
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
    return torch.cat(outputs, dim=0)


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
) -> torch.Tensor:
    """Evaluate VDN's learned linear complement over heterogeneous frame grids."""
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
            return x_video.new_zeros((x_video.shape[0], heads * head_dim))
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
        )
        out = inner.new_zeros((x_video.shape[0], inner.shape[-1]))
        out[first_stop:last_start] = inner
        return out

    return _core_readout(
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
    )


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
