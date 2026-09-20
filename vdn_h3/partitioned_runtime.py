"""Runtime bridge for Flow partitioned exact-prefix sequences.

The first API-3 prototype reused VDN's historical external-sequence predicate.
That predicate intentionally disabled both the grouped temporal softmax windows
and the geometry-dependent linear complement. It was appropriate for the retired
Mixed-Grid experiment, but it is too weak for the replacement production path.

API 4 wraps the already-installed VDN object patches on the cloned MODEL. Ordinary
calls delegate byte-for-byte to the released forward. Only an explicit Flow
partition contract selects heterogeneous grouped softmax plus the oracle-backed
variable-grid learned linear complement. Prefix frames retain the target grid,
suffix frames retain the source grid, and the linear state uses the same physical
per-frame measure correction as partitioned softmax.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
import time
from typing import Any

from .partitioned_grouped import PartitionedGroupedPlan, build_partitioned_grouped_plan
from .partitioned_sequence import (
    PARTITIONED_PREFIX_KEY,
    PARTITIONED_PREFIX_TOPOLOGY,
    VDN_PARTITIONED_SEQUENCE_API,
    VDN_PARTITIONED_SEQUENCE_MODE,
    make_vdn_partitioned_external_contract,
    validate_flow_partition_contract,
)

VDN_EXTERNAL_SEQUENCE_KEY = "vdn_h3_external_sequence_v1"
FLOW_PARTITIONED_STAGE_KEY = "h3_flow_partitioned_stage_v1"
SOL_CUDA_DIAGNOSTICS_KEY = "sol_h3_cuda_diagnostics_v1"
VDN_PARTITIONED_LINEAR_DIAGNOSTIC_KEY = "h3_flow_partitioned_vdn_linear_diagnostic_v1"
VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API = 1
VDN_PARTITIONED_LINEAR_DIAGNOSTIC_NORMAL = "normal"
VDN_PARTITIONED_LINEAR_DIAGNOSTIC_BYPASS = "bypass_partitioned_linear"
VDN_PARTITIONED_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL = "suppress_cross_grid_temporal_taps"
VDN_PARTITIONED_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE = "raw_token_measure"
VDN_PARTITIONED_LINEAR_DIAGNOSTIC_OPTIONS = (
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_NORMAL,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_BYPASS,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE,
)
_BRIDGE_MARKER = "_vdn_partitioned_exact_prefix_bridge_v3"


def _component_recorder(options):
    runtime = options.get(FLOW_PARTITIONED_STAGE_KEY) if isinstance(options, dict) else None
    recorder = getattr(runtime, "record_host_component", None)
    return recorder if callable(recorder) else None


def _record_component(recorder, name, started):
    if recorder is not None:
        recorder(name, time.perf_counter() - started)


def _partitioned_linear_diagnostic_mode(options: dict[str, Any]) -> str:
    raw = options.get(
        VDN_PARTITIONED_LINEAR_DIAGNOSTIC_KEY,
        VDN_PARTITIONED_LINEAR_DIAGNOSTIC_NORMAL,
    )
    if raw not in VDN_PARTITIONED_LINEAR_DIAGNOSTIC_OPTIONS:
        raise RuntimeError(
            "partitioned VDN linear diagnostic mode must be one of "
            f"{VDN_PARTITIONED_LINEAR_DIAGNOSTIC_OPTIONS!r}, got {raw!r}"
        )
    return str(raw)


def _resolve_partitioned_linear_runtime(layout, cfg, options: dict[str, Any]):
    """Return (active, bypassed, mode) without changing released/native VDN semantics."""

    would_run = bool(not layout.full_cover and cfg.get("linear_enabled", True))
    mode = _partitioned_linear_diagnostic_mode(options)
    bypassed = bool(
        would_run and mode == VDN_PARTITIONED_LINEAR_DIAGNOSTIC_BYPASS
    )
    return bool(would_run and not bypassed), bypassed, mode


def _record_partitioned_linear_bypass(options: dict[str, Any], video_rows: int) -> None:
    runtime = options.get(FLOW_PARTITIONED_STAGE_KEY)
    metrics = getattr(runtime, "metrics", None)
    increment = getattr(metrics, "increment", None)
    if not callable(increment):
        return
    increment("partitioned_vdn_linear_bypass_calls")
    increment("partitioned_vdn_linear_bypass_video_rows", int(video_rows))


def _record_partitioned_raw_token_measure(options: dict[str, Any], plan) -> None:
    if not math.isclose(float(plan.prefix_log_key_measure), 0.0, rel_tol=0.0, abs_tol=0.0):
        runtime = options.get(FLOW_PARTITIONED_STAGE_KEY)
        metrics = getattr(runtime, "metrics", None)
        increment = getattr(metrics, "increment", None)
        if callable(increment):
            increment("partitioned_vdn_raw_token_measure_calls")
            increment("partitioned_vdn_raw_token_measure_prefix_frames", int(plan.prefix_t))


def _resolve_partitioned_prefix_measure(plan, resolved_measure: float, *, has_prefix: bool) -> tuple[float, float]:
    requested = float(plan.prefix_log_key_measure) if has_prefix else 0.0
    applied = float(resolved_measure) if has_prefix else 0.0
    return requested, applied


def _record_partitioned_prefix_measure_route(
    options: dict[str, Any],
    *,
    route: str,
    requested: float,
    applied: float,
) -> None:
    if route not in {"global", "local", "anchor"}:
        raise RuntimeError(f"unsupported partitioned prefix-measure route {route!r}")
    runtime = options.get(FLOW_PARTITIONED_STAGE_KEY)
    metrics = getattr(runtime, "metrics", None)
    if metrics is None:
        return
    increment = getattr(metrics, "increment", None)
    if callable(increment):
        increment(f"partitioned_vdn_prefix_measure_{route}_calls")
        if not math.isclose(float(requested), float(applied), rel_tol=0.0, abs_tol=0.0):
            increment(f"partitioned_vdn_prefix_measure_{route}_adjustments")
    event = getattr(metrics, "event", None)
    if callable(event):
        event(
            "partitioned_vdn_prefix_measure_route",
            route=route,
            requested=float(requested),
            applied=float(applied),
            adjusted=not math.isclose(float(requested), float(applied), rel_tol=0.0, abs_tol=0.0),
        )


def _record_partitioned_cross_grid_temporal_suppression(
    options: dict[str, Any],
    stats: dict[str, int],
) -> None:
    suppressed_taps = int(stats.get("suppressed_taps", 0))
    suppressed_rows = int(stats.get("suppressed_rows", 0))
    if suppressed_taps <= 0 or suppressed_rows <= 0:
        raise RuntimeError(
            "partitioned VDN cross-grid temporal diagnostic was requested but no "
            "cross-grid short-conv taps were suppressed"
        )
    runtime = options.get(FLOW_PARTITIONED_STAGE_KEY)
    metrics = getattr(runtime, "metrics", None)
    increment = getattr(metrics, "increment", None)
    if not callable(increment):
        return
    increment("partitioned_vdn_cross_grid_temporal_suppression_calls")
    increment("partitioned_vdn_cross_grid_temporal_suppressed_taps", suppressed_taps)
    increment("partitioned_vdn_cross_grid_temporal_suppressed_rows", suppressed_rows)


@dataclass(frozen=True, slots=True)
class PartitionedQueryPositionSummary:
    tag: str
    schema: int
    mode: str
    owner_generation: str
    plan_digest: str
    seq_len: int
    video_start: int
    video_end: int
    num_frames: int
    tokens_per_frame: tuple[str, int, int]
    anchor_frames: str
    groups: tuple[tuple[Any, ...], ...]


def _closure_values(function):
    code = getattr(function, "__code__", None)
    cells = getattr(function, "__closure__", None)
    if code is None or cells is None or len(code.co_freevars) != len(cells):
        return None
    values = {}
    for name, cell in zip(code.co_freevars, cells, strict=True):
        try:
            values[name] = cell.cell_contents
        except ValueError:
            return None
    return values


def _mixed_layout_matches_plan(layout, plan) -> bool:
    if layout is None:
        return False
    signature = getattr(layout, "signature", None)
    segments = getattr(layout, "segments", None)
    return bool(
        int(getattr(layout, "seq_len", -1)) == plan.sequence_rows
        and isinstance(signature, tuple)
        and signature
        and signature[0] == PARTITIONED_PREFIX_KEY
        and isinstance(segments, (tuple, list))
        and segments
        and tuple(segments[-1]) == (plan.video_start, plan.sequence_rows, "video")
    )


def validate_partitioned_external_execution(
    transformer_options: dict[str, Any],
    layout: Any,
    sequence_rows: int,
    rope_freqs: Any,
):
    """Validate the complete Flow+VDN partitioned wire contract for one call."""
    if not isinstance(transformer_options, dict):
        raise RuntimeError("VDN partitioned execution requires transformer options")
    flow_contract = transformer_options.get(PARTITIONED_PREFIX_KEY)
    plan = validate_flow_partition_contract(flow_contract, sequence_rows=int(sequence_rows))

    external = transformer_options.get(VDN_EXTERNAL_SEQUENCE_KEY)
    expected = make_vdn_partitioned_external_contract(plan)
    if not isinstance(external, dict):
        raise RuntimeError("VDN partitioned execution is missing its external-sequence contract")
    if external != expected:
        raise RuntimeError("VDN partitioned external-sequence contract does not match Flow geometry")
    if (
        external.get("api") != VDN_PARTITIONED_SEQUENCE_API
        or external.get("mode") != VDN_PARTITIONED_SEQUENCE_MODE
        or external.get("topology") != PARTITIONED_PREFIX_TOPOLOGY
    ):
        raise RuntimeError("VDN partitioned external-sequence mode is unsupported")

    native_rows = plan.video_start + plan.temporal * plan.source_rows
    if (
        int(getattr(layout, "seq_len", -1)) != native_rows
        or int(getattr(layout, "video_start", -1)) != plan.video_start
        or int(getattr(layout, "video_end", -1)) != native_rows
        or int(getattr(layout, "num_frames", -1)) != plan.temporal
        or int(getattr(layout, "tokens_per_frame", -1)) != plan.source_rows
    ):
        raise RuntimeError("VDN native low-grid layout does not match partitioned Flow geometry")
    if rope_freqs is None or getattr(rope_freqs, "ndim", 0) < 2:
        raise RuntimeError("VDN partitioned execution requires explicit mixed-domain RoPE rows")
    if int(rope_freqs.shape[1]) != int(sequence_rows):
        raise RuntimeError("VDN partitioned RoPE rows do not match the mixed hidden sequence")
    return plan


def _grouped_plan(plan, layout, *, semantic_digest: str) -> PartitionedGroupedPlan:
    return build_partitioned_grouped_plan(
        plan,
        bounds=layout.bounds,
        anchor_frames=layout.anchor_frames,
        semantic_digest=semantic_digest,
    )


def _indices_from_ranges(ranges, *, device, resources, identity):
    if resources is None:
        raise RuntimeError("partitioned exact-prefix VDN requires active runtime buffers")
    return resources.partition_indices(identity, ranges, device)


def _partitioned_query_summary(current, values, options, layout):
    flow_contract = (options or {}).get(PARTITIONED_PREFIX_KEY)
    if flow_contract is None:
        released = getattr(current, "vdn_query_position_plan_v1", None)
        return released(options, layout) if callable(released) else None
    try:
        state = values["state"]
        if getattr(state, "softmax_backend", None) != "grouped":
            return None
        plan = validate_flow_partition_contract(
            flow_contract,
            sequence_rows=int(flow_contract.get("sequence_rows", -1)),
        )
        if not _mixed_layout_matches_plan(layout, plan):
            return None
        external = (options or {}).get(VDN_EXTERNAL_SEQUENCE_KEY)
        if external != make_vdn_partitioned_external_contract(plan):
            return None
        cfg = values["cfg"]
        from .window import window_bounds

        bounds = window_bounds(plan.temporal, cfg["radius"], cfg["chunk"])
        grouped = build_partitioned_grouped_plan(
            plan,
            bounds=bounds,
            anchor_frames=cfg["anchor_frames"],
            semantic_digest=str(flow_contract["semantic_digest"]),
        )
        owner = state.query_position_owner_generation
        return PartitionedQueryPositionSummary(
            tag="vdn_query_position_plan_v1",
            schema=1,
            mode="grouped",
            owner_generation=owner,
            plan_digest=grouped.plan_digest,
            seq_len=grouped.sequence_rows,
            video_start=grouped.video_start,
            video_end=grouped.sequence_rows,
            num_frames=grouped.temporal,
            tokens_per_frame=("heterogeneous", grouped.target_rows, grouped.source_rows),
            anchor_frames=grouped.anchor_frames,
            groups=grouped.wires(owner),
        )
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None


def _partitioned_vdn_forward(current, values, x, rope_freqs, transformer_options):
    import torch
    import torch.nn.functional as F

    import comfy.model_management
    import comfy.quant_ops

    options = transformer_options or {}
    record_component = _component_recorder(options)
    state = values["state"]
    if getattr(state, "softmax_backend", None) != "grouped":
        raise RuntimeError("partitioned exact-prefix VDN requires the grouped softmax backend")
    base_branch = values["base_branch"]
    if base_branch is None:
        raise RuntimeError("partitioned exact-prefix VDN requires an active VDN branch")
    layout = state.layout
    if layout is None:
        raise RuntimeError("partitioned exact-prefix VDN requires an active native low-grid layout")
    plan = validate_partitioned_external_execution(options, layout, int(x.shape[0]), rope_freqs)
    flow_contract = options[PARTITIONED_PREFIX_KEY]
    semantic_digest = str(flow_contract["semantic_digest"])
    grouped = _grouped_plan(plan, layout, semantic_digest=semantic_digest)
    if grouped.sequence_rows != int(x.shape[0]):
        raise RuntimeError("partitioned VDN grouped plan does not match hidden-state rows")

    qkv_proj = values["qkv_proj"]
    out_proj = values["out_proj"]
    q_norm = values["q_norm"]
    k_norm = values["k_norm"]
    heads = int(values["heads"])
    head_dim = int(values["head_dim"])
    block_index = int(values["block_index"])
    cfg = values["cfg"]
    s = int(x.shape[0])
    device, dtype = x.device, x.dtype
    cuda_diagnostics = options.get(SOL_CUDA_DIAGNOSTICS_KEY)
    cuda_sample = None
    if getattr(cuda_diagnostics, "enabled", False):
        cuda_sample = cuda_diagnostics.begin_sample(
            "vdn_partitioned_components",
            device,
            context={
                "flow_request_id": options.get("h3_flow_request_id_v1"),
                "flow_stage": options.get("h3_flow_stage"),
                "flow_stage_id": options.get("h3_flow_stage_id_v1"),
                "flow_evaluation_id": options.get("h3_flow_evaluation_id_v1"),
                "block_index": block_index,
                "plan_digest": grouped.plan_digest,
            },
        )

    def cuda_span(name):
        if cuda_sample is None:
            return nullcontext()
        return cuda_diagnostics.span(cuda_sample, name)

    resources = state.runtime.current()
    if resources is None:
        raise RuntimeError("partitioned exact-prefix VDN requires an active runtime-buffer lease")
    index_identity = ("partitioned_exact_prefix_v1", grouped.plan_digest)

    q, k, v = qkv_proj(x).split(heads * head_dim, dim=-1)
    v = v.view(s, heads, head_dim)
    q_raw = q.view(s, heads, head_dim)
    k_raw = k.view(s, heads, head_dim)

    # The released VDN linear branch consumes raw pre-QK-norm/pre-RoPE video
    # features. Preserve exactly those rows before the in-place H3 RoPE helper.
    # The diagnostic bypass is deliberately scoped to the explicit partitioned
    # runtime only. Ordinary/native VDN never reaches this function, and normal
    # mode remains byte-for-byte on the existing branch.
    linear_active, linear_bypassed, linear_diagnostic_mode = _resolve_partitioned_linear_runtime(
        layout,
        cfg,
        options,
    )
    if linear_bypassed:
        _record_partitioned_linear_bypass(
            options,
            grouped.sequence_rows - grouped.video_start,
        )
    q_raw_video = k_raw_video = v_raw_video = None
    text_x = text_k_raw = text_v_raw = None
    if linear_active:
        video_start = grouped.video_start
        video_end = grouped.sequence_rows
        text_len = int(layout.text_len) if base_branch.enable_text_state else 0
        scratch = resources.activation_scratch(
            video_end - video_start,
            text_len,
            heads,
            head_dim,
            device,
            dtype,
        )
        if scratch is None:
            q_raw_video = q_raw[video_start:video_end].clone()
            k_raw_video = k_raw[video_start:video_end].clone()
            v_raw_video = v[video_start:video_end].clone()
        else:
            q_raw_video = scratch["q"]
            k_raw_video = scratch["k"]
            v_raw_video = scratch["v"]
            q_raw_video.copy_(q_raw[video_start:video_end])
            k_raw_video.copy_(k_raw[video_start:video_end])
            v_raw_video.copy_(v[video_start:video_end])

        if base_branch.enable_text_state and int(layout.text_len):
            text_start = int(layout.text_start)
            text_end = text_start + int(layout.text_len)
            if not 0 <= text_start < text_end <= grouped.video_start:
                raise RuntimeError("partitioned VDN text rows are outside the non-video prefix")
            text_x = x[text_start:text_end]
            if scratch is None:
                text_k_raw = k_raw[text_start:text_end].clone()
                text_v_raw = v[text_start:text_end].clone()
            else:
                text_k_raw = scratch["tk"]
                text_v_raw = scratch["tv"]
                text_k_raw.copy_(k_raw[text_start:text_end])
                text_v_raw.copy_(v[text_start:text_end])

    if rope_freqs is not None:
        q4 = q.view(1, s, heads, head_dim)
        k4 = k.view(1, s, heads, head_dim)
        qw = comfy.model_management.cast_to(q_norm.weight, device=device)
        kw = comfy.model_management.cast_to(k_norm.weight, device=device)
        rot = rope_freqs.shape[-3] * 2
        comfy.quant_ops.ck.rms_rope_split_half_(
            q4,
            k4,
            rope_freqs,
            qw,
            kw,
            epsilon=q_norm.eps,
            rot_dim=rot,
        )
        q, k = q4[0], k4[0]
    else:
        q = q_norm(q_raw)
        k = k_norm(k_raw)
    v = v.clone()

    # Apply the inherited key/QKV preprocessing once on the complete physical
    # sequence before any VDN gather. This preserves transformations whose
    # semantics depend on original packed-row coordinates.
    from .softmax_provider import preprocess

    preprocess_started = time.perf_counter()
    with cuda_span("vdn_preprocess"):
        q, k, v = preprocess(options, q, k, v, heads)
    _record_component(record_component, "vdn_preprocess_host_wall_s", preprocess_started)

    try:
        from sol_h3.partitioned_request import partitioned_request_attention
    except ImportError as exc:
        raise RuntimeError(
            "partitioned exact-prefix VDN requires the matching Sol-H3 partitioned backend branch"
        ) from exc

    scale = head_dim**-0.5
    raw_token_measure = (
        linear_diagnostic_mode == VDN_PARTITIONED_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE
    )
    if raw_token_measure and not linear_active:
        raise RuntimeError(
            "partitioned VDN raw-token measure diagnostic requires the learned linear complement"
        )
    prefix_log_key_measure = 0.0 if raw_token_measure else plan.prefix_log_key_measure
    softmax_out = torch.empty_like(q)
    covered_rows = grouped.video_start

    if grouped.video_start:
        global_requested_measure, global_applied_measure = _resolve_partitioned_prefix_measure(
            plan,
            prefix_log_key_measure,
            has_prefix=grouped.full_prefix_k_range is not None,
        )
        _record_partitioned_prefix_measure_route(
            options,
            route="global",
            requested=global_requested_measure,
            applied=global_applied_measure,
        )
        gather_started = time.perf_counter()
        with cuda_span("vdn_gather"):
            global_index = _indices_from_ranges(
                ((0, grouped.video_start),),
                device=device,
                resources=resources,
                identity=(*index_identity, "global"),
            )
            q_global = q[global_index]
        _record_component(record_component, "vdn_gather_host_wall_s", gather_started)
        softmax_started = time.perf_counter()
        with cuda_span("vdn_softmax"):
            softmax_out[global_index] = partitioned_request_attention(
                q_global,
                k,
                v,
                transformer_options=options,
                block_index=block_index,
                kind="global",
                scale=scale,
                sink_rows=0,
                prefix_k_range=grouped.full_prefix_k_range,
                prefix_log_key_measure=global_applied_measure,
                semantic_digest=semantic_digest,
                force_dense=True,
            )
        _record_component(record_component, "vdn_softmax_host_wall_s", softmax_started)
        del q_global

    owner = state.query_position_owner_generation
    for group in grouped.groups:
        gather_started = time.perf_counter()
        with cuda_span("vdn_gather"):
            q_index = _indices_from_ranges(
                group.q_ranges,
                device=device,
                resources=resources,
                identity=(*index_identity, "q", group.group_index),
            )
            global_range = ((0, grouped.video_start),) if grouped.video_start else ()
            k_index = _indices_from_ranges(
                (*global_range, *group.key_ranges),
                device=device,
                resources=resources,
                identity=(*index_identity, "k", group.group_index),
            )
            if q_index.numel() != group.q_rows or k_index.numel() != group.kv_rows:
                raise RuntimeError(
                    "partitioned VDN runtime gather does not match the CPU geometry plan"
                )
            wire = group.wire(owner_generation=owner, plan_digest=grouped.plan_digest)
            requested_measure, measure = _resolve_partitioned_prefix_measure(
                plan,
                prefix_log_key_measure,
                has_prefix=group.prefix_k_range is not None,
            )
            _record_partitioned_prefix_measure_route(
                options,
                route="local",
                requested=requested_measure,
                applied=measure,
            )
            q_group = q[q_index]
            k_group = k[k_index]
            v_group = v[k_index]
        _record_component(record_component, "vdn_gather_host_wall_s", gather_started)
        softmax_started = time.perf_counter()
        with cuda_span("vdn_softmax"):
            softmax_out[q_index] = partitioned_request_attention(
                q_group,
                k_group,
                v_group,
                transformer_options=options,
                block_index=block_index,
                kind="local",
                scale=scale,
                sink_rows=group.sink_rows,
                prefix_k_range=group.prefix_k_range,
                prefix_log_key_measure=measure,
                semantic_digest=semantic_digest,
                query_position_map=wire,
                # Target-prefix hidden rows become K/V context for every deeper
                # block. Keep those query updates exact; suffix query groups use the
                # mapped sparse Sol route.
                force_dense=group.query_prefix_domain,
            )
        _record_component(record_component, "vdn_softmax_host_wall_s", softmax_started)
        del q_group, k_group, v_group
        covered_rows += group.q_rows

    if grouped.anchor_slices:
        anchor_requested_measure, anchor_applied_measure = _resolve_partitioned_prefix_measure(
            plan,
            prefix_log_key_measure,
            has_prefix=grouped.full_prefix_k_range is not None,
        )
        _record_partitioned_prefix_measure_route(
            options,
            route="anchor",
            requested=anchor_requested_measure,
            applied=anchor_applied_measure,
        )
        gather_started = time.perf_counter()
        with cuda_span("vdn_gather"):
            anchor_index = _indices_from_ranges(
                grouped.anchor_slices,
                device=device,
                resources=resources,
                identity=(*index_identity, "anchor"),
            )
            q_anchor = q[anchor_index]
        _record_component(record_component, "vdn_gather_host_wall_s", gather_started)
        softmax_started = time.perf_counter()
        with cuda_span("vdn_softmax"):
            softmax_out[anchor_index] = partitioned_request_attention(
                q_anchor,
                k,
                v,
                transformer_options=options,
                block_index=block_index,
                kind="anchor",
                scale=scale,
                sink_rows=0,
                prefix_k_range=grouped.full_prefix_k_range,
                prefix_log_key_measure=anchor_applied_measure,
                semantic_digest=semantic_digest,
                force_dense=True,
            )
        _record_component(record_component, "vdn_softmax_host_wall_s", softmax_started)
        del q_anchor
        covered_rows += int(anchor_index.numel())

    if covered_rows != grouped.sequence_rows:
        raise RuntimeError("partitioned VDN grouped queries do not cover the complete hidden sequence")

    weights_started = time.perf_counter()
    with cuda_span("vdn_weights"):
        weights = state.weights_on(block_index, device, dtype)
    _record_component(record_component, "vdn_weights_host_wall_s", weights_started)
    softmax_epilogue_started = time.perf_counter()
    with cuda_span("vdn_softmax_epilogue"):
        if cfg["enable_softmax_gate"]:
            gate = torch.sigmoid(
                F.linear(
                    x,
                    weights["softmax_gate.up.weight"],
                    weights["softmax_gate.up.bias"],
                )
            )
            flat = (softmax_out * gate.view(s, heads, 1).to(softmax_out.dtype)).reshape(s, -1)
        else:
            flat = softmax_out.reshape(s, -1)
        out = out_proj(flat.type_as(x))
    _record_component(
        record_component,
        "vdn_softmax_epilogue_host_wall_s",
        softmax_epilogue_started,
    )

    linear_added = False
    cross_grid_temporal_suppressed = False
    cross_grid_temporal_stats: dict[str, int] | None = None
    if linear_active:
        from .partitioned_linear import partitioned_frame_contract, partitioned_linear_readout

        suppress_cross_grid_temporal = (
            linear_diagnostic_mode == VDN_PARTITIONED_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL
        )
        cross_grid_temporal_stats = {} if suppress_cross_grid_temporal else None
        frame_sizes, measure_scales = partitioned_frame_contract(plan)
        if raw_token_measure:
            measure_scales = tuple(1.0 for _ in measure_scales)
        linear_started = time.perf_counter()
        readout = partitioned_linear_readout(
            base_branch,
            weights,
            x[grouped.video_start : grouped.sequence_rows],
            q_raw_video,
            k_raw_video,
            v_raw_video,
            frame_sizes=frame_sizes,
            bounds=tuple(tuple(int(value) for value in pair) for pair in layout.bounds),
            measure_scales=measure_scales,
            text_x=text_x,
            text_k_raw=text_k_raw,
            text_v_raw=text_v_raw,
            skip_ends=(cfg["anchor_frames"] == "both"),
            suppress_cross_grid_temporal_taps=suppress_cross_grid_temporal,
            diagnostic_stats=cross_grid_temporal_stats,
            record_component=record_component,
            cuda_span=cuda_span,
        )
        if cross_grid_temporal_stats is not None:
            _record_partitioned_cross_grid_temporal_suppression(
                options,
                cross_grid_temporal_stats,
            )
            cross_grid_temporal_suppressed = True
        _record_component(
            record_component,
            "vdn_linear_readout_total_host_wall_s",
            linear_started,
        )
        expected_shape = (grouped.sequence_rows - grouped.video_start, heads * head_dim)
        if tuple(readout.shape) != expected_shape:
            raise RuntimeError("partitioned VDN linear complement returned incompatible rows")
        linear_projection_started = time.perf_counter()
        with cuda_span("vdn_linear_projection"):
            out[grouped.video_start : grouped.sequence_rows] += F.linear(
                readout.type_as(x),
                weights["to_out_linear.weight"],
            )
        _record_component(
            record_component,
            "vdn_linear_projection_host_wall_s",
            linear_projection_started,
        )
        linear_added = True
        if raw_token_measure:
            _record_partitioned_raw_token_measure(options, plan)

    from .hybrid import _once

    if linear_bypassed:
        _once(
            (
                "partitioned-grouped-v4",
                grouped.plan_digest,
                block_index,
                "diagnostic-bypassed",
            ),
            "partitioned exact-prefix: grouped VDN softmax active; variable-grid linear complement "
            f"diagnostic-bypassed; diagnostic_mode={linear_diagnostic_mode}",
        )
    elif raw_token_measure:
        _once(
            (
                "partitioned-grouped-v4",
                grouped.plan_digest,
                block_index,
                "diagnostic-raw-token-measure",
            ),
            "partitioned exact-prefix: grouped VDN softmax and learned-linear complement active; "
            "target-prefix density correction diagnostic-disabled so both paths use raw token measure; "
            f"diagnostic_mode={linear_diagnostic_mode}",
        )
    elif cross_grid_temporal_suppressed:
        _once(
            (
                "partitioned-grouped-v4",
                grouped.plan_digest,
                block_index,
                "diagnostic-cross-grid-temporal-suppressed",
            ),
            "partitioned exact-prefix: grouped VDN softmax active; variable-grid linear complement "
            "active with cross-grid temporal short-conv taps diagnostic-suppressed; "
            f"diagnostic_mode={linear_diagnostic_mode}",
        )
    else:
        # Preserve the existing normal-path logging identity and wording exactly.
        _once(
            (
                "partitioned-grouped-v4",
                grouped.plan_digest,
                block_index,
                linear_added,
            ),
            "partitioned exact-prefix: grouped VDN softmax active; variable-grid linear complement "
            + ("active" if linear_added else "inactive by released full-coverage/config semantics"),
        )
    return out


def _wrap_vdn_forward(current):
    if getattr(current, _BRIDGE_MARKER, False):
        return current
    if not getattr(current, "_vdn_forward", False):
        raise RuntimeError("partitioned exact-prefix can only wrap a VDN-owned attention forward")
    values = _closure_values(current)
    required = {
        "base_branch",
        "block_index",
        "cfg",
        "head_dim",
        "heads",
        "k_norm",
        "out_proj",
        "q_norm",
        "qkv_proj",
        "state",
    }
    if values is None or not required.issubset(values):
        raise RuntimeError("partitioned exact-prefix cannot recover the VDN forward ownership contract")

    def partitioned_aware(x, rope_freqs=None, transformer_options=None):
        options = transformer_options or {}
        if options.get(PARTITIONED_PREFIX_KEY) is None:
            return current(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
        return _partitioned_vdn_forward(current, values, x, rope_freqs, options)

    def query_position_plan(options, layout):
        return _partitioned_query_summary(current, values, options, layout)

    setattr(partitioned_aware, _BRIDGE_MARKER, True)
    partitioned_aware._vdn_forward = True
    partitioned_aware._vdn_external_sequence_api = VDN_PARTITIONED_SEQUENCE_API
    partitioned_aware._vdn_partitioned_linear_diagnostic_api = VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API
    partitioned_aware._vdn_partitioned_linear_diagnostic_modes = VDN_PARTITIONED_LINEAR_DIAGNOSTIC_OPTIONS
    partitioned_aware._vdn_partitioned_released_forward = current
    partitioned_aware.vdn_query_position_plan_v1 = query_position_plan
    return partitioned_aware


def install_partitioned_external_sequence_bridge(model) -> None:
    """Install model-local partition-aware VDN forwards on the cloned MODEL."""
    object_patches = getattr(model, "object_patches", None)
    if not isinstance(object_patches, dict):
        raise RuntimeError("partitioned exact-prefix requires Comfy ModelPatcher object patches")
    matched = 0
    for key, current in tuple(object_patches.items()):
        if not key.startswith("diffusion_model.blocks.") or not key.endswith(".attn.forward"):
            continue
        if not getattr(current, "_vdn_forward", False):
            continue
        matched += 1
        model.add_object_patch(key, _wrap_vdn_forward(current))
    if matched == 0:
        raise RuntimeError("partitioned exact-prefix requires VDN-H3 to be applied before Flow")


__all__ = [
    "PartitionedQueryPositionSummary",
    "VDN_EXTERNAL_SEQUENCE_KEY",
    "VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API",
    "VDN_PARTITIONED_LINEAR_DIAGNOSTIC_BYPASS",
    "VDN_PARTITIONED_LINEAR_DIAGNOSTIC_KEY",
    "VDN_PARTITIONED_LINEAR_DIAGNOSTIC_NORMAL",
    "VDN_PARTITIONED_LINEAR_DIAGNOSTIC_OPTIONS",
    "VDN_PARTITIONED_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE",
    "VDN_PARTITIONED_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL",
    "install_partitioned_external_sequence_bridge",
    "validate_partitioned_external_execution",
]
