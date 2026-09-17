"""Runtime bridge for Flow partitioned exact-prefix sequences.

The first API-3 prototype reused VDN's historical external-sequence predicate.
That predicate intentionally disables both the grouped temporal softmax windows
and the geometry-dependent linear complement. It was appropriate for the retired
Mixed-Grid experiment, but it is too weak for the replacement production path.

This bridge instead wraps the already-installed VDN object patches on the cloned
MODEL. Ordinary calls delegate byte-for-byte to the released forward. Only an
explicit Flow partition contract selects the heterogeneous grouped softmax path.
The learned linear complement remains disabled for this experimental topology
until it has its own variable-grid oracle; that omission is explicit and remains
a decoded-media promotion gate.
"""

from __future__ import annotations

from dataclasses import dataclass
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
_BRIDGE_MARKER = "_vdn_partitioned_exact_prefix_bridge_v2"


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
    resources = state.runtime.current()
    if resources is None:
        raise RuntimeError("partitioned exact-prefix VDN requires an active runtime-buffer lease")
    index_identity = ("partitioned_exact_prefix_v1", grouped.plan_digest)

    q, k, v = qkv_proj(x).split(heads * head_dim, dim=-1)
    v = v.view(s, heads, head_dim)
    q_raw = q.view(s, heads, head_dim)
    k_raw = k.view(s, heads, head_dim)
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

    q, k, v = preprocess(options, q, k, v, heads)

    try:
        from sol_h3.partitioned_request import partitioned_request_attention
    except ImportError as exc:
        raise RuntimeError(
            "partitioned exact-prefix VDN requires the matching Sol-H3 partitioned backend branch"
        ) from exc

    scale = head_dim**-0.5
    softmax_out = torch.empty_like(q)
    covered_rows = grouped.video_start

    if grouped.video_start:
        global_index = _indices_from_ranges(
            ((0, grouped.video_start),),
            device=device,
            resources=resources,
            identity=(*index_identity, "global"),
        )
        softmax_out[global_index] = partitioned_request_attention(
            q[global_index],
            k,
            v,
            transformer_options=options,
            block_index=block_index,
            kind="global",
            scale=scale,
            sink_rows=0,
            prefix_k_range=grouped.full_prefix_k_range,
            prefix_log_key_measure=plan.prefix_log_key_measure,
            semantic_digest=semantic_digest,
            force_dense=True,
        )

    owner = state.query_position_owner_generation
    for group in grouped.groups:
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
            raise RuntimeError("partitioned VDN runtime gather does not match the CPU geometry plan")
        wire = group.wire(owner_generation=owner, plan_digest=grouped.plan_digest)
        measure = plan.prefix_log_key_measure if group.prefix_k_range is not None else 0.0
        softmax_out[q_index] = partitioned_request_attention(
            q[q_index],
            k[k_index],
            v[k_index],
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
        covered_rows += group.q_rows

    if grouped.anchor_slices:
        anchor_index = _indices_from_ranges(
            grouped.anchor_slices,
            device=device,
            resources=resources,
            identity=(*index_identity, "anchor"),
        )
        softmax_out[anchor_index] = partitioned_request_attention(
            q[anchor_index],
            k,
            v,
            transformer_options=options,
            block_index=block_index,
            kind="anchor",
            scale=scale,
            sink_rows=0,
            prefix_k_range=grouped.full_prefix_k_range,
            prefix_log_key_measure=plan.prefix_log_key_measure,
            semantic_digest=semantic_digest,
            force_dense=True,
        )
        covered_rows += int(anchor_index.numel())

    if covered_rows != grouped.sequence_rows:
        raise RuntimeError("partitioned VDN grouped queries do not cover the complete hidden sequence")

    weights = state.weights_on(block_index, device, dtype)
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

    # Variable-grid linear state is intentionally not approximated. The released
    # branch's spatial short-conv and recurrent statistics assume one fixed
    # tokens-per-frame value. Resizing the exact prefix merely for that branch
    # would reintroduce the context distortion this path exists to remove.
    from .hybrid import _once

    _once(
        (
            "partitioned-grouped",
            grouped.plan_digest,
            block_index,
            bool(cfg.get("linear_enabled", True)),
        ),
        "partitioned exact-prefix: grouped VDN softmax restored; variable-grid linear complement disabled pending oracle",
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
    "install_partitioned_external_sequence_bridge",
    "validate_partitioned_external_execution",
]
