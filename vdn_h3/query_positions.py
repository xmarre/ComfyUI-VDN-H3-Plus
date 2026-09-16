"""Pure CPU geometry and v4 query-position transport for grouped VDN attention.

VDN owns the mapping from each requested local query row to its row in the
already-gathered restricted K/V domain.  This module is deliberately tensor- and
CUDA-free: the same immutable geometry drives retained.py's actual gathers and
Sol-H3's preflight/history contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import json
from typing import Any

WIRE_TAG = "vdn_query_positions"
WIRE_SCHEMA = 1
PLAN_TAG = "vdn_query_position_plan_v1"
PLAN_SCHEMA = 1
GEOMETRY_SCHEMA = "vdn-grouped-window-geometry-v1"
_INT32_MAX = 2**31 - 1


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > _INT32_MAX:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]")
    return value


def _merge_runs(runs: list[tuple[int, int, int]]) -> tuple[tuple[int, int, int], ...]:
    merged: list[tuple[int, int, int]] = []
    for q_begin, q_end, kv_begin in runs:
        if merged:
            prev_q_begin, prev_q_end, prev_kv_begin = merged[-1]
            prev_offset = prev_kv_begin - prev_q_begin
            if prev_q_end == q_begin and kv_begin - q_begin == prev_offset:
                merged[-1] = (prev_q_begin, q_end, prev_kv_begin)
                continue
        merged.append((q_begin, q_end, kv_begin))
    return tuple(merged)


@dataclass(frozen=True)
class GroupGeometry:
    group_index: int
    query_frames: tuple[int, ...]
    key_frames: tuple[int, ...]
    q_rows: int
    kv_rows: int
    sink_rows: int
    square_aligned: bool
    query_position_runs: tuple[tuple[int, int, int], ...]
    map_digest: str


@dataclass(frozen=True)
class WindowGeometry:
    schema: str
    seq_len: int
    video_start: int
    video_end: int
    num_frames: int
    tokens_per_frame: int
    bounds: tuple[tuple[int, int], ...]
    anchor_frames: str
    global_rows: tuple[int, ...]
    anchor_slices: tuple[tuple[int, int], ...]
    groups: tuple[GroupGeometry, ...]
    max_kv_rows: int
    plan_digest: str


@dataclass(frozen=True)
class QueryPositionPlan:
    tag: str
    schema: int
    mode: str
    owner_generation: str
    plan_digest: str
    seq_len: int
    video_start: int
    video_end: int
    num_frames: int
    tokens_per_frame: int
    anchor_frames: str
    groups: tuple[tuple[Any, ...], ...]


def _validate_anchor_mode(anchor_frames: str) -> str:
    if anchor_frames not in {"none", "columns", "rows", "both"}:
        raise ValueError(f"unsupported VDN anchor mode {anchor_frames!r}")
    return anchor_frames


def describe_window_geometry(
    video_start: int,
    video_end: int,
    num_frames: int,
    tokens_per_frame: int,
    bounds: tuple[tuple[int, int], ...],
    anchor_frames: str,
    seq_len: int,
) -> WindowGeometry:
    """Describe exactly the grouped gather topology without allocating tensors."""
    return _describe_window_geometry_cached(
        GEOMETRY_SCHEMA,
        video_start,
        video_end,
        num_frames,
        tokens_per_frame,
        bounds,
        anchor_frames,
        seq_len,
    )


@functools.lru_cache(maxsize=8)
def _describe_window_geometry_cached(
    geometry_schema: str,
    video_start: int,
    video_end: int,
    num_frames: int,
    tokens_per_frame: int,
    bounds: tuple[tuple[int, int], ...],
    anchor_frames: str,
    seq_len: int,
) -> WindowGeometry:
    if geometry_schema != GEOMETRY_SCHEMA:
        raise ValueError("VDN query-position geometry schema is unsupported")
    video_start = _strict_int(video_start, "video_start")
    video_end = _strict_int(video_end, "video_end")
    num_frames = _strict_int(num_frames, "num_frames", minimum=1)
    tokens_per_frame = _strict_int(tokens_per_frame, "tokens_per_frame", minimum=1)
    seq_len = _strict_int(seq_len, "seq_len", minimum=1)
    anchor_frames = _validate_anchor_mode(anchor_frames)
    if not video_start <= video_end <= seq_len:
        raise ValueError("VDN video interval is outside the packed sequence")
    if video_end - video_start != num_frames * tokens_per_frame:
        raise ValueError("VDN video interval does not match frame geometry")
    if len(bounds) != num_frames:
        raise ValueError("VDN window bounds do not cover every video frame")

    normalized_bounds: list[tuple[int, int]] = []
    for index, pair in enumerate(bounds):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"VDN bound {index} must contain (lo, hi)")
        if type(pair[0]) is not int or type(pair[1]) is not int:
            raise ValueError(f"VDN bound {index} must use integer endpoints")
        lo = max(pair[0], 0)
        hi = min(pair[1], num_frames - 1)
        _strict_int(lo, f"bounds[{index}].lo")
        _strict_int(hi, f"bounds[{index}].hi")
        if lo > hi:
            raise ValueError(f"VDN bound {index} is empty after clamping")
        normalized_bounds.append((lo, hi))
    bounds_tuple = tuple(normalized_bounds)

    global_rows = tuple(range(video_start)) + tuple(range(video_end, seq_len))
    sink_rows = len(global_rows)
    anchors = (0, num_frames - 1)
    anchor_rows = tuple(
        frame for frame in anchors if anchor_frames in {"rows", "both"}
    )
    anchor_set = set(anchor_rows)

    grouped: dict[tuple[int, int], list[int]] = {}
    for frame in range(num_frames):
        if frame in anchor_set:
            continue
        grouped.setdefault(bounds_tuple[frame], []).append(frame)

    groups: list[GroupGeometry] = []
    max_kv_rows = sink_rows
    for group_index, ((lo, hi), frames_list) in enumerate(grouped.items()):
        query_frames = tuple(frames_list)
        extra = tuple(
            frame
            for frame in anchors
            if anchor_frames in {"columns", "both"} and not lo <= frame <= hi
        )
        key_frames = tuple(sorted(set(range(lo, hi + 1)) | set(extra)))
        if len(key_frames) != len(set(key_frames)) or any(
            first >= second for first, second in zip(key_frames, key_frames[1:])
        ):
            raise ValueError("VDN grouped key-frame domain is not sorted and unique")
        rank = {frame: index for index, frame in enumerate(key_frames)}
        if any(frame not in rank for frame in query_frames):
            raise ValueError("VDN grouped query frame is absent from its restricted K/V domain")

        q_rows = len(query_frames) * tokens_per_frame
        kv_rows = sink_rows + len(key_frames) * tokens_per_frame
        runs: list[tuple[int, int, int]] = []
        for query_rank, frame in enumerate(query_frames):
            q_begin = query_rank * tokens_per_frame
            q_end = q_begin + tokens_per_frame
            kv_begin = sink_rows + rank[frame] * tokens_per_frame
            runs.append((q_begin, q_end, kv_begin))
        merged = _merge_runs(runs)
        map_payload = {
            "schema": WIRE_SCHEMA,
            "group_index": group_index,
            "q_rows": q_rows,
            "kv_rows": kv_rows,
            "sink_rows": sink_rows,
            "runs": merged,
        }
        groups.append(
            GroupGeometry(
                group_index=group_index,
                query_frames=query_frames,
                key_frames=key_frames,
                q_rows=q_rows,
                kv_rows=kv_rows,
                sink_rows=sink_rows,
                square_aligned=(sink_rows == 0 and query_frames == key_frames),
                query_position_runs=merged,
                map_digest=_sha256_json(map_payload),
            )
        )
        max_kv_rows = max(max_kv_rows, kv_rows)

    anchor_slices = tuple(
        (
            video_start + frame * tokens_per_frame,
            video_start + (frame + 1) * tokens_per_frame,
        )
        for frame in anchor_rows
    )
    plan_payload = {
        "schema": geometry_schema,
        "seq_len": seq_len,
        "video_start": video_start,
        "video_end": video_end,
        "num_frames": num_frames,
        "tokens_per_frame": tokens_per_frame,
        "bounds": bounds_tuple,
        "anchor_frames": anchor_frames,
        "global_rows": global_rows,
        "groups": [
            {
                "group_index": group.group_index,
                "query_frames": group.query_frames,
                "key_frames": group.key_frames,
                "square_aligned": group.square_aligned,
                "q_rows": group.q_rows,
                "kv_rows": group.kv_rows,
                "sink_rows": group.sink_rows,
                "query_position_runs": group.query_position_runs,
            }
            for group in groups
        ],
        "anchor_slices": anchor_slices,
    }
    return WindowGeometry(
        schema=geometry_schema,
        seq_len=seq_len,
        video_start=video_start,
        video_end=video_end,
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        bounds=bounds_tuple,
        anchor_frames=anchor_frames,
        global_rows=global_rows,
        anchor_slices=anchor_slices,
        groups=tuple(groups),
        max_kv_rows=max_kv_rows,
        plan_digest=_sha256_json(plan_payload),
    )


def bind_query_map(
    geometry: WindowGeometry,
    group_index: int,
    owner_generation: str,
) -> tuple[Any, ...]:
    """Bind pure geometry to one Apply-VDN ownership generation."""
    if not isinstance(owner_generation, str) or not owner_generation:
        raise ValueError("VDN query-position owner generation is missing")
    group_index = _strict_int(group_index, "group_index")
    if group_index >= len(geometry.groups):
        raise ValueError("VDN query-position group index is outside the plan")
    group = geometry.groups[group_index]
    return (
        WIRE_TAG,
        WIRE_SCHEMA,
        owner_generation,
        geometry.plan_digest,
        group.group_index,
        group.q_rows,
        group.kv_rows,
        group.sink_rows,
        group.query_position_runs,
    )


def _layout_parts(layout: Any) -> tuple[int, int, int, int, int]:
    signature = getattr(layout, "signature", None)
    if not isinstance(signature, tuple) or len(signature) != 5:
        raise ValueError("VDN query-position preflight requires the native MiniMax-H3 PackedLayout signature")
    text_len, latent_t, lat_h, lat_w, audio_t = signature
    for name, value in zip(("text_len", "latent_t", "lat_h", "lat_w", "audio_t"), signature):
        _strict_int(value, name)
    if latent_t <= 0 or lat_h <= 0 or lat_w <= 0 or lat_h % 2 or lat_w % 2:
        raise ValueError("VDN PackedLayout video signature is invalid")
    segments = tuple(getattr(layout, "segments", ()))
    try:
        video = next(segment for segment in segments if segment[2] == "video")
        audio = next(segment for segment in segments if segment[2] == "audio")
        text = next(segment for segment in segments if segment[2] == "text")
    except (StopIteration, IndexError, TypeError) as exc:
        raise ValueError("VDN PackedLayout is missing text/audio/video segments") from exc
    for segment in (video, audio, text):
        if len(segment) < 3 or type(segment[0]) is not int or type(segment[1]) is not int:
            raise ValueError("VDN PackedLayout segment bounds must be integers")
    video_start, video_end = video[0], video[1]
    if audio[1] != video_start or text[1] - text[0] != text_len:
        raise ValueError("VDN PackedLayout segment ordering does not match the native contract")
    tokens_per_frame = (lat_h // 2) * (lat_w // 2)
    if video_end - video_start != latent_t * tokens_per_frame:
        raise ValueError("VDN PackedLayout video rows do not match its signature")
    seq_len = getattr(layout, "seq_len", None)
    _strict_int(seq_len, "seq_len", minimum=1)
    if video_end > seq_len:
        raise ValueError("VDN PackedLayout video rows exceed sequence length")
    return video_start, video_end, latent_t, tokens_per_frame, seq_len


def native_plan_summary(layout: Any, cfg: dict[str, Any], owner_generation: str) -> QueryPositionPlan:
    """Derive the exact upcoming native/grouped plan from the supplied layout.

    This function intentionally does not read VDNState.layout.  Spectrum calls it
    during preflight, before the actual VDN layout wrapper may install execution
    state for the current evaluation.
    """
    if not isinstance(cfg, dict):
        raise ValueError("VDN query-position preflight requires a configuration dictionary")
    if not isinstance(owner_generation, str) or not owner_generation:
        raise ValueError("VDN query-position owner generation is missing")
    video_start, video_end, num_frames, tokens_per_frame, seq_len = _layout_parts(layout)
    radius = _strict_int(cfg.get("radius"), "radius")
    chunk = _strict_int(cfg.get("chunk"), "chunk")
    anchor_frames = _validate_anchor_mode(cfg.get("anchor_frames"))
    if chunk <= 0:
        raw_bounds = tuple((frame - radius, frame + radius) for frame in range(num_frames))
    else:
        raw_bounds = tuple(
            (
                ((frame // chunk) - radius) * chunk,
                ((frame // chunk) + radius + 1) * chunk - 1,
            )
            for frame in range(num_frames)
        )
    geometry = describe_window_geometry(
        video_start,
        video_end,
        num_frames,
        tokens_per_frame,
        raw_bounds,
        anchor_frames,
        seq_len,
    )
    full_cover = all(lo == 0 and hi == num_frames - 1 for lo, hi in geometry.bounds)
    mode = "native" if full_cover else "grouped"
    maps = () if full_cover else tuple(
        bind_query_map(geometry, index, owner_generation)
        for index in range(len(geometry.groups))
    )
    return QueryPositionPlan(
        tag=PLAN_TAG,
        schema=PLAN_SCHEMA,
        mode=mode,
        owner_generation=owner_generation,
        plan_digest=geometry.plan_digest,
        seq_len=seq_len,
        video_start=video_start,
        video_end=video_end,
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        anchor_frames=anchor_frames,
        groups=maps,
    )


__all__ = [
    "GEOMETRY_SCHEMA",
    "GroupGeometry",
    "PLAN_SCHEMA",
    "PLAN_TAG",
    "QueryPositionPlan",
    "WIRE_SCHEMA",
    "WIRE_TAG",
    "WindowGeometry",
    "bind_query_map",
    "describe_window_geometry",
    "native_plan_summary",
]
