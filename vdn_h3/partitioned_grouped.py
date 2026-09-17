"""Pure heterogeneous grouped geometry for partitioned exact-prefix VDN attention.

Unlike the retired external Mixed-Grid mode, this plan preserves VDN's temporal
window/global/anchor ownership. Prefix frames retain target-grid row counts while
generated suffix frames retain source-grid row counts. The same immutable plan
builds the real gathers and the Sol mapped-neighbor wire maps.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .query_positions import WIRE_SCHEMA, WIRE_TAG
from .partitioned_sequence import PartitionedSequence

PARTITIONED_GROUPED_SCHEMA = "vdn-partitioned-grouped-geometry-v1"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _anchor_rows(anchor_frames: str, temporal: int) -> tuple[int, ...]:
    if anchor_frames not in {"none", "columns", "rows", "both"}:
        raise ValueError("partitioned VDN anchor mode is unsupported")
    if anchor_frames not in {"rows", "both"}:
        return ()
    return tuple(sorted({0, temporal - 1}))


def _frame_rows(plan: PartitionedSequence, frame: int) -> int:
    return plan.target_rows if frame < plan.prefix_t else plan.source_rows


def _frame_ranges(plan: PartitionedSequence) -> tuple[tuple[int, int], ...]:
    cursor = plan.video_start
    ranges = []
    for frame in range(plan.temporal):
        rows = _frame_rows(plan, frame)
        ranges.append((cursor, cursor + rows))
        cursor += rows
    if cursor != plan.sequence_rows:
        raise RuntimeError("partitioned VDN frame ranges do not cover the mixed sequence")
    return tuple(ranges)


@dataclass(frozen=True, slots=True)
class PartitionedGroup:
    group_index: int
    query_frames: tuple[int, ...]
    key_frames: tuple[int, ...]
    query_prefix_domain: bool
    q_ranges: tuple[tuple[int, int], ...]
    key_ranges: tuple[tuple[int, int], ...]
    q_rows: int
    kv_rows: int
    sink_rows: int
    prefix_k_range: tuple[int, int] | None
    query_position_runs: tuple[tuple[int, int, int], ...]

    def wire(self, *, owner_generation: str, plan_digest: str) -> tuple[Any, ...]:
        if not isinstance(owner_generation, str) or not owner_generation:
            raise ValueError("partitioned VDN query-position owner generation is missing")
        return (
            WIRE_TAG,
            WIRE_SCHEMA,
            owner_generation,
            plan_digest,
            self.group_index,
            self.q_rows,
            self.kv_rows,
            self.sink_rows,
            self.query_position_runs,
        )


@dataclass(frozen=True, slots=True)
class PartitionedGroupedPlan:
    schema: str
    semantic_digest: str
    sequence_rows: int
    video_start: int
    temporal: int
    prefix_t: int
    source_rows: int
    target_rows: int
    bounds: tuple[tuple[int, int], ...]
    anchor_frames: str
    frame_ranges: tuple[tuple[int, int], ...]
    anchor_slices: tuple[tuple[int, int], ...]
    groups: tuple[PartitionedGroup, ...]
    plan_digest: str

    @property
    def full_prefix_k_range(self) -> tuple[int, int]:
        return self.video_start, self.video_start + self.prefix_t * self.target_rows

    def wires(self, owner_generation: str) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            group.wire(owner_generation=owner_generation, plan_digest=self.plan_digest)
            for group in self.groups
        )


def build_partitioned_grouped_plan(
    plan: PartitionedSequence,
    *,
    bounds,
    anchor_frames: str,
    semantic_digest: str,
) -> PartitionedGroupedPlan:
    if not isinstance(plan, PartitionedSequence):
        raise TypeError("partitioned VDN grouped geometry requires a validated Flow plan")
    if not isinstance(semantic_digest, str) or len(semantic_digest) != 64:
        raise ValueError("partitioned VDN semantic digest is invalid")
    normalized_bounds = tuple((int(lo), int(hi)) for lo, hi in bounds)
    if len(normalized_bounds) != plan.temporal:
        raise ValueError("partitioned VDN temporal bounds do not cover every frame")
    for frame, (lo, hi) in enumerate(normalized_bounds):
        if not 0 <= lo <= frame <= hi < plan.temporal:
            raise ValueError("partitioned VDN temporal bound is invalid")

    ranges = _frame_ranges(plan)
    row_anchors = _anchor_rows(anchor_frames, plan.temporal)
    anchor_set = set(row_anchors)
    grouped: dict[tuple[tuple[int, int], bool], list[int]] = {}
    for frame, bound in enumerate(normalized_bounds):
        if frame in anchor_set:
            continue
        key = (bound, frame < plan.prefix_t)
        grouped.setdefault(key, []).append(frame)

    groups = []
    column_anchors = tuple(sorted({0, plan.temporal - 1})) if anchor_frames in {"columns", "both"} else ()
    for group_index, ((lo, hi), query_prefix_domain) in enumerate(grouped.items()):
        query_frames = tuple(grouped[((lo, hi), query_prefix_domain)])
        key_frames = tuple(sorted(set(range(lo, hi + 1)) | {f for f in column_anchors if not lo <= f <= hi}))
        if any(frame not in key_frames for frame in query_frames):
            raise RuntimeError("partitioned VDN query frame is absent from its K/V domain")

        q_ranges = tuple(ranges[frame] for frame in query_frames)
        key_ranges = tuple(ranges[frame] for frame in key_frames)
        q_rows = sum(end - start for start, end in q_ranges)
        sink_rows = plan.video_start
        frame_k_offsets = {}
        cursor = sink_rows
        prefix_k_start = None
        prefix_k_end = None
        for frame, (start, end) in zip(key_frames, key_ranges, strict=True):
            del start
            frame_k_offsets[frame] = cursor
            rows = end - ranges[frame][0]
            if frame < plan.prefix_t:
                if prefix_k_start is None:
                    prefix_k_start = cursor
                prefix_k_end = cursor + rows
            cursor += rows
        kv_rows = cursor
        prefix_k_range = (
            None if prefix_k_start is None else (int(prefix_k_start), int(prefix_k_end))
        )

        runs = []
        q_cursor = 0
        previous_k = -1
        for frame, (start, end) in zip(query_frames, q_ranges, strict=True):
            del start
            rows = end - ranges[frame][0]
            k_begin = frame_k_offsets[frame]
            if k_begin <= previous_k:
                raise RuntimeError("partitioned VDN query-position map is not monotonic")
            runs.append((q_cursor, q_cursor + rows, k_begin))
            q_cursor += rows
            previous_k = k_begin + rows - 1
        if q_cursor != q_rows:
            raise RuntimeError("partitioned VDN query-position runs do not cover Q")

        groups.append(
            PartitionedGroup(
                group_index=group_index,
                query_frames=query_frames,
                key_frames=key_frames,
                query_prefix_domain=bool(query_prefix_domain),
                q_ranges=q_ranges,
                key_ranges=key_ranges,
                q_rows=q_rows,
                kv_rows=kv_rows,
                sink_rows=sink_rows,
                prefix_k_range=prefix_k_range,
                query_position_runs=tuple(runs),
            )
        )

    payload = {
        "schema": PARTITIONED_GROUPED_SCHEMA,
        "semantic_digest": semantic_digest,
        "sequence_rows": plan.sequence_rows,
        "video_start": plan.video_start,
        "temporal": plan.temporal,
        "prefix_t": plan.prefix_t,
        "source_rows": plan.source_rows,
        "target_rows": plan.target_rows,
        "bounds": normalized_bounds,
        "anchor_frames": anchor_frames,
        "groups": [
            {
                "group_index": group.group_index,
                "query_frames": group.query_frames,
                "key_frames": group.key_frames,
                "query_prefix_domain": group.query_prefix_domain,
                "q_rows": group.q_rows,
                "kv_rows": group.kv_rows,
                "sink_rows": group.sink_rows,
                "prefix_k_range": group.prefix_k_range,
                "runs": group.query_position_runs,
            }
            for group in groups
        ],
    }
    return PartitionedGroupedPlan(
        schema=PARTITIONED_GROUPED_SCHEMA,
        semantic_digest=semantic_digest,
        sequence_rows=plan.sequence_rows,
        video_start=plan.video_start,
        temporal=plan.temporal,
        prefix_t=plan.prefix_t,
        source_rows=plan.source_rows,
        target_rows=plan.target_rows,
        bounds=normalized_bounds,
        anchor_frames=anchor_frames,
        frame_ranges=ranges,
        anchor_slices=tuple(ranges[frame] for frame in row_anchors),
        groups=tuple(groups),
        plan_digest=_sha256_json(payload),
    )


__all__ = [
    "PARTITIONED_GROUPED_SCHEMA",
    "PartitionedGroup",
    "PartitionedGroupedPlan",
    "build_partitioned_grouped_plan",
]
