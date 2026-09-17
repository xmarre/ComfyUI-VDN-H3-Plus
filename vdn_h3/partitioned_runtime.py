"""Runtime bridge for Flow partitioned exact-prefix sequences.

The released VDN external-sequence parser knows only the retired Mixed-Grid API
and older reduced-sequence control path.  This bridge adds one explicit API-3
execution mode without reinterpreting either historical contract.  When the new
Flow metadata is absent, VDN executes the released parser byte-for-byte.
"""
from __future__ import annotations

from typing import Any

from .partitioned_sequence import (
    PARTITIONED_PREFIX_KEY,
    PARTITIONED_PREFIX_TOPOLOGY,
    VDN_PARTITIONED_SEQUENCE_API,
    VDN_PARTITIONED_SEQUENCE_MODE,
    make_vdn_partitioned_external_contract,
    validate_flow_partition_contract,
)

VDN_EXTERNAL_SEQUENCE_KEY = "vdn_h3_external_sequence_v1"
_BRIDGE_MARKER = "_vdn_partitioned_exact_prefix_bridge_v1"


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


def install_partitioned_external_sequence_bridge() -> None:
    """Extend VDN's external-sequence predicate with the explicit API-3 mode."""
    from . import hybrid

    current = hybrid._external_reduced_sequence_active
    if getattr(current, _BRIDGE_MARKER, False):
        return

    released = current

    def partitioned_aware(transformer_options, layout, sequence_rows, rope_freqs):
        options = transformer_options or {}
        if options.get(PARTITIONED_PREFIX_KEY) is None:
            return released(options, layout, sequence_rows, rope_freqs)
        validate_partitioned_external_execution(
            options,
            layout,
            sequence_rows,
            rope_freqs,
        )
        return True

    setattr(partitioned_aware, _BRIDGE_MARKER, True)
    partitioned_aware._vdn_released_external_sequence_parser = released
    hybrid._external_reduced_sequence_active = partitioned_aware


__all__ = [
    "VDN_EXTERNAL_SEQUENCE_KEY",
    "install_partitioned_external_sequence_bridge",
    "validate_partitioned_external_execution",
]
