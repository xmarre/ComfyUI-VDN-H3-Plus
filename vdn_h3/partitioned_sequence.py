"""Fail-closed parser for Flow's partitioned exact-prefix sequence contract.

This is deliberately separate from the deprecated Mixed-Grid external-sequence
API. VDN remains owner of its learned gate/out projection and variable-grid linear
complement; the contract only describes the heterogeneous packed rows presented to
VDN during the low/probe phase of exact-prefix progressive continuation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

PARTITIONED_PREFIX_KEY = "h3_flow_partitioned_exact_prefix_v1"
PARTITIONED_PREFIX_API = 1
PARTITIONED_PREFIX_TOPOLOGY = "target_prefix_source_suffix"
VDN_PARTITIONED_SEQUENCE_API = 4
VDN_PARTITIONED_SEQUENCE_MODE = "partitioned_attention_variable_grid_linear"
VDN_TEMPORAL_CARRIER_KEY = "h3_flow_partitioned_vdn_temporal_carrier_v1"
VDN_TEMPORAL_CARRIER_API = 1
VDN_TEMPORAL_CARRIER_NATIVE = "native_grid_then_map_v1"
VDN_TEMPORAL_CARRIER_DESTINATION = "destination_grid_stencil_v1"
VDN_TEMPORAL_CARRIER_POLICIES = (
    VDN_TEMPORAL_CARRIER_NATIVE,
    VDN_TEMPORAL_CARRIER_DESTINATION,
)
VDN_TEMPORAL_CARRIER_MAPPING_POLICY = "h3_physical_bilinear_border_fp32_restore_dtype_v1"


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _digest(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PartitionedSequence:
    video_start: int
    temporal: int
    prefix_t: int
    source_grid_h: int
    source_grid_w: int
    target_grid_h: int
    target_grid_w: int

    def __post_init__(self):
        for name in (
            "video_start",
            "temporal",
            "prefix_t",
            "source_grid_h",
            "source_grid_w",
            "target_grid_h",
            "target_grid_w",
        ):
            _positive_int(getattr(self, name), name)
        if self.prefix_t >= self.temporal:
            raise ValueError("partitioned sequence requires a generated suffix")
        if self.source_grid_h > self.target_grid_h or self.source_grid_w > self.target_grid_w:
            raise ValueError("partitioned source grid exceeds target grid")
        if self.source_rows > self.target_rows:
            raise ValueError("partitioned source grid must not exceed target grid")

    @property
    def source_rows(self):
        return self.source_grid_h * self.source_grid_w

    @property
    def target_rows(self):
        return self.target_grid_h * self.target_grid_w

    @property
    def prefix_rows(self):
        return self.prefix_t * self.target_rows

    @property
    def suffix_rows(self):
        return (self.temporal - self.prefix_t) * self.source_rows

    @property
    def sequence_rows(self):
        return self.video_start + self.prefix_rows + self.suffix_rows

    @property
    def prefix_range(self):
        return self.video_start, self.video_start + self.prefix_rows

    @property
    def suffix_range(self):
        return self.prefix_range[1], self.sequence_rows

    @property
    def prefix_log_key_measure(self):
        return math.log(self.source_rows / self.target_rows)

    def canonical_contract(self):
        payload = {
            "api": PARTITIONED_PREFIX_API,
            "topology": PARTITIONED_PREFIX_TOPOLOGY,
            "sequence_rows": self.sequence_rows,
            "video_start": self.video_start,
            "temporal": self.temporal,
            "prefix_t": self.prefix_t,
            "source_grid_h": self.source_grid_h,
            "source_grid_w": self.source_grid_w,
            "target_grid_h": self.target_grid_h,
            "target_grid_w": self.target_grid_w,
            "source_rows_per_frame": self.source_rows,
            "target_rows_per_frame": self.target_rows,
            "prefix_range": list(self.prefix_range),
            "suffix_range": list(self.suffix_range),
            "prefix_log_key_measure": self.prefix_log_key_measure,
            "nonvideo_log_key_measure": 0.0,
            "suffix_log_key_measure": 0.0,
            "exact_prefix_queries_preserved": True,
            "generated_suffix_queries_preserved": True,
            "heterogeneous_spatial_domains": True,
        }
        payload["semantic_digest"] = _digest(payload)
        return payload


def make_temporal_carrier_contract(
    *,
    policy: str,
    flow_semantic_digest: str,
    diagnostic_mode: str,
    short_conv_spec: str,
):
    """Build the deterministic Flow-to-VDN numerical-policy leaf."""
    policy = str(policy)
    if policy not in VDN_TEMPORAL_CARRIER_POLICIES:
        raise ValueError(f"unsupported VDN temporal-carrier policy {policy!r}")
    if not isinstance(flow_semantic_digest, str) or len(flow_semantic_digest) != 64:
        raise ValueError("VDN temporal-carrier policy requires a Flow semantic digest")
    if not isinstance(diagnostic_mode, str) or not diagnostic_mode:
        raise ValueError("VDN temporal-carrier policy requires a diagnostic mode")
    if not isinstance(short_conv_spec, str) or not short_conv_spec:
        raise ValueError("VDN temporal-carrier policy requires a checkpoint short-conv specification")
    payload = {
        "api": VDN_TEMPORAL_CARRIER_API,
        "policy": policy,
        "flow_semantic_digest": flow_semantic_digest,
        "diagnostic_mode": diagnostic_mode,
        "short_conv_spec": short_conv_spec,
        "precision_mapping_policy": VDN_TEMPORAL_CARRIER_MAPPING_POLICY,
    }
    payload["numerical_digest"] = _digest(payload)
    return payload


def validate_temporal_carrier_contract(
    contract,
    *,
    flow_semantic_digest: str,
    diagnostic_mode: str,
    short_conv_spec: str,
):
    """Validate an explicitly selected carrier policy; absence is legacy/native."""
    if contract is None:
        return VDN_TEMPORAL_CARRIER_NATIVE, None
    if not isinstance(contract, dict):
        raise ValueError("VDN temporal-carrier policy must be a dictionary")
    policy = contract.get("policy")
    expected = make_temporal_carrier_contract(
        policy=policy,
        flow_semantic_digest=flow_semantic_digest,
        diagnostic_mode=diagnostic_mode,
        short_conv_spec=short_conv_spec,
    )
    if contract != expected:
        raise ValueError("VDN temporal-carrier numerical policy does not match the active execution")
    return str(policy), expected


def validate_flow_partition_contract(contract, *, sequence_rows):
    if not isinstance(contract, dict):
        raise ValueError("partitioned exact-prefix metadata must be a dictionary")
    if (
        contract.get("api") != PARTITIONED_PREFIX_API
        or contract.get("topology") != PARTITIONED_PREFIX_TOPOLOGY
    ):
        raise ValueError("unsupported partitioned exact-prefix metadata")
    geometry_names = (
        "video_start",
        "temporal",
        "prefix_t",
        "source_grid_h",
        "source_grid_w",
        "target_grid_h",
        "target_grid_w",
    )
    if any(type(contract.get(name)) is not int for name in geometry_names):
        raise ValueError("partitioned exact-prefix geometry must use integer fields")
    parsed = PartitionedSequence(**{name: contract[name] for name in geometry_names})
    canonical = parsed.canonical_contract()
    for name, expected in canonical.items():
        if contract.get(name) != expected:
            raise ValueError(f"partitioned exact-prefix field {name!r} is inconsistent")
    if parsed.sequence_rows != int(sequence_rows):
        raise ValueError("partitioned exact-prefix rows do not match the current hidden sequence")
    return parsed


def make_vdn_partitioned_external_contract(plan: PartitionedSequence):
    """Bind validated Flow geometry to VDN's explicit external execution mode."""
    if not isinstance(plan, PartitionedSequence):
        raise TypeError("VDN partitioned external contract requires a validated plan")
    return {
        "api": VDN_PARTITIONED_SEQUENCE_API,
        "mode": VDN_PARTITIONED_SEQUENCE_MODE,
        "topology": PARTITIONED_PREFIX_TOPOLOGY,
        "sequence_rows": plan.sequence_rows,
        "video_start": plan.video_start,
        "temporal": plan.temporal,
        "prefix_t": plan.prefix_t,
        "source_rows_per_frame": plan.source_rows,
        "target_rows_per_frame": plan.target_rows,
        "flow_semantic_digest": plan.canonical_contract()["semantic_digest"],
    }


__all__ = [
    "PARTITIONED_PREFIX_API",
    "PARTITIONED_PREFIX_KEY",
    "PARTITIONED_PREFIX_TOPOLOGY",
    "VDN_PARTITIONED_SEQUENCE_API",
    "VDN_PARTITIONED_SEQUENCE_MODE",
    "VDN_TEMPORAL_CARRIER_API",
    "VDN_TEMPORAL_CARRIER_DESTINATION",
    "VDN_TEMPORAL_CARRIER_KEY",
    "VDN_TEMPORAL_CARRIER_MAPPING_POLICY",
    "VDN_TEMPORAL_CARRIER_NATIVE",
    "VDN_TEMPORAL_CARRIER_POLICIES",
    "PartitionedSequence",
    "make_temporal_carrier_contract",
    "make_vdn_partitioned_external_contract",
    "validate_flow_partition_contract",
    "validate_temporal_carrier_contract",
]
