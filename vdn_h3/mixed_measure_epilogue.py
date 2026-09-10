"""VDN-owned epilogue for external Mixed-Grid sparse softmax providers.

The external Mixed-Grid path keeps VDN's released learned softmax gate and the
native attention output projection even when a companion provider computes the
unprojected softmax result directly.  This module owns only that epilogue: it
never performs attention and never invokes VDN's geometry-dependent linear
complement.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import uuid
from typing import Any, Mapping

import torch
import torch.nn.functional as F

EPILOGUE_KEY = "vdn_h3_external_softmax_epilogue_v1"
EXTERNAL_SEQUENCE_KEY = "vdn_h3_external_sequence_v1"
EXTERNAL_SEQUENCE_API = 2
EXTERNAL_SEQUENCE_MODE = "dense_gate_no_linear"
EXTERNAL_TOPOLOGY = "mixed_grid_low_suffix"


def _freeze(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _freeze(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_freeze(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_freeze(v) for v in value), key=repr)
    return repr(value)


def _digest(value: Any) -> str:
    payload = json.dumps(_freeze(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _validate_external_contract(contract, layout, rows: int, rope_freqs) -> dict[str, int]:
    if not isinstance(contract, Mapping):
        raise RuntimeError("VDN Mixed-Grid epilogue requires an external-sequence contract")
    if (
        contract.get("api") != EXTERNAL_SEQUENCE_API
        or contract.get("mode") != EXTERNAL_SEQUENCE_MODE
        or contract.get("topology") != EXTERNAL_TOPOLOGY
    ):
        raise RuntimeError("VDN Mixed-Grid epilogue received an unsupported external-sequence contract")
    names = (
        "native_sequence_rows",
        "sequence_rows",
        "video_start",
        "temporal",
        "prefix_t",
        "source_rows_per_frame",
        "prefix_rows_per_frame",
    )
    if any(type(contract.get(name)) is not int for name in names):
        raise RuntimeError("VDN Mixed-Grid epilogue row counts must be integers")
    native, actual, start, temporal, prefix_t, source_rows, prefix_rows = (contract[name] for name in names)
    if (
        actual != rows
        or native != getattr(layout, "seq_len", None)
        or start != getattr(layout, "video_start", None)
        or temporal != getattr(layout, "num_frames", None)
        or source_rows != getattr(layout, "tokens_per_frame", None)
        or not 0 < prefix_t < temporal
        or not 0 < source_rows < prefix_rows
        or native != start + temporal * source_rows
        or actual != start + prefix_t * prefix_rows + (temporal - prefix_t) * source_rows
    ):
        raise RuntimeError("VDN Mixed-Grid epilogue contract does not match the active VDN layout")
    if rope_freqs is None or rope_freqs.ndim < 2 or int(rope_freqs.shape[1]) != rows:
        raise RuntimeError("VDN Mixed-Grid epilogue requires explicit RoPE rows matching the mixed stream")
    return {name: int(contract[name]) for name in names}


@dataclass
class BoundVDNEpilogue:
    state: object
    block_index: int
    out_proj: object
    rows: int
    heads: int
    head_dim: int
    owner_generation: str
    config_digest: str
    weight_owner_digest: str
    external_digest: str
    gate_expected: bool
    projection_calls: int = 0
    gate_calls: int = 0
    completed: bool = False

    def apply(self, softmax_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply VDN's gate, then the original output projection, exactly once."""
        if self.completed or self.projection_calls:
            raise RuntimeError("VDN Mixed-Grid epilogue cannot be applied more than once")
        if (
            not torch.is_tensor(softmax_out)
            or softmax_out.ndim != 3
            or tuple(softmax_out.shape) != (self.rows, self.heads, self.head_dim)
        ):
            raise RuntimeError("VDN Mixed-Grid epilogue received an incompatible softmax tensor")
        if not torch.is_tensor(x) or x.ndim != 2 or x.shape[0] != self.rows:
            raise RuntimeError("VDN Mixed-Grid epilogue received an incompatible hidden-state tensor")
        if softmax_out.device != x.device:
            raise RuntimeError("VDN Mixed-Grid epilogue softmax and hidden state must share a device")

        branch = self.state.branches[self.block_index]
        weights = None
        if branch is not None:
            weights = self.state.weights_on(self.block_index, x.device, x.dtype)

        if self.gate_expected:
            if branch is None or weights is None:
                raise RuntimeError("VDN Mixed-Grid epilogue expected a learned gate but has no branch weights")
            try:
                gate_weight = weights["softmax_gate.up.weight"]
                gate_bias = weights["softmax_gate.up.bias"]
            except (KeyError, TypeError) as exc:
                raise RuntimeError("VDN Mixed-Grid epilogue is missing learned softmax-gate weights") from exc
            gate = torch.sigmoid(F.linear(x, gate_weight, gate_bias))
            if gate.ndim != 2 or tuple(gate.shape) != (self.rows, self.heads):
                raise RuntimeError("VDN Mixed-Grid softmax gate returned incompatible geometry")
            flat = (softmax_out * gate.view(self.rows, self.heads, 1).to(softmax_out.dtype)).reshape(self.rows, -1)
            self.gate_calls += 1
        else:
            flat = softmax_out.reshape(self.rows, -1)

        result = self.out_proj(flat.type_as(x))
        self.projection_calls += 1
        if not torch.is_tensor(result) or result.ndim != 2 or result.shape[0] != self.rows:
            raise RuntimeError("VDN Mixed-Grid output projection returned incompatible geometry")
        self.completed = True
        return result

    def receipt_fields(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("vdn_owner_generation", self.owner_generation),
            ("vdn_config_digest", self.config_digest),
            ("vdn_weight_owner_digest", self.weight_owner_digest),
            ("vdn_external_digest", self.external_digest),
            ("vdn_gate_expected", self.gate_expected),
            ("vdn_gate_calls", self.gate_calls),
            ("vdn_projection_calls", self.projection_calls),
            ("vdn_completed", self.completed),
        )


class ExternalSoftmaxEpilogueCapability:
    """Capability attached to one VDN-patched MiniMax-H3 attention owner."""

    api = 1

    def __init__(self, state, block_index: int, out_proj, heads: int, head_dim: int):
        if type(block_index) is not int or block_index < 0:
            raise ValueError("VDN epilogue block index must be a nonnegative integer")
        self.state = state
        self.block_index = block_index
        self.out_proj = out_proj
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.owner_generation = f"vdn-epilogue-{uuid.uuid4().hex}"
        branch = state.branches[block_index]
        self.config_digest = _digest({
            "name": getattr(state, "name", None),
            "cfg": getattr(state, "cfg", None),
            "block_index": block_index,
            "branch_present": branch is not None,
            "branch_type": None if branch is None else f"{type(branch).__module__}.{type(branch).__qualname__}",
        })
        weight_owner = getattr(state, "managed_weights", None) or branch
        self.weight_owner_digest = _digest({
            "owner_type": None if weight_owner is None else f"{type(weight_owner).__module__}.{type(weight_owner).__qualname__}",
            "block_index": block_index,
            "branch_present": branch is not None,
        })

    def prepare(self, x, rope_freqs, options, block_index):
        if type(block_index) is not int or block_index != self.block_index:
            raise RuntimeError("VDN Mixed-Grid epilogue was requested for the wrong block owner")
        if not torch.is_tensor(x) or x.ndim != 2:
            raise RuntimeError("VDN Mixed-Grid epilogue requires [T,D] hidden states")
        layout = self.state.layout
        if layout is None:
            raise RuntimeError("VDN Mixed-Grid epilogue called outside the VDN execution lifetime")
        contract = (options or {}).get(EXTERNAL_SEQUENCE_KEY)
        normalized = _validate_external_contract(contract, layout, int(x.shape[0]), rope_freqs)
        branch = self.state.branches[block_index]
        gate_expected = bool(branch is not None and self.state.cfg.get("enable_softmax_gate", True))
        return BoundVDNEpilogue(
            state=self.state,
            block_index=block_index,
            out_proj=self.out_proj,
            rows=int(x.shape[0]),
            heads=self.heads,
            head_dim=self.head_dim,
            owner_generation=self.owner_generation,
            config_digest=self.config_digest,
            weight_owner_digest=self.weight_owner_digest,
            external_digest=_digest(normalized),
            gate_expected=gate_expected,
        )


__all__ = ["EPILOGUE_KEY", "BoundVDNEpilogue", "ExternalSoftmaxEpilogueCapability"]
