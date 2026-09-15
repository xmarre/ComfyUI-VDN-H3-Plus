"""VDN-owned restricted-domain dispatch for bounded first-high Sol-local E.

The overlay is inert without ``h3_first_high_sol_local_diagnostic_v1``.  Under E
it reproduces the ordinary grouped-window gather and leaves the released linear
complement enabled.  The only numerical intervention is owned by the Sol
companion after VDN has supplied the already-restricted local Q/K/V domain.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import torch

from .first_high_sol_local_bridge import enter_local_group, exit_local_group, parse_sol_request

REQUEST_KEY = "h3_first_high_sol_local_diagnostic_v1"
RECEIPTS_KEY = "h3_first_high_sol_local_receipts_v1"
EVIDENCE_KEY = "h3_first_high_sol_local_evidence_v1"
_BLOCK_KEY = "h3_first_high_sol_local_vdn_block_v1"
_GATE_FP_KEY = "h3_first_high_sol_local_vdn_gate_fingerprint_v1"
_ADAPTER_FP_KEY = "h3_first_high_sol_local_vdn_adapter_fingerprint_v1"
_WITNESS_GROUPS = frozenset({0, 2, 10})
_EXPECTED_LOCAL_Q_ROWS = (4096,) + (5120,) * 9 + (1024,)
_EXPECTED_WINDOW_KV_ROWS = (14365, 19485) + (20509,) * 7 + (16413, 11293)
_REQUIRED_FIELDS = (
    "api",
    "capture_id",
    "mode",
    "stage",
    "logical_call_limit",
    "sigma",
    "target_shapes_digest",
    "source_contract_digest",
)
_HEX = frozenset("0123456789abcdef")
_INSTALLED = False
_ORIGINAL_MAKE_VDN_FORWARD = None
_ORIGINAL_WINDOW_SOFTMAX = None


def _request_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != len(_REQUIRED_FIELDS):
        raise RuntimeError("first-high Sol-local E request must be an immutable exact-field tuple")
    result: dict[str, Any] = {}
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str):
            raise RuntimeError("first-high Sol-local E request entries must be (name, value) tuples")
        key, field_value = item
        if key in result:
            raise RuntimeError(f"duplicate first-high Sol-local E field: {key}")
        result[key] = field_value
    if tuple(result) != _REQUIRED_FIELDS or result["api"] != 1:
        raise RuntimeError("first-high Sol-local E request does not match API 1")
    if result["mode"] != "all_selected_e":
        raise RuntimeError(f"unsupported first-high Sol-local E mode: {result['mode']!r}")
    if result["stage"] != "high" or result["logical_call_limit"] != 1:
        raise RuntimeError("first-high Sol-local E is restricted to one high-stage logical call")
    if not isinstance(result["capture_id"], str) or not result["capture_id"]:
        raise RuntimeError("first-high Sol-local E capture_id is invalid")
    sigma = result["sigma"]
    if type(sigma) is not float or not math.isfinite(sigma) or not 0.0 < sigma <= 1.0:
        raise RuntimeError("first-high Sol-local E sigma must be a finite float in (0, 1]")
    for name in ("target_shapes_digest", "source_contract_digest"):
        digest = result[name]
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in _HEX for ch in digest):
            raise RuntimeError(f"first-high Sol-local E {name} must be lowercase SHA-256")
    return result


def parse_request(options: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return _request_dict((options or {}).get(REQUEST_KEY))


def _freeze(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _freeze(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_freeze(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_freeze(v) for v in value), key=repr)
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _digest(value: Any) -> str:
    payload = json.dumps(_freeze(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _gate_fingerprint(state: Any) -> str:
    cfg = getattr(state, "cfg", {}) or {}
    return _digest(
        {
            "enable_softmax_gate": cfg.get("enable_softmax_gate", True),
            "global_gate_mode": cfg.get("global_gate_mode", "checkpoint"),
            "conditioning_video_context_strength": cfg.get("conditioning_video_context_strength", 1.0),
            "audio_video_context_strength": cfg.get("audio_video_context_strength", 1.0),
        }
    )


def _adapter_fingerprint(state: Any, block_index: int) -> str:
    branch = state.branches[block_index]
    managed = getattr(state, "managed_weights", None)
    return _digest(
        {
            "state_name": getattr(state, "name", None),
            "block_index": block_index,
            "branch_type": None if branch is None else f"{type(branch).__module__}.{type(branch).__qualname__}",
            "managed_weights_type": None if managed is None else f"{type(managed).__module__}.{type(managed).__qualname__}",
            "linear_enabled": (getattr(state, "cfg", {}) or {}).get("linear_enabled", True),
            "enable_text_state": None if branch is None else bool(getattr(branch, "enable_text_state", False)),
            "delta_rule": None if branch is None else getattr(branch, "delta_rule", None),
            "bridge": None if branch is None else getattr(branch, "bridge", None),
        }
    )


def _sampled_tensor_digest(*values: torch.Tensor, max_values: int = 128) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(f"{tuple(value.shape)}|{value.dtype}|{value.device.type}".encode())
        flat = value.detach().reshape(-1)
        if flat.numel() == 0:
            continue
        count = min(int(max_values), int(flat.numel()))
        if count == 1:
            index = torch.zeros(1, dtype=torch.long, device=flat.device)
        else:
            ordinal = torch.arange(count, dtype=torch.long, device=flat.device)
            index = torch.div(ordinal * (int(flat.numel()) - 1), count - 1, rounding_mode="floor")
        sampled = flat.index_select(0, index).contiguous().view(torch.uint8).to(device="cpu")
        digest.update(memoryview(sampled.numpy()))
    return digest.hexdigest()


def _append_receipt(options: dict[str, Any], **fields: Any) -> None:
    sink = options.get(RECEIPTS_KEY)
    append = getattr(sink, "append", None)
    if not callable(append):
        raise RuntimeError("first-high Sol-local E receipt sink is missing or invalid")
    try:
        count = len(sink)
    except TypeError as exc:
        raise RuntimeError("first-high Sol-local E receipt sink is not bounded") from exc
    if count >= 800:
        raise RuntimeError("first-high Sol-local E receipt bound exceeded")
    append(fields)


def _tensor_contract(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": [int(dim) for dim in value.shape],
        "stride": [int(item) for item in value.stride()],
        "storage_offset": int(value.storage_offset()),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def _preserve_witness(
    options: dict[str, Any],
    *,
    block_index: int,
    group_index: int,
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    q_idx: torch.Tensor,
    domain_idx: torch.Tensor,
    query_positions: torch.Tensor,
    scale: float,
    sink_rows: int,
) -> dict[str, Any]:
    sink = options.get(EVIDENCE_KEY)
    append = getattr(sink, "append", None)
    items = getattr(sink, "items", None)
    if not callable(append) or not isinstance(items, list):
        raise RuntimeError("first-high Sol-local E evidence sink is missing or invalid")
    if any(
        isinstance(item, dict)
        and item.get("kind") == "operator_witness"
        and item.get("block_index") == block_index
        and item.get("group_index") == group_index
        for item in items
    ):
        raise RuntimeError("first-high Sol-local E witness was preserved more than once")
    record: dict[str, Any] = {
        "kind": "operator_witness",
        "block_index": int(block_index),
        "group_index": int(group_index),
        "scale": float(scale),
        "original_sink_rows": int(sink_rows),
        "q_contract": _tensor_contract(q_rows),
        "k_contract": _tensor_contract(k_rows),
        "v_contract": _tensor_contract(v_rows),
        "q_row_indices": q_idx.detach().to(device="cpu", copy=True),
        "kv_row_indices": domain_idx.detach().to(device="cpu", copy=True),
        "query_positions_in_restricted_kv": query_positions.detach().to(device="cpu", copy=True),
        "q": q_rows.detach().to(device="cpu", copy=True),
        "k": k_rows.detach().to(device="cpu", copy=True),
        "v": v_rows.detach().to(device="cpu", copy=True),
        # Private execution-only pointer. Flow strips private fields from durable
        # evidence; it lets the Sol sidecar recover the clone-stable owner without
        # publishing mutable model options as evidence.
        "_transformer_options": options,
        "completed": False,
    }
    append(record)
    return record


def _diagnostic_window_softmax(
    query,
    key,
    value,
    video_start,
    video_end,
    num_frames,
    tokens_per_frame,
    bounds,
    scale,
    anchor_frames="none",
    transformer_options=None,
):
    from vdn_h3 import retained as retained
    from vdn_h3 import window as W
    from vdn_h3.runtime import current_runtime_buffers
    from vdn_h3.softmax_provider import dispatch, preprocess

    options = transformer_options or {}
    request = parse_request(options)
    if request is None:
        return _ORIGINAL_WINDOW_SOFTMAX(
            query,
            key,
            value,
            video_start,
            video_end,
            num_frames,
            tokens_per_frame,
            bounds,
            scale,
            anchor_frames=anchor_frames,
            transformer_options=transformer_options,
        )
    # Require the Sol companion to parse the same live request before any gather.
    sol_request = parse_sol_request(options)
    if sol_request != request:
        raise RuntimeError("first-high Sol-local E request differs between VDN and Sol companions")
    if options.get("vdn_h3_external_sequence_v1") is not None:
        raise RuntimeError("first-high Sol-local E forbids external/reduced VDN sequence")
    if options.get("attention_measure_v1") is not None or options.get("h3_flow_mixed_grid_attention_measure_v1") is not None:
        raise RuntimeError("first-high Sol-local E forbids weighted/Mixed-Grid attention")
    block_index = options.get(_BLOCK_KEY)
    gate_fingerprint = options.get(_GATE_FP_KEY)
    adapter_fingerprint = options.get(_ADAPTER_FP_KEY)
    if type(block_index) is not int or not 0 <= block_index < 50:
        raise RuntimeError("first-high Sol-local E VDN block identity is missing")
    if not isinstance(gate_fingerprint, str) or not isinstance(adapter_fingerprint, str):
        raise RuntimeError("first-high Sol-local E VDN fingerprints are missing")

    query, key, value = preprocess(options, query, key, value, query.shape[1])
    heads, head_dim = query.shape[1], query.shape[2]
    seq = int(query.shape[0])
    resources = current_runtime_buffers()
    plan_key = (
        video_start,
        video_end,
        num_frames,
        tokens_per_frame,
        tuple(map(tuple, bounds)),
        anchor_frames,
        seq,
        str(query.device),
    )
    builder = lambda: retained._build_window_plan(
        video_start,
        video_end,
        num_frames,
        tokens_per_frame,
        bounds,
        anchor_frames,
        seq,
        query.device,
    )
    plan = resources.window_plan(plan_key, builder) if resources is not None else builder()
    pre_attention_qkv_digest = _sampled_tensor_digest(query, key, value) if block_index == 0 else None

    out = torch.empty_like(query)
    global_idx = plan["global_idx"]
    global_count = int(global_idx.numel())

    def native_with_provider(q, k, v, kind, *, aligned=False, sink_rows=0):
        return dispatch(
            options,
            lambda: W._sdpa(q, k, v, scale, None),
            q,
            k,
            v,
            kind=kind,
            scale=scale,
            square_aligned=aligned,
            sink_rows=sink_rows,
        )

    if global_count:
        qg = query.index_select(0, global_idx)
        out[global_idx] = native_with_provider(qg, key, value, "global")
        _append_receipt(
            options,
            block=block_index,
            kind="global",
            group_index=None,
            q_rows=int(qg.shape[0]),
            kv_rows=int(key.shape[0]),
            packed_rows=seq,
            video_start=int(video_start),
            video_end=int(video_end),
            support_mode="full",
            complement_executed=False,
            provider_route="vdn_global_native",
            gate_fingerprint=gate_fingerprint,
            adapter_fingerprint=adapter_fingerprint,
            pre_attention_qkv_digest=pre_attention_qkv_digest,
        )

    groups = plan["groups"]
    if len(groups) != 11:
        raise RuntimeError(f"first-high Sol-local E expected 11 local groups, got {len(groups)}")
    if groups:
        if resources is None:
            shape = (plan["max_kv_rows"], heads, head_dim)
            k_scratch = torch.empty(shape, device=key.device, dtype=key.dtype)
            v_scratch = torch.empty(shape, device=value.device, dtype=value.dtype)
        else:
            k_scratch, v_scratch = resources.kv_scratch(
                plan["max_kv_rows"], heads, head_dim, key.device, key.dtype
            )
        if global_count:
            torch.index_select(key, 0, global_idx, out=k_scratch[:global_count])
            torch.index_select(value, 0, global_idx, out=v_scratch[:global_count])

        for group_index, (q_idx, win_idx) in enumerate(groups):
            window_rows = int(win_idx.numel())
            domain_rows = global_count + window_rows
            torch.index_select(key, 0, win_idx, out=k_scratch[global_count:domain_rows])
            torch.index_select(value, 0, win_idx, out=v_scratch[global_count:domain_rows])
            q_rows = query.index_select(0, q_idx)
            k_rows = k_scratch[:domain_rows]
            v_rows = v_scratch[:domain_rows]
            if int(q_rows.shape[0]) != _EXPECTED_LOCAL_Q_ROWS[group_index]:
                raise RuntimeError(
                    f"first-high Sol-local E local Q geometry changed for group {group_index}: {q_rows.shape[0]}"
                )
            if domain_rows != _EXPECTED_WINDOW_KV_ROWS[group_index]:
                raise RuntimeError(
                    f"first-high Sol-local E restricted K/V geometry changed for group {group_index}: {domain_rows}"
                )
            domain_idx = torch.cat((global_idx, win_idx)) if global_count else win_idx
            query_positions = torch.searchsorted(win_idx, q_idx) + global_count
            witness_record = None
            if block_index == 2 and group_index in _WITNESS_GROUPS:
                witness_record = _preserve_witness(
                    options,
                    block_index=block_index,
                    group_index=group_index,
                    q_rows=q_rows,
                    k_rows=k_rows,
                    v_rows=v_rows,
                    q_idx=q_idx,
                    domain_idx=domain_idx,
                    query_positions=query_positions,
                    scale=float(scale),
                    sink_rows=global_count,
                )
            metadata = {
                "block_index": block_index,
                "group_index": group_index,
                "q_rows": int(q_rows.shape[0]),
                "kv_rows": domain_rows,
                "original_sink_rows": global_count,
                "scale": float(scale),
                "witness_record": witness_record,
            }
            token = enter_local_group(options, metadata)
            try:
                out[q_idx] = native_with_provider(
                    q_rows,
                    k_rows,
                    v_rows,
                    "local",
                    aligned=plan["square_aligned"][group_index],
                    sink_rows=global_count,
                )
            finally:
                exit_local_group(token)
            _append_receipt(
                options,
                block=block_index,
                kind="local",
                group_index=group_index,
                q_rows=int(q_rows.shape[0]),
                kv_rows=domain_rows,
                packed_rows=seq,
                video_start=int(video_start),
                video_end=int(video_end),
                support_mode="restricted_window",
                complement_executed=True,
                provider_route=("vdn_dense_warmup" if block_index < 2 else "vdn_local_sol_all_selected_e"),
                canonical_full_kv=False,
                original_sink_rows=global_count,
                gate_fingerprint=gate_fingerprint,
                adapter_fingerprint=adapter_fingerprint,
            )

    for anchor_index, (start, stop) in enumerate(plan["anchor_slices"]):
        qa = query[start:stop]
        out[start:stop] = native_with_provider(qa, key, value, "anchor")
        _append_receipt(
            options,
            block=block_index,
            kind="anchor",
            group_index=anchor_index,
            q_rows=int(qa.shape[0]),
            kv_rows=int(key.shape[0]),
            packed_rows=seq,
            video_start=int(video_start),
            video_end=int(video_end),
            support_mode="full",
            complement_executed=False,
            provider_route="vdn_anchor_native",
            gate_fingerprint=gate_fingerprint,
            adapter_fingerprint=adapter_fingerprint,
        )
    return out


def _diagnostic_make_vdn_forward(attn, state, block_index):
    underlying = _ORIGINAL_MAKE_VDN_FORWARD(attn, state, block_index)

    @functools.wraps(underlying)
    def wrapped(x, rope_freqs=None, transformer_options=None):
        options = transformer_options or {}
        request = parse_request(options)
        if request is None:
            return underlying(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
        if request["stage"] != "high" or options.get("h3_flow_stage") not in {None, "high"}:
            raise RuntimeError("first-high Sol-local E reached VDN outside the high stage")
        if not bool((getattr(state, "cfg", {}) or {}).get("linear_enabled", True)):
            raise RuntimeError("first-high Sol-local E requires the released VDN linear complement to remain enabled")
        local_options = dict(options)
        for key in (_BLOCK_KEY, _GATE_FP_KEY, _ADAPTER_FP_KEY):
            if key in local_options:
                raise RuntimeError(f"first-high Sol-local E private VDN option already exists: {key}")
        local_options[_BLOCK_KEY] = int(block_index)
        local_options[_GATE_FP_KEY] = _gate_fingerprint(state)
        local_options[_ADAPTER_FP_KEY] = _adapter_fingerprint(state, block_index)
        return underlying(x, rope_freqs=rope_freqs, transformer_options=local_options)

    wrapped._h3_first_high_sol_local_original_forward = underlying
    wrapped._h3_first_high_sol_local_diagnostic_v1 = True
    wrapped._vdn_forward = True
    return wrapped


def install() -> None:
    global _INSTALLED, _ORIGINAL_MAKE_VDN_FORWARD, _ORIGINAL_WINDOW_SOFTMAX
    if _INSTALLED:
        return
    from vdn_h3 import hybrid, retained

    if getattr(hybrid.make_vdn_forward, "_h3_first_high_sol_local_installer", False):
        _INSTALLED = True
        return
    _ORIGINAL_MAKE_VDN_FORWARD = hybrid.make_vdn_forward
    _ORIGINAL_WINDOW_SOFTMAX = retained.window_softmax_grouped_runtime
    _diagnostic_make_vdn_forward._h3_first_high_sol_local_installer = True
    hybrid.make_vdn_forward = _diagnostic_make_vdn_forward
    retained.window_softmax_grouped_runtime = _diagnostic_window_softmax
    _INSTALLED = True


__all__ = ["EVIDENCE_KEY", "RECEIPTS_KEY", "REQUEST_KEY", "install", "parse_request"]
