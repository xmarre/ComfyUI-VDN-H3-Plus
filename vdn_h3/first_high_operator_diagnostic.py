"""Bounded first-high operator comparison W for the VDN MiniMax-H3 path.

This module is diagnostic-only.  It leaves ordinary VDN execution untouched when
``h3_first_high_operator_diagnostic_v1`` is absent.  With a validated request it
provides the two controlled native-SDPA arms from the Flow first-high design:
restricted-window support with the released linear complement, or canonical full
K/V support for the same local query groups with that complement disabled.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import torch

REQUEST_KEY = "h3_first_high_operator_diagnostic_v1"
RECEIPTS_KEY = "h3_first_high_operator_receipts_v1"
_BLOCK_KEY = "h3_first_high_operator_vdn_block_v1"
_COMPLEMENT_KEY = "h3_first_high_operator_vdn_complement_v1"
_GATE_FP_KEY = "h3_first_high_operator_vdn_gate_fingerprint_v1"
_ADAPTER_FP_KEY = "h3_first_high_operator_vdn_adapter_fingerprint_v1"
_ALLOWED_MODES = frozenset({"native_window", "native_full_support"})
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
        raise RuntimeError("first-high operator diagnostic request must be an immutable tuple of exact fields")
    result: dict[str, Any] = {}
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str):
            raise RuntimeError("first-high operator diagnostic request entries must be (name, value) tuples")
        key, field_value = item
        if key in result:
            raise RuntimeError(f"duplicate first-high operator diagnostic field: {key}")
        result[key] = field_value
    if tuple(result) != _REQUIRED_FIELDS:
        raise RuntimeError("first-high operator diagnostic request fields/order do not match API 1")
    if result["api"] != 1:
        raise RuntimeError("unsupported first-high operator diagnostic API")
    if not isinstance(result["capture_id"], str) or not result["capture_id"]:
        raise RuntimeError("first-high operator diagnostic capture_id is invalid")
    if result["mode"] not in _ALLOWED_MODES:
        raise RuntimeError(f"unsupported first-high operator diagnostic mode: {result['mode']!r}")
    if result["stage"] != "high" or result["logical_call_limit"] != 1:
        raise RuntimeError("first-high operator diagnostic is restricted to one high-stage logical call")
    sigma = result["sigma"]
    if type(sigma) is not float or not math.isfinite(sigma) or not 0.0 < sigma <= 1.0:
        raise RuntimeError("first-high operator diagnostic sigma must be a finite float in (0, 1]")
    for name in ("target_shapes_digest", "source_contract_digest"):
        digest = result[name]
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in _HEX for ch in digest):
            raise RuntimeError(f"first-high operator diagnostic {name} must be lowercase SHA-256")
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
            "managed_weights_type": (
                None if managed is None else f"{type(managed).__module__}.{type(managed).__qualname__}"
            ),
            "linear_enabled": (getattr(state, "cfg", {}) or {}).get("linear_enabled", True),
            "enable_text_state": None if branch is None else bool(getattr(branch, "enable_text_state", False)),
            "delta_rule": None if branch is None else getattr(branch, "delta_rule", None),
            "bridge": None if branch is None else getattr(branch, "bridge", None),
        }
    )


class _FullSupportStateProxy:
    """Read-through VDN state with a private immutable full-support config view."""

    def __init__(self, state: Any):
        self._state = state
        self.cfg = dict(state.cfg)
        self.cfg["linear_enabled"] = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._state, name)

    @property
    def layout(self):
        return self._state.layout

    @property
    def retain_buffers(self):
        return self._state.retain_buffers

    def weights_on(self, index, device, dtype):
        return self._state.weights_on(index, device, dtype)


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
            index = torch.div(
                ordinal * (int(flat.numel()) - 1), count - 1, rounding_mode="floor"
            )
        sampled = flat.index_select(0, index).contiguous().view(torch.uint8).to(device="cpu")
        digest.update(bytes(sampled.tolist()))
    return digest.hexdigest()


def _append_backend_receipt(
    options: dict[str, Any], block_index: int, route: str, request: dict[str, Any]
) -> None:
    sink = options.get("attention_backend_receipts_v1")
    if sink is None:
        return
    if not isinstance(sink, list):
        raise RuntimeError("first-high operator diagnostic found an invalid attention backend receipt sink")
    sink.append(
        (
            "sol_h3",
            int(block_index),
            str(route),
            (
                ("capture_id", request["capture_id"]),
                ("mode", request["mode"]),
                ("source_contract_digest", request["source_contract_digest"]),
                ("completed", True),
            ),
        )
    )


def _append_receipt(options: dict[str, Any], **fields: Any) -> None:
    sink = options.get(RECEIPTS_KEY)
    append = getattr(sink, "append", None)
    if not callable(append):
        raise RuntimeError("first-high operator diagnostic receipt sink is missing or invalid")
    try:
        count = len(sink)
    except TypeError as exc:
        raise RuntimeError("first-high operator diagnostic receipt sink is not bounded") from exc
    if count >= 800:
        raise RuntimeError("first-high operator diagnostic receipt bound exceeded")
    append(fields)


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
    if options.get("vdn_h3_external_sequence_v1") is not None:
        raise RuntimeError("first-high operator diagnostic forbids an external/reduced VDN sequence")
    if (
        options.get("attention_measure_v1") is not None
        or options.get("h3_flow_mixed_grid_attention_measure_v1") is not None
    ):
        raise RuntimeError("first-high operator diagnostic forbids weighted/Mixed-Grid attention")
    block_index = options.get(_BLOCK_KEY)
    complement = options.get(_COMPLEMENT_KEY)
    gate_fingerprint = options.get(_GATE_FP_KEY)
    adapter_fingerprint = options.get(_ADAPTER_FP_KEY)
    if type(block_index) is not int or not 0 <= block_index < 50:
        raise RuntimeError("first-high operator diagnostic VDN block identity is missing")
    if type(complement) is not bool:
        raise RuntimeError("first-high operator diagnostic complement identity is missing")
    if not isinstance(gate_fingerprint, str) or not isinstance(adapter_fingerprint, str):
        raise RuntimeError("first-high operator diagnostic VDN fingerprints are missing")

    # Exactly one shape-preserving preprocessing pass on canonical packed Q/K/V.
    query, key, value = preprocess(options, query, key, value, query.shape[1])
    heads, head_dim = query.shape[1], query.shape[2]
    seq = query.shape[0]
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

    pre_attention_qkv_digest = (
        _sampled_tensor_digest(query, key, value) if block_index == 0 else None
    )

    out = torch.empty_like(query)
    global_idx = plan["global_idx"]
    global_count = int(global_idx.numel())

    def native_with_provider(q, k, v, kind, *, aligned=False, sink_rows=0):
        # Non-local provider dispatch is intentionally retained: Sol's existing
        # provider records the native global/anchor route and then calls this exact
        # VDN SDPA closure.  Local calls bypass the provider below so SOL sparse
        # approximation cannot enter either W arm.
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
    mode = request["mode"]
    local_route = "vdn_local_native_window_w" if mode == "native_window" else "vdn_local_native_full_w"
    if groups:
        k_scratch = v_scratch = None
        if mode == "native_window":
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
            q_rows = query.index_select(0, q_idx)
            if mode == "native_window":
                window_rows = int(win_idx.numel())
                domain_rows = global_count + window_rows
                torch.index_select(key, 0, win_idx, out=k_scratch[global_count:domain_rows])
                torch.index_select(value, 0, win_idx, out=v_scratch[global_count:domain_rows])
                k_rows = k_scratch[:domain_rows]
                v_rows = v_scratch[:domain_rows]
                support_mode = "restricted_window"
                canonical_full_kv = False
            else:
                # Canonical packed K/V is consumed directly: no global/window
                # concatenation and therefore no duplicated rows.
                k_rows = key
                v_rows = value
                domain_rows = int(key.shape[0])
                support_mode = "canonical_full"
                canonical_full_kv = True
            out[q_idx] = W._sdpa(q_rows, k_rows, v_rows, scale, None)
            _append_backend_receipt(options, block_index, local_route, request)
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
                support_mode=support_mode,
                complement_executed=bool(complement),
                provider_route=local_route,
                canonical_full_kv=canonical_full_kv,
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
    normal_forward = _ORIGINAL_MAKE_VDN_FORWARD(attn, state, block_index)
    full_forward = _ORIGINAL_MAKE_VDN_FORWARD(attn, _FullSupportStateProxy(state), block_index)

    @functools.wraps(normal_forward)
    def wrapped(x, rope_freqs=None, transformer_options=None):
        options = transformer_options or {}
        request = parse_request(options)
        if request is None:
            return normal_forward(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
        if request["stage"] != "high":
            raise RuntimeError("first-high operator diagnostic reached VDN outside the high stage")
        if options.get("h3_flow_stage") not in {None, "high"}:
            raise RuntimeError("first-high operator diagnostic VDN stage marker changed")
        if options.get("vdn_h3_external_sequence_v1") is not None:
            raise RuntimeError("first-high operator diagnostic forbids external/reduced VDN sequence")
        local_options = dict(options)
        if any(key in local_options for key in (_BLOCK_KEY, _COMPLEMENT_KEY, _GATE_FP_KEY, _ADAPTER_FP_KEY)):
            raise RuntimeError("first-high operator diagnostic private VDN fields were already populated")
        complement = bool(request["mode"] == "native_window" and state.cfg.get("linear_enabled", True))
        local_options[_BLOCK_KEY] = int(block_index)
        local_options[_COMPLEMENT_KEY] = complement
        local_options[_GATE_FP_KEY] = _gate_fingerprint(state)
        local_options[_ADAPTER_FP_KEY] = _adapter_fingerprint(state, block_index)
        selected = normal_forward if request["mode"] == "native_window" else full_forward
        return selected(x, rope_freqs=rope_freqs, transformer_options=local_options)

    # Sol's diagnostic history bridge uses this exact underlying forward to
    # preserve the ordinary grouped identity when W is absent.
    wrapped._h3_first_high_operator_original_forward = normal_forward
    wrapped._h3_first_high_operator_diagnostic_v1 = True
    wrapped._vdn_forward = True
    return wrapped


def install() -> None:
    global _INSTALLED, _ORIGINAL_MAKE_VDN_FORWARD, _ORIGINAL_WINDOW_SOFTMAX
    if _INSTALLED:
        return
    from vdn_h3 import hybrid, retained

    if getattr(hybrid.make_vdn_forward, "_h3_first_high_operator_installer", False):
        _INSTALLED = True
        return
    _ORIGINAL_MAKE_VDN_FORWARD = hybrid.make_vdn_forward
    _ORIGINAL_WINDOW_SOFTMAX = retained.window_softmax_grouped_runtime
    _diagnostic_make_vdn_forward._h3_first_high_operator_installer = True
    hybrid.make_vdn_forward = _diagnostic_make_vdn_forward
    retained.window_softmax_grouped_runtime = _diagnostic_window_softmax
    _INSTALLED = True


__all__ = ["REQUEST_KEY", "RECEIPTS_KEY", "install", "parse_request"]
