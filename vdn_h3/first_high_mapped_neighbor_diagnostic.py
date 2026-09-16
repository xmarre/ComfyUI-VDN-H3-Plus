"""VDN-owned query-position transport for bounded mapped-neighbor arm M.

M is a diagnostic overlay on the reviewed E stack.  It preserves the released
restricted local K/V gather and linear complement, derives query positions from
VDN's exact live window plan, and passes only a bounded per-Q64 mapped-neighbor
descriptor to the Sol M companion.  Ordinary VDN behavior is unchanged without
the M request.
"""
from __future__ import annotations

import hashlib
from typing import Any

import torch

from . import first_high_sol_local_diagnostic as e
from .first_high_mapped_neighbor_bridge import enter_mapped_group, exit_mapped_group
from .first_high_sol_local_bridge import parse_sol_request

MODE = "mapped_neighbor_m"
ROUTE = "vdn_local_sol_mapped_neighbor_m"
_INSTALLED = False
_CURRENT_WINDOW_SOFTMAX = None


def _tensor_sha256(value: torch.Tensor) -> str:
    work = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(memoryview(work.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _request(options: dict[str, Any]) -> dict[str, Any] | None:
    value = options.get(e.REQUEST_KEY)
    if value is None:
        return None
    request = parse_sol_request(options)
    if not isinstance(request, dict) or request.get("mode") != MODE:
        raise RuntimeError("mapped-neighbor M request does not match the Sol companion")
    return request


def _mapped_intervals(
    q_idx: torch.Tensor,
    domain_idx: torch.Tensor,
    query_positions: torch.Tensor,
) -> tuple[tuple[int, int], ...]:
    """Map each contiguous Q64 tile to represented K blocks plus +/-1 neighbors."""
    if q_idx.ndim != 1 or domain_idx.ndim != 1 or query_positions.ndim != 1:
        raise RuntimeError("mapped-neighbor M row maps must be one-dimensional")
    if int(q_idx.numel()) != int(query_positions.numel()) or int(q_idx.numel()) <= 0:
        raise RuntimeError("mapped-neighbor M query-position map has the wrong length")
    if q_idx.dtype != torch.long or domain_idx.dtype != torch.long or query_positions.dtype != torch.long:
        raise RuntimeError("mapped-neighbor M row maps must use torch.long")
    if bool((query_positions < 0).any().item()) or bool((query_positions >= int(domain_idx.numel())).any().item()):
        raise RuntimeError("mapped-neighbor M query position is outside restricted K/V")
    mapped_rows = domain_idx.index_select(0, query_positions)
    if not torch.equal(mapped_rows, q_idx):
        raise RuntimeError("mapped-neighbor M query positions do not map back to the requested VDN rows")
    if int(query_positions.numel()) > 1 and not bool((query_positions[1:] > query_positions[:-1]).all().item()):
        raise RuntimeError("mapped-neighbor M query positions are not strictly increasing")

    k_blocks = (int(domain_idx.numel()) + 63) // 64
    q_rows = int(q_idx.numel())
    intervals = []
    for start in range(0, q_rows, 64):
        stop = min(q_rows, start + 64)
        tile = query_positions[start:stop]
        if int(tile.numel()) != stop - start:
            raise RuntimeError("mapped-neighbor M lost a Q64 tile")
        if int(tile.numel()) > 1 and not bool((tile[1:] == tile[:-1] + 1).all().item()):
            raise RuntimeError(
                "mapped-neighbor M first arm supports only contiguous VDN query-position tiles; "
                "general maps require a separately reviewed bounded representation"
            )
        first_block = int(tile[0].item()) // 64
        last_block = int(tile[-1].item()) // 64
        if last_block - first_block > 1:
            raise RuntimeError("mapped-neighbor M Q64 tile spans more than two K64 blocks")
        interval_start = max(0, first_block - 1)
        interval_end = min(k_blocks, last_block + 2)
        if not interval_start < interval_end or interval_end - interval_start > 4:
            raise RuntimeError("mapped-neighbor M derived interval violates the reviewed +/-1 bound")
        intervals.append((interval_start, interval_end))
    return tuple(intervals)


def _mapped_window_softmax(
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
    from vdn_h3 import retained
    from vdn_h3 import window as W
    from vdn_h3.runtime import current_runtime_buffers
    from vdn_h3.softmax_provider import dispatch, preprocess

    options = transformer_options or {}
    request = _request(options)
    if request is None:
        return _CURRENT_WINDOW_SOFTMAX(
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
        raise RuntimeError("mapped-neighbor M forbids external/reduced VDN sequence")
    if options.get("attention_measure_v1") is not None or options.get("h3_flow_mixed_grid_attention_measure_v1") is not None:
        raise RuntimeError("mapped-neighbor M forbids weighted/Mixed-Grid attention")

    block_index = options.get(e._BLOCK_KEY)
    gate_fingerprint = options.get(e._GATE_FP_KEY)
    adapter_fingerprint = options.get(e._ADAPTER_FP_KEY)
    if type(block_index) is not int or not 0 <= block_index < 50:
        raise RuntimeError("mapped-neighbor M VDN block identity is missing")
    if not isinstance(gate_fingerprint, str) or not isinstance(adapter_fingerprint, str):
        raise RuntimeError("mapped-neighbor M VDN fingerprints are missing")

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
    pre_attention_qkv_digest = e._sampled_tensor_digest(query, key, value) if block_index == 0 else None

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
        e._append_receipt(
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
        raise RuntimeError(f"mapped-neighbor M expected 11 local groups, got {len(groups)}")
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
            if int(q_rows.shape[0]) != e._EXPECTED_LOCAL_Q_ROWS[group_index]:
                raise RuntimeError(
                    f"mapped-neighbor M local Q geometry changed for group {group_index}: {q_rows.shape[0]}"
                )
            if domain_rows != e._EXPECTED_WINDOW_KV_ROWS[group_index]:
                raise RuntimeError(
                    f"mapped-neighbor M restricted K/V geometry changed for group {group_index}: {domain_rows}"
                )
            domain_idx = torch.cat((global_idx, win_idx)) if global_count else win_idx
            query_positions = torch.searchsorted(win_idx, q_idx) + global_count

            if block_index < 2:
                out[q_idx] = native_with_provider(
                    q_rows,
                    k_rows,
                    v_rows,
                    "local",
                    aligned=plan["square_aligned"][group_index],
                    sink_rows=global_count,
                )
                provider_route = "vdn_dense_warmup"
            else:
                intervals = _mapped_intervals(q_idx, domain_idx, query_positions)
                metadata = {
                    "block_index": block_index,
                    "group_index": group_index,
                    "q_rows": int(q_rows.shape[0]),
                    "kv_rows": domain_rows,
                    "original_sink_rows": global_count,
                    "scale": float(scale),
                    "mapped_neighbor_intervals": intervals,
                    "query_positions_sha256": _tensor_sha256(query_positions),
                }
                token = enter_mapped_group(options, metadata)
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
                    exit_mapped_group(token)
                provider_route = ROUTE

            e._append_receipt(
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
                provider_route=provider_route,
                canonical_full_kv=False,
                original_sink_rows=global_count,
                gate_fingerprint=gate_fingerprint,
                adapter_fingerprint=adapter_fingerprint,
            )

    for anchor_index, (start, stop) in enumerate(plan["anchor_slices"]):
        qa = query[start:stop]
        out[start:stop] = native_with_provider(qa, key, value, "anchor")
        e._append_receipt(
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


def install() -> None:
    global _INSTALLED, _CURRENT_WINDOW_SOFTMAX
    if _INSTALLED:
        return
    from vdn_h3 import retained

    current = retained.window_softmax_grouped_runtime
    if getattr(current, "_h3_first_high_mapped_neighbor_m_v1", False):
        _INSTALLED = True
        return
    if not getattr(current, "__module__", "").endswith("first_high_sol_local_diagnostic"):
        raise RuntimeError("mapped-neighbor M requires the reviewed E VDN window wrapper underneath")
    _CURRENT_WINDOW_SOFTMAX = current
    _mapped_window_softmax._h3_first_high_mapped_neighbor_m_v1 = True
    _mapped_window_softmax._h3_first_high_mapped_neighbor_inner = current
    retained.window_softmax_grouped_runtime = _mapped_window_softmax
    _INSTALLED = True


__all__ = ["MODE", "ROUTE", "install", "_mapped_intervals"]
