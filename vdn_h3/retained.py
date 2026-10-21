"""Execution-owned retained helpers layered on the released VDN math.

This module intentionally subclasses :class:`vdn_h3.branch.LinearBranch` instead of
turning the reference-math module into a cache owner. The arithmetic remains the
same; only recurrence banks and grouped-window gather storage are borrowed from the
execution-local ``RuntimeBuffers`` lease when retention is enabled.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vdn_h3 import branch as B
from vdn_h3 import window as W
from vdn_h3.query_positions import (
    GEOMETRY_SCHEMA,
    bind_query_map,
    describe_window_geometry,
)
from vdn_h3.runtime import current_runtime_buffers


# Limit frame-statistics temporaries without changing the per-frame VDN
# equations. The native uniform path otherwise multiplies an entire high-grid
# value tensor at once (for example 55 * 56 * 1150 * 128 BF16 = 864.75 MiB),
# while fp32 key/scaled-key temporaries and the original features are live.
_FRAME_STATISTICS_VALUE_BUDGET_BYTES = 128 << 20


def _bounded_frame_statistics(key, value, beta, *, a_fp32):
    """Evaluate independent frame statistics in bounded contiguous frame runs.

    Only the batch dimension is partitioned. Spatial tokens, accumulation
    dtype, TF32 policy and the recurrence's frame order remain unchanged.
    """
    frames = int(value.shape[0])
    if frames < 2:
        return B.frame_statistics(key, value, beta, a_fp32=a_fp32)
    bytes_per_frame = value[0].numel() * value.element_size()
    frames_per_run = max(1, _FRAME_STATISTICS_VALUE_BUDGET_BYTES // bytes_per_frame)
    if frames <= frames_per_run:
        return B.frame_statistics(key, value, beta, a_fp32=a_fp32)
    a_parts, b_parts = [], []
    for first in range(0, frames, frames_per_run):
        last = min(first + frames_per_run, frames)
        a, b = B.frame_statistics(
            key[first:last], value[first:last], beta[first:last],
            a_fp32=a_fp32,
        )
        a_parts.append(a)
        b_parts.append(b)
    return torch.cat(a_parts, dim=0), torch.cat(b_parts, dim=0)


def run_scans_runtime(backend, alpha, a_raw, b_raw, text_state=None):
    """Reference recurrence with execution-local bank reuse when available."""
    with torch.autocast(device_type=a_raw.device.type, enabled=False):
        transitions, injections = backend.factor_apply(alpha, a_raw, b_raw)
        num_frames = transitions.shape[0]
        start = (
            torch.zeros_like(injections[0])
            if text_state is None else text_state.to(injections.dtype)
        )
        resources = current_runtime_buffers()
        if resources is None:
            prefix = torch.empty(
                (num_frames, *start.shape), dtype=injections.dtype,
                device=injections.device)
            suffix = torch.empty_like(prefix)
        else:
            prefix, suffix = resources.scan_banks(
                num_frames, start.shape, injections.dtype, injections.device)

        state = start
        for frame in range(num_frames):
            torch.baddbmm(
                injections[frame], state, transitions[frame], out=prefix[frame])
            state = prefix[frame]
        state = start
        for frame in range(num_frames - 1, -1, -1):
            torch.baddbmm(
                injections[frame], state, transitions[frame], out=suffix[frame])
            state = suffix[frame]
        return prefix, suffix


class RuntimeLinearBranch(B.LinearBranch):
    """LinearBranch whose reusable state banks belong to the current VDN execution."""

    def _readout(self, w, xv, qkv_raw, num_frames, tokens_per_frame, bounds,
                 frame_size, text_x, text_k_raw, text_v_raw):
        n_heads, head_dim = self.num_heads, self.head_dim
        backend = self._delta_backend(tokens_per_frame)
        shape = (num_frames, tokens_per_frame, n_heads, head_dim)

        query, key, value = self._features(
            w, *qkv_raw, num_frames, frame_size,
            q_fhsd=self.fuse_epilogue)
        key_by_frame = key.view(shape).permute(0, 2, 1, 3)
        value_by_frame = value.view(shape).permute(0, 2, 1, 3)
        beta = torch.sigmoid(F.linear(xv, w["beta_proj.weight"]))
        beta = beta.view(
            num_frames, tokens_per_frame, n_heads).permute(0, 2, 1)

        a, b = _bounded_frame_statistics(
            key_by_frame, value_by_frame, beta, a_fp32=self.a_fp32)
        # Features and their views are no longer needed after frame statistics.
        del key_by_frame, value_by_frame, key, value, beta
        frame_mean = xv.view(num_frames, tokens_per_frame, -1).mean(
            dim=1, dtype=torch.float32)
        alpha = B.alpha_gate(
            frame_mean,
            w["alpha.down.weight"],
            w["alpha.up.weight"],
            w["alpha.dt_bias"],
            w["alpha.A_log"],
            n_heads,
            head_dim,
        )
        del frame_mean

        text_state = self._text_state(w, text_x, text_k_raw, text_v_raw)
        prefix_states, suffix_states = run_scans_runtime(
            backend, alpha, a, b, text_state=text_state)
        del a, b

        gate = torch.sigmoid(
            F.linear(xv, w["output_gate.down.weight"])
            @ w["output_gate.up.weight"].T
            + w["output_gate.up.bias"]
        )
        linear_state = B.gather_linear_state(
            prefix_states,
            suffix_states,
            alpha,
            bounds,
            bridge=self.bridge,
            text_state=text_state,
            out_dtype=gate.dtype,
            fuse=self.fuse_epilogue,
        )
        del prefix_states, suffix_states

        if query.dim() == 4:
            query_fhsd = query
        else:
            query_fhsd = query.view(shape).permute(0, 2, 1, 3)
        readout = torch.matmul(
            query_fhsd, linear_state.transpose(-1, -2))
        del query_fhsd, query, linear_state
        return B.linear_epilogue(
            readout,
            w["norm.weight"],
            gate,
            B.linear_norm_epsilon(w["norm.weight"].dtype),
            fuse=self.fuse_epilogue,
        )


def _build_window_plan(video_start, video_end, num_frames, tokens_per_frame,
                       bounds, anchor_frames, seq, device):
    """Materialize row tensors from the same pure geometry exported to provider v4."""
    geometry = describe_window_geometry(
        int(video_start),
        int(video_end),
        int(num_frames),
        int(tokens_per_frame),
        tuple(tuple(int(value) for value in pair) for pair in bounds),
        str(anchor_frames),
        int(seq),
    )

    global_idx = torch.tensor(geometry.global_rows, dtype=torch.long, device=device)
    groups = []
    square_aligned = []
    contiguous_query_spans = []
    for group in geometry.groups:
        # Grouping by identical window bounds usually yields contiguous
        # query frames (especially clipped edge windows). Detect this from
        # immutable CPU geometry so the runtime may alias Q without a gather
        # allocation or indexed output scatter. Noncontiguous groups preserve
        # the original gather path and its index order.
        frames = group.query_frames
        contiguous = bool(frames) and all(
            right == left + 1 for left, right in zip(frames, frames[1:])
        )
        if contiguous:
            start = geometry.video_start + frames[0] * geometry.tokens_per_frame
            stop = geometry.video_start + (frames[-1] + 1) * geometry.tokens_per_frame
            contiguous_query_spans.append((start, stop))
        else:
            contiguous_query_spans.append(None)
        q_parts = []
        for frame in group.query_frames:
            start = geometry.video_start + frame * geometry.tokens_per_frame
            q_parts.append(torch.arange(start, start + geometry.tokens_per_frame, device=device))
        k_parts = []
        for frame in group.key_frames:
            start = geometry.video_start + frame * geometry.tokens_per_frame
            k_parts.append(torch.arange(start, start + geometry.tokens_per_frame, device=device))
        q_idx = torch.cat(q_parts) if q_parts else torch.empty(0, dtype=torch.long, device=device)
        win_idx = torch.cat(k_parts) if k_parts else torch.empty(0, dtype=torch.long, device=device)
        groups.append((q_idx, win_idx))
        square_aligned.append(group.square_aligned)

    return {
        "geometry_schema": GEOMETRY_SCHEMA,
        "geometry": geometry,
        "plan_digest": geometry.plan_digest,
        "global_idx": global_idx,
        "groups": groups,
        "square_aligned": square_aligned,
        "contiguous_query_spans": contiguous_query_spans,
        "anchor_slices": list(geometry.anchor_slices),
        "max_kv_rows": geometry.max_kv_rows,
    }


def window_softmax_grouped_runtime(query, key, value, video_start, video_end,
                                   num_frames, tokens_per_frame, bounds, scale,
                                   anchor_frames="none", transformer_options=None,
                                   query_position_owner=None):
    """Grouped exact window softmax with execution-owned plan/KV scratch reuse.

    v4 receives VDN-owned query positions in the exact restricted K/V domain used
    by the gather below. v3 remains a direct rectangular ABI; v2 remains the legacy
    lazy square-Q compatibility ABI. A present but malformed v4 suppresses v2/v3
    routing and is handled by v4 dispatch as native for that call.
    """
    from .softmax_provider import dispatch, execute_grouped_kv, has_v2, has_v3, has_v4, preprocess

    query, key, value = preprocess(
        transformer_options, query, key, value, query.shape[1])

    def attend(q, k, v, kind, aligned=False, **contract):
        return dispatch(transformer_options, lambda: W._sdpa(q, k, v, scale, None),
                        q, k, v, kind=kind, scale=scale,
                        square_aligned=aligned, **contract)
    heads, head_dim = query.shape[1], query.shape[2]
    seq = query.shape[0]
    resources = current_runtime_buffers()
    plan_key = (
        GEOMETRY_SCHEMA,
        video_start,
        video_end,
        num_frames,
        tokens_per_frame,
        tuple(map(tuple, bounds)),
        anchor_frames,
        seq,
        str(query.device),
    )
    builder = lambda: _build_window_plan(
        video_start, video_end, num_frames, tokens_per_frame,
        bounds, anchor_frames, seq, query.device)
    plan = resources.window_plan(plan_key, builder, device=query.device) if resources is not None else builder()

    out = torch.empty_like(query)
    global_idx = plan["global_idx"]
    global_count = global_idx.numel()
    if global_count:
        out[global_idx] = attend(query[global_idx], key, value, "global")

    groups = plan["groups"]
    from .softmax_provider import KEY as V1, KEY_V2 as V2, KEY_V3 as V3, KEY_V4 as V4
    opts = transformer_options or {}
    v4 = opts.get(V4)
    # A custom provider may mutate its Q argument. The old advanced-index
    # gather isolates that mutation from the underlying full-sequence Q.
    # Only the built-in native SDPA or a provider explicitly guaranteeing
    # read-only Q may receive zero-copy query views.
    readonly_q = (
        not any(key in opts for key in (V1, V2, V3, V4))
        or (callable(v4) and getattr(v4, "_vdn_readonly_query_contract_v1", False) is True)
    )
    if groups:
        if resources is None:
            shape = (plan["max_kv_rows"], heads, head_dim)
            k_scratch = torch.empty(shape, device=key.device, dtype=key.dtype)
            v_scratch = torch.empty(shape, device=value.device, dtype=value.dtype)
        else:
            k_scratch, v_scratch = resources.kv_scratch(
                plan["max_kv_rows"], heads, head_dim, key.device, key.dtype)
        if global_count:
            torch.index_select(key, 0, global_idx, out=k_scratch[:global_count])
            torch.index_select(value, 0, global_idx, out=v_scratch[:global_count])
        v4 = has_v4(transformer_options)
        v3 = not v4 and has_v3(transformer_options)
        v2 = not v4 and not v3 and has_v2(transformer_options)
        for group_index, (q_idx, win_idx) in enumerate(groups):
            q_span = plan["contiguous_query_spans"][group_index] if readonly_q and query.is_contiguous() else None
            window_rows = win_idx.numel()
            domain_rows = global_count + window_rows
            q_rows = (
                query[q_span[0]:q_span[1]]
                if q_span is not None else query.index_select(0, q_idx)
            )
            contract = {"sink_rows": int(global_count)}
            if v4:
                query_position_map = None
                if isinstance(query_position_owner, str) and query_position_owner:
                    query_position_map = bind_query_map(
                        plan["geometry"], group_index, query_position_owner
                    )
                contract["query_position_map"] = query_position_map
            elif v2:
                domain_idx = torch.cat((global_idx, win_idx)) if global_count else win_idx
                query_positions = torch.searchsorted(win_idx, q_idx) + global_count
                contract.update(
                    square_q=query.index_select(0, domain_idx),
                    query_positions=query_positions,
                )
            result = execute_grouped_kv(
                opts.get(V4) if v4 else None,
                key, value, k_scratch[:domain_rows], v_scratch[:domain_rows],
                win_idx, global_count,
                lambda kg, vg: attend(q_rows, kg, vg, "local",
                                      plan["square_aligned"][group_index], **contract),
                transformer_options=transformer_options,
            )
            if q_span is not None:
                out[q_span[0]:q_span[1]] = result
            else:
                out[q_idx] = result

    for start, stop in plan["anchor_slices"]:
        out[start:stop] = attend(query[start:stop], key, value, "anchor")
    return out
