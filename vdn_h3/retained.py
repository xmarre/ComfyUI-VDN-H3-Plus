"""Execution-owned retained helpers layered on the released VDN math.

This module intentionally subclasses :class:`vdn_h3.branch.LinearBranch` instead of
turning the reference-math module into a cache owner. The arithmetic remains the
same; only recurrence banks and grouped-window gather storage are borrowed from the
execution-local ``RuntimeBuffers`` lease when retention is enabled.
"""
from __future__ import annotations

import collections

import torch
import torch.nn.functional as F

from vdn_h3 import branch as B
from vdn_h3 import window as W
from vdn_h3.runtime import current_runtime_buffers


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

        a, b = B.frame_statistics(
            key_by_frame, value_by_frame, beta, a_fp32=self.a_fp32)
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

        text_state = self._text_state(w, text_x, text_k_raw, text_v_raw)
        prefix_states, suffix_states = run_scans_runtime(
            backend, alpha, a, b, text_state=text_state)
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

        if query.dim() == 4:
            query_fhsd = query
        else:
            query_fhsd = query.view(shape).permute(0, 2, 1, 3)
        readout = torch.matmul(
            query_fhsd, linear_state.transpose(-1, -2))
        return B.linear_epilogue(
            readout,
            w["norm.weight"],
            gate,
            w["norm.weight"].new_tensor(1e-6).item(),
            fuse=self.fuse_epilogue,
        )


def _build_window_plan(video_start, video_end, num_frames, tokens_per_frame,
                       bounds, anchor_frames, seq, device):
    """Build immutable row-index geometry for one packed-sequence window layout."""
    def frame_rows(frame):
        start = video_start + frame * tokens_per_frame
        return torch.arange(start, start + tokens_per_frame, device=device)

    global_idx = torch.cat([
        torch.arange(video_start, device=device),
        torch.arange(video_end, seq, device=device),
    ])
    anchors = (0, num_frames - 1)
    anchor_rows = sorted(
        frame for frame in anchors
        if anchor_frames in ("rows", "both"))
    anchor_set = set(anchor_rows)

    grouped = collections.OrderedDict()
    for frame in range(num_frames):
        if frame in anchor_set:
            continue
        lo = max(bounds[frame][0], 0)
        hi = min(bounds[frame][1], num_frames - 1)
        grouped.setdefault((lo, hi), []).append(frame)

    groups = []
    square_aligned = []
    max_kv_rows = global_idx.numel()
    global_count = global_idx.numel()
    for (lo, hi), frames in grouped.items():
        extra = [
            frame for frame in anchors
            if anchor_frames in ("columns", "both") and not lo <= frame <= hi
        ]
        key_frames = sorted(set(range(lo, hi + 1)) | set(extra))
        win_idx = torch.cat([frame_rows(frame) for frame in key_frames])
        q_idx = torch.cat([frame_rows(frame) for frame in frames])
        # K/V are consumed in [global rows, window rows] order. The v2 square
        # domain uses real Q rows in exactly that same order. Mapping requested Q
        # rows into the square output is therefore independent of attention math.
        domain_idx = torch.cat((global_idx, win_idx)) if global_count else win_idx
        query_positions = torch.searchsorted(win_idx, q_idx) + global_count
        groups.append((q_idx, win_idx, domain_idx, query_positions))
        square_aligned.append(global_count == 0 and frames == key_frames)
        max_kv_rows = max(max_kv_rows, global_count + win_idx.numel())

    return {
        "global_idx": global_idx,
        "groups": groups,
        "square_aligned": square_aligned,
        "anchor_slices": [
            (
                video_start + frame * tokens_per_frame,
                video_start + (frame + 1) * tokens_per_frame,
            )
            for frame in anchor_rows
        ],
        "max_kv_rows": max_kv_rows,
    }


def window_softmax_grouped_runtime(query, key, value, video_start, video_end,
                                   num_frames, tokens_per_frame, bounds, scale,
                                   anchor_frames="none", transformer_options=None):
    """Grouped exact window softmax with execution-owned plan/KV scratch reuse.

    VDN's native local-window operator is exact SDPA. Generic model-level dense
    overrides do not leak into the trained local branch. Explicit v1/v2 providers
    may consume only the row domains VDN already selected. v2 may evaluate real
    extra Q rows aligned to the restricted KV domain, then return the requested
    rows; it may not expand or alter the KV domain.
    """
    from .softmax_provider import dispatch, has_v2, preprocess

    # This is still the complete post-RoPE packed sequence. Run explicit QKV
    # preprocessing here, before any VDN row gathering, so transforms that depend
    # on original packed coordinates (for example Untwist) remain well-defined.
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
    plan = resources.window_plan(plan_key, builder) if resources is not None else builder()

    out = torch.empty_like(query)
    global_idx = plan["global_idx"]
    global_count = global_idx.numel()
    if global_count:
        out[global_idx] = attend(query[global_idx], key, value, "global")

    groups = plan["groups"]
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
        v2 = has_v2(transformer_options)
        for group_index, (q_idx, win_idx, domain_idx, query_positions) in enumerate(groups):
            window_rows = win_idx.numel()
            domain_rows = global_count + window_rows
            torch.index_select(
                key, 0, win_idx,
                out=k_scratch[global_count:domain_rows])
            torch.index_select(
                value, 0, win_idx,
                out=v_scratch[global_count:domain_rows])
            q_rows = query.index_select(0, q_idx)
            square_q = query.index_select(0, domain_idx) if v2 else None
            out[q_idx] = attend(
                q_rows,
                k_scratch[:domain_rows],
                v_scratch[:domain_rows],
                "local",
                plan["square_aligned"][group_index],
                square_q=square_q,
                query_positions=query_positions if v2 else None,
                sink_rows=global_count,
            )

    for start, stop in plan["anchor_slices"]:
        out[start:stop] = attend(query[start:stop], key, value, "anchor")
    return out