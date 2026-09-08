"""VDN-H3 hybrid-attention integration for ComfyUI MiniMax-H3."""
from __future__ import annotations

import collections
import contextvars
import copy
import logging

import torch
import torch.nn.functional as F

import comfy.ldm.common_dit
import comfy.ldm.minimax.model as minimax_model
import comfy.model_management
import comfy.quant_ops
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
from comfy.patcher_extension import WrappersMP

from vdn_h3.runtime import RuntimeBufferOwner
from vdn_h3.spec import resolve_branch_weights
from vdn_h3.window import full_coverage, window_bounds

_log = logging.getLogger("comfy.vdn")
_seen = collections.OrderedDict()
_MAX_SEEN = 128

VDN_EXTERNAL_SEQUENCE_KEY = "vdn_h3_external_sequence_v1"
VDN_EXTERNAL_SEQUENCE_API_VERSION = 2
VDN_EXTERNAL_SEQUENCE_MODE = "dense_gate_no_linear"


def _once(key, message):
    if key in _seen:
        _seen.move_to_end(key)
        return
    _seen[key] = None
    while len(_seen) > _MAX_SEEN:
        _seen.popitem(last=False)
    _log.info("[vdn] %s", message)


class VDNLayout:
    __slots__ = (
        "video_start", "video_end", "num_frames", "tokens_per_frame",
        "frame_size", "text_start", "text_len", "bounds", "full_cover",
        "seq_len", "anchor_frames",
    )

    def __init__(self, video_start, video_end, num_frames, tokens_per_frame,
                 frame_size, text_start, text_len, seq_len, radius, chunk,
                 anchor_frames):
        self.video_start = video_start
        self.video_end = video_end
        self.num_frames = num_frames
        self.tokens_per_frame = tokens_per_frame
        self.frame_size = frame_size
        self.text_start = text_start
        self.text_len = text_len
        self.seq_len = seq_len
        self.bounds = window_bounds(num_frames, radius, chunk)
        self.full_cover = full_coverage(self.bounds, num_frames)
        self.anchor_frames = anchor_frames


class VDNState:
    """One Apply-VDN application's config, branch ownership and runtime resources."""

    def __init__(self, name, cfg, branches, num_heads, head_dim,
                 managed_weights=None, retain_buffers=False):
        self.name = name
        self.cfg = cfg
        self.branches = branches
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.managed_weights = managed_weights
        self.softmax_backend = "grouped"
        self.runtime = RuntimeBufferOwner(retain_buffers)
        self._layout = contextvars.ContextVar(f"vdn_layout_{id(self)}", default=None)

    @property
    def layout(self):
        return self._layout.get()

    @property
    def retain_buffers(self):
        return self.runtime.retain

    def _stream_weights(self, index, device, dtype):
        return resolve_branch_weights(self.branches[index].w, device, dtype)

    def weights_on(self, index, device, dtype):
        if self.managed_weights is not None:
            return self.managed_weights.weights_on(index, device, dtype)

        resources = self.runtime.current()
        if (resources is None or not resources.retain
                or torch.device(device).type != "cuda"):
            return self._stream_weights(index, device, dtype)

        placement = (str(torch.device(device)), dtype)
        prefetch_key = (index, *placement)
        hit = resources.prefetch_take(prefetch_key)
        if hit is None:
            hit = self._stream_weights(index, device, dtype)

        next_index = (index + 1) % len(self.branches)
        if self.branches[next_index] is not None:
            next_key = (next_index, *placement)
            resources.prefetch_request(
                next_key,
                lambda i=next_index, d=device, t=dtype: self._stream_weights(i, d, t),
            )
        return hit


def layout_from_payload(payload, x, context, cfg):
    payload = payload or {}
    layout = payload.get("layout")
    video_x = x[0]
    padded = comfy.ldm.common_dit.pad_to_patch_size(video_x, (1, 2, 2))
    latent_t, lat_h, lat_w = padded.shape[2], padded.shape[3], padded.shape[4]
    audio_t = x[1].shape[-1]
    text_len = context.shape[1]
    signature = (text_len, latent_t, lat_h, lat_w, audio_t)
    if layout is None or layout.signature != signature:
        layout = minimax_model.PackedLayout(
            text_len, latent_t, lat_h, lat_w, audio_t,
            keyframes=payload.get("keyframes"), refs=payload.get("refs"))
    video_seg = next(s for s in layout.segments if s[2] == "video")
    text_seg = next(s for s in layout.segments if s[2] == "text")
    tokens_per_frame = (lat_h // 2) * (lat_w // 2)
    return VDNLayout(
        video_seg[0], video_seg[1],
        (video_seg[1] - video_seg[0]) // tokens_per_frame,
        tokens_per_frame, (lat_h // 2, lat_w // 2),
        text_seg[0], text_seg[1] - text_seg[0], layout.seq_len,
        cfg["radius"], cfg["chunk"], cfg["anchor_frames"],
    )


def make_layout_wrapper(state):
    def wrap(executor, *args, **kwargs):
        layout = layout_from_payload(
            kwargs.get("minimax_payload"), args[0], args[2], state.cfg)
        with state.runtime.execution():
            token = state._layout.set(layout)
            _once(
                ("layout", layout.seq_len, layout.num_frames, layout.tokens_per_frame,
                 tuple(layout.bounds), layout.anchor_frames, state.retain_buffers),
                f"layout: seq {layout.seq_len}, video [{layout.video_start}, "
                f"{layout.video_end}), F={layout.num_frames}, S={layout.tokens_per_frame}, "
                f"window={'dense' if layout.full_cover else layout.bounds[0]}, "
                f"buffers={'retained' if state.retain_buffers else 'transient'}",
            )
            try:
                return executor(*args, **kwargs)
            except comfy.model_management.InterruptProcessingException:
                # All persistent scratch belongs to this VDNState, so a cancelled
                # execution can release it without touching another node/model.
                state.runtime.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
            finally:
                state._layout.reset(token)
    return wrap


def _base_attention(attn, x, rope_freqs, transformer_options):
    transformer_options = transformer_options or {}
    s = x.shape[0]
    q, k, v = attn.qkv_proj(x).split(attn.heads * attn.head_dim, dim=-1)
    v = v.view(s, attn.heads, attn.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, attn.heads, attn.head_dim)
        k = k.view(1, s, attn.heads, attn.head_dim)
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
        q, k = q[0], k[0]
    else:
        q = attn.q_norm(q.view(s, attn.heads, attn.head_dim))
        k = attn.k_norm(k.view(s, attn.heads, attn.head_dim))
    q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
    out = optimized_attention(
        q, k, v, attn.heads, mask=None, skip_reshape=True,
        transformer_options=transformer_options)
    return attn.out_proj(out.squeeze(0))


def _external_reduced_sequence_active(transformer_options, layout, sequence_rows, rope_freqs):
    contract = transformer_options.get(VDN_EXTERNAL_SEQUENCE_KEY)
    if isinstance(contract, dict) and contract.get("api") == 2:
        if contract.get("mode") != VDN_EXTERNAL_SEQUENCE_MODE or contract.get("topology") != "mixed_grid_low_suffix":
            raise RuntimeError("VDN external mixed-sequence contract mode/topology is unsupported")
        names = ("native_sequence_rows", "sequence_rows", "video_start", "temporal", "prefix_t",
                 "source_rows_per_frame", "prefix_rows_per_frame")
        if any(type(contract.get(name)) is not int for name in names):
            raise RuntimeError("VDN mixed-sequence counts must be integers")
        native, actual, start, temporal, prefix, source_rows, prefix_rows = (contract[name] for name in names)
        if (native != layout.seq_len or actual != sequence_rows or start != layout.video_start
                or temporal != layout.num_frames or source_rows != layout.tokens_per_frame
                or not 0 < prefix < temporal or not 0 < source_rows < prefix_rows
                or native != start + temporal * source_rows
                or actual != start + prefix * prefix_rows + (temporal - prefix) * source_rows):
            raise RuntimeError("VDN mixed-sequence contract does not match the native layout and current stream")
        if rope_freqs is None or rope_freqs.ndim < 2 or int(rope_freqs.shape[1]) != sequence_rows:
            raise RuntimeError("VDN mixed sequence requires matching explicit RoPE rows")
        return True
    if sequence_rows == layout.seq_len:
        if contract is not None:
            raise RuntimeError("VDN full native sequence received a stale external-sequence contract")
        return False
    if sequence_rows > layout.seq_len:
        raise RuntimeError(
            f"VDN packed sequence length {sequence_rows} exceeds published layout {layout.seq_len}")

    if not isinstance(contract, dict):
        raise RuntimeError(
            "VDN received a reduced packed sequence without an explicit external-sequence contract")
    if int(contract.get("api", -1)) != 1:
        raise RuntimeError("VDN external reduced-sequence contract API is unsupported")
    if contract.get("mode") != VDN_EXTERNAL_SEQUENCE_MODE:
        raise RuntimeError("VDN external reduced-sequence contract mode is unsupported")
    if int(contract.get("full_sequence_rows", -1)) != layout.seq_len:
        raise RuntimeError("VDN external reduced-sequence full-row count does not match the published layout")
    if int(contract.get("reduced_sequence_rows", -1)) != sequence_rows:
        raise RuntimeError("VDN external reduced-sequence row count does not match the current hidden stream")
    if rope_freqs is not None and (rope_freqs.ndim < 2 or int(rope_freqs.shape[1]) != sequence_rows):
        raise RuntimeError("VDN external reduced sequence requires RoPE rows matching the reduced hidden stream")
    return True


def make_vdn_forward(attn, state, block_index):
    heads, head_dim = attn.heads, attn.head_dim
    inner = heads * head_dim
    qkv_proj, out_proj = attn.qkv_proj, attn.out_proj
    q_norm, k_norm = attn.q_norm, attn.k_norm
    base_branch = state.branches[block_index]
    cfg = state.cfg

    def vdn_forward(x, rope_freqs=None, transformer_options=None):
        transformer_options = transformer_options or {}
        layout = state.layout
        if layout is None or base_branch is None:
            return _base_attention(attn, x, rope_freqs, transformer_options)

        # ModelPatcher clones share VDNState. Keep the small lazy backend selector on
        # an execution-local shallow branch copy so simultaneous geometry cannot race.
        branch = copy.copy(base_branch)
        branch._backend = None
        branch._backend_key = None

        s = x.shape[0]
        external_reduced = _external_reduced_sequence_active(
            transformer_options, layout, s, rope_freqs)
        if external_reduced:
            external_kind = (
                "mixed" if transformer_options[VDN_EXTERNAL_SEQUENCE_KEY].get("api") == 2 else "reduced")
            _once(
                ("external-reduced", layout.seq_len, s),
                f"external {external_kind} sequence {s}/{layout.seq_len}: using dense gated "
                "attention with the geometry-dependent linear complement disabled",
            )

        device, dtype = x.device, x.dtype
        q, k, v = qkv_proj(x).split(inner, dim=-1)
        v = v.view(s, heads, head_dim)
        q_raw = q.view(s, heads, head_dim)
        k_raw = k.view(s, heads, head_dim)

        window_active = not layout.full_cover and not external_reduced
        linear_active = window_active and cfg.get("linear_enabled", True)
        q_raw_video = k_raw_video = v_video = None
        text_x = text_k_raw = text_v_raw = None
        if linear_active:
            a, b = layout.video_start, layout.video_end
            resources = state.runtime.current()
            text_rows = layout.text_len if branch.enable_text_state else 0
            scratch = (
                resources.activation_scratch(
                    b - a, text_rows, heads, head_dim, device, dtype)
                if resources is not None else None
            )
            if scratch is None:
                q_raw_video = q_raw[a:b].clone()
                k_raw_video = k_raw[a:b].clone()
                v_video = v[a:b].clone()
            else:
                q_raw_video = scratch["q"]
                k_raw_video = scratch["k"]
                v_video = scratch["v"]
                q_raw_video.copy_(q_raw[a:b])
                k_raw_video.copy_(k_raw[a:b])
                v_video.copy_(v[a:b])

            if branch.enable_text_state and layout.text_len:
                ta, tb = layout.text_start, layout.text_start + layout.text_len
                text_x = x[ta:tb]
                if scratch is None:
                    text_k_raw = k_raw[ta:tb].clone()
                    text_v_raw = v[ta:tb].clone()
                else:
                    text_k_raw = scratch["tk"]
                    text_v_raw = scratch["tv"]
                    text_k_raw.copy_(k_raw[ta:tb])
                    text_v_raw.copy_(v[ta:tb])

        if rope_freqs is not None:
            q4 = q.view(1, s, heads, head_dim)
            k4 = k.view(1, s, heads, head_dim)
            qw = comfy.model_management.cast_to(q_norm.weight, device=device)
            kw = comfy.model_management.cast_to(k_norm.weight, device=device)
            rot = rope_freqs.shape[-3] * 2
            comfy.quant_ops.ck.rms_rope_split_half_(
                q4, k4, rope_freqs, qw, kw, epsilon=q_norm.eps, rot_dim=rot)
            q, k = q4[0], k4[0]
        else:
            q = q_norm(q_raw)
            k = k_norm(k_raw)
        v = v.clone()

        if window_active:
            backend = state.softmax_backend
            if backend == "flex":
                from vdn_h3.window import window_softmax_flex
                try:
                    softmax_out = window_softmax_flex(
                        q, k, v, layout.video_start, layout.video_end,
                        layout.num_frames, layout.tokens_per_frame, layout.bounds,
                        head_dim ** -0.5, anchor_frames=cfg["anchor_frames"])
                except Exception as exc:
                    backend = "grouped"
                    _log.warning(
                        "[vdn] flex attention failed (%s); falling back to grouped SDPA "
                        "for this execution", exc)
            if backend != "flex":
                from vdn_h3.retained import window_softmax_grouped_runtime
                softmax_out = window_softmax_grouped_runtime(
                    q, k, v, layout.video_start, layout.video_end,
                    layout.num_frames, layout.tokens_per_frame, layout.bounds,
                    head_dim ** -0.5, anchor_frames=cfg["anchor_frames"],
                    transformer_options=transformer_options)
        else:
            qc = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
            kc = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
            vc = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
            softmax_out = optimized_attention(
                qc, kc, vc, heads, mask=None, skip_reshape=True,
                transformer_options=transformer_options).squeeze(0).reshape(s, heads, head_dim)

        del q, k, v, q_raw, k_raw
        weights = state.weights_on(block_index, device, dtype)

        if cfg["enable_softmax_gate"]:
            gate = torch.sigmoid(F.linear(
                x, weights["softmax_gate.up.weight"],
                weights["softmax_gate.up.bias"]))
            flat = (
                softmax_out
                * gate.view(s, heads, 1).to(softmax_out.dtype)
            ).reshape(s, -1)
        else:
            flat = softmax_out.reshape(s, -1)
        out = out_proj(flat.type_as(x))
        del softmax_out

        if linear_active:
            readout = branch.readout(
                weights,
                x[layout.video_start:layout.video_end],
                q_raw_video, k_raw_video, v_video,
                layout.num_frames, layout.tokens_per_frame, layout.bounds,
                frame_size=layout.frame_size,
                text_x=text_x, text_k_raw=text_k_raw, text_v_raw=text_v_raw,
                skip_ends=(cfg["anchor_frames"] == "both"),
            )
            out[layout.video_start:layout.video_end] += F.linear(
                readout.type_as(x), weights["to_out_linear.weight"])
        return out

    vdn_forward._vdn_forward = True
    vdn_forward._vdn_external_sequence_api = VDN_EXTERNAL_SEQUENCE_API_VERSION
    return vdn_forward


def apply_vdn(new_model, state):
    dm = new_model.get_model_object("diffusion_model")
    blocks = getattr(dm, "blocks", None)
    if blocks is None or not blocks or not hasattr(getattr(blocks[0], "attn", None), "qkv_proj"):
        raise RuntimeError(
            "ApplyVDNH3 requires ComfyUI MiniMax-H3 blocks[].attn.qkv_proj")
    if len(blocks) != len(state.branches):
        raise RuntimeError(
            f"VDN checkpoint has {len(state.branches)} blocks but the loaded base has "
            f"{len(blocks)}")
    for index, block in enumerate(blocks):
        key = f"diffusion_model.blocks.{index}.attn.forward"
        existing = new_model.object_patches.get(key)
        if existing is not None and not getattr(existing, "_vdn_forward", False):
            raise RuntimeError(
                f"VDN cannot safely replace existing object patch {key}; compose the "
                "other provider through Comfy transformer/model patch APIs instead")
        new_model.add_object_patch(key, make_vdn_forward(block.attn, state, index))
    new_model.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL, "vdn_h3", make_layout_wrapper(state))
