"""Generated-audio and conditioning adapter scoping for MiniMax-H3 VDN bypass."""
from __future__ import annotations

import contextvars
from dataclasses import dataclass

import torch

import comfy.ldm.common_dit
import comfy.ldm.minimax.model as minimax_model


@dataclass(frozen=True)
class AudioAdapterScope:
    conditioning_end: int
    audio_start: int
    audio_end: int
    audio_mod_rows: tuple[int, ...]


_CURRENT_SCOPE = contextvars.ContextVar("vdn_h3_audio_adapter_scope", default=None)


def current_scope():
    return _CURRENT_SCOPE.get()


def enter_scope(scope: AudioAdapterScope):
    return _CURRENT_SCOPE.set(scope)


def exit_scope(token):
    _CURRENT_SCOPE.reset(token)


def scope_from_model_call(dm, args, kwargs) -> AudioAdapterScope:
    if len(args) < 3:
        raise RuntimeError("VDN audio scope expected MiniMax-H3 x/timestep/context arguments")
    x, timestep, context = args[0], args[1], args[2]
    transformer_options = (
        args[3] if len(args) > 3 and isinstance(args[3], dict)
        else kwargs.get("transformer_options", {})
    ) or {}
    payload = kwargs.get("minimax_payload") or {}

    video_x = x[0]
    padded = comfy.ldm.common_dit.pad_to_patch_size(video_x, (1, 2, 2))
    latent_t, lat_h, lat_w = padded.shape[2], padded.shape[3], padded.shape[4]
    audio_t = x[1].shape[-1]
    text_len = context.shape[1]
    layout = payload.get("layout")
    signature = (text_len, latent_t, lat_h, lat_w, audio_t)
    if layout is None or layout.signature != signature:
        layout = minimax_model.PackedLayout(
            text_len, latent_t, lat_h, lat_w, audio_t,
            keyframes=payload.get("keyframes"), refs=payload.get("refs"))
    aa, ab, _ = next(seg for seg in layout.segments if seg[2] == "audio")
    va, _vb, _ = next(seg for seg in layout.segments if seg[2] == "video")
    if ab != va:
        raise RuntimeError(
            "VDN expected target audio immediately before target video in packed H3 layout")

    shift_v = float(transformer_options.get(
        "minimax_h3_sigma_shift_video", getattr(dm, "sigma_shift_video", 12.0)))
    shift_a = float(transformer_options.get(
        "minimax_h3_sigma_shift_audio", getattr(dm, "sigma_shift_audio", 3.0)))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - minimax_model.time_shift_sigma(sigma_v, shift_v, shift_a))

    vis_aug = float(payload.get(
        "visual_cond_noise_aug", getattr(minimax_model, "VISUAL_COND_TIMESTEP", 0.999)))
    aud_aug = float(payload.get(
        "audio_cond_noise_aug", getattr(minimax_model, "AUDIO_COND_TIMESTEP", 1.0)))
    seg_t = {
        "text": t_v, "video": t_v, "audio": t_a,
        "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
        "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug),
    }

    video_rows_t = None
    denoise_mask = kwargs.get("denoise_mask")
    if denoise_mask is not None:
        m = minimax_model.mask_row_values(
            denoise_mask[0, 0].to(torch.float32), latent_t, lat_h, lat_w)
        if m is not None:
            rows_t = (1.0 - m * sigma_v.to(m.device)).clamp(
                max=max(t_v, getattr(minimax_model, "VISUAL_COND_TIMESTEP", 0.999)))
            if rows_t.unique().numel() == 1:
                seg_t["video"] = float(rows_t[0])
            else:
                video_rows_t = rows_t

    audio_rows_t = None
    audio_denoise_mask = kwargs.get("audio_denoise_mask")
    if audio_denoise_mask is not None:
        m = audio_denoise_mask[0, 0].to(torch.float32).reshape(-1)
        if not bool((m >= 1.0 - 1e-3).all()):
            rows_t = (1.0 - m * (1.0 - t_a)).clamp(
                max=max(t_a, getattr(minimax_model, "AUDIO_COND_TIMESTEP", 1.0)))
            if rows_t.unique().numel() == 1:
                seg_t["audio"] = float(rows_t[0])
            else:
                audio_rows_t = rows_t

    unique_t = sorted(
        {t_v, t_a}
        | {seg_t[kind] for _, _, kind in layout.segments}
        | (set(video_rows_t.unique().tolist()) if video_rows_t is not None else set())
        | (set(audio_rows_t.unique().tolist()) if audio_rows_t is not None else set())
    )
    t_row = {value: index for index, value in enumerate(unique_t)}
    audio_values = (
        {seg_t["audio"]} if audio_rows_t is None
        else set(audio_rows_t.unique().tolist())
    )
    return AudioAdapterScope(
        conditioning_end=int(aa),
        audio_start=int(aa),
        audio_end=int(ab),
        audio_mod_rows=tuple(sorted(t_row[value] for value in audio_values)),
    )


def _is_audio_scoped_path(path):
    """Whether this adapter target executes on the packed target stream.

    Token-refiner adapters execute before the packed diffusion-model call and remain
    outside this scope. Sequence-row controls therefore target only ``blocks.*``
    attention/MLP residuals after text/keyframe/reference/target rows have been packed.
    """
    if path.startswith("blocks."):
        return (
            ".attn." in path
            or ".mlp." in path
            or path.endswith(".adaln_proj.linear")
        )
    return path == "final_layer.adaln_proj.linear"


def scale_adapter_delta(
    delta,
    x,
    path,
    audio_strength,
    conditioning_strength=1.0,
):
    """Scale packed conditioning/audio adapter residuals while preserving video rows."""
    audio_strength = float(audio_strength)
    conditioning_strength = float(conditioning_strength)
    if ((audio_strength == 1.0 and conditioning_strength == 1.0)
            or not _is_audio_scoped_path(path)):
        return delta

    scope = current_scope()
    if scope is None:
        raise RuntimeError(
            f"VDN audio-scoped adapter target {path!r} ran outside its model-call scope")

    if path.startswith("blocks.") and (".attn." in path or ".mlp." in path):
        if delta.ndim >= 2 and delta.shape[0] >= scope.audio_end:
            # PackedLayout puts every conditioning/reference row before target audio,
            # followed by target audio and then generated video. Keep those ranges
            # independent so conditioning isolation never weakens generated video.
            if conditioning_strength != 1.0 and scope.conditioning_end > 0:
                delta[:scope.conditioning_end].mul_(conditioning_strength)
            if audio_strength != 1.0 and scope.audio_end > scope.audio_start:
                delta[scope.audio_start:scope.audio_end].mul_(audio_strength)
        return delta

    if path.startswith("blocks.") and path.endswith(".adaln_proj.linear"):
        if delta.ndim != 2 or delta.shape[-1] % 3:
            raise RuntimeError(
                f"Expected 3-modality AdaLN output at {path}, got {tuple(delta.shape)}")
        chunk = delta.shape[-1] // 3
        for row in scope.audio_mod_rows:
            if row < delta.shape[0]:
                delta[row, 2 * chunk:3 * chunk].mul_(audio_strength)
        return delta

    if path == "final_layer.adaln_proj.linear":
        for row in scope.audio_mod_rows:
            if row < delta.shape[0]:
                delta[row].mul_(audio_strength)
    return delta