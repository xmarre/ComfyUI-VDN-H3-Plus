"""Small, inference-only helpers for MiniMax-H3 packed-row drift diagnostics.

The helpers in this file do not change model execution.  They only classify the
current authoritative PackedLayout and compare already-produced tensors.  Keeping
this separate from the VDN runtime avoids turning Sol-H3-inspired investigation into
another production attention mode or strength control.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PairMetrics:
    rel_rms: float
    cosine: float
    max_abs: float


def packed_target_ranges(layout) -> dict[str, tuple[int, int]]:
    """Return grouped prefix/target ranges from an authoritative H3 PackedLayout.

    MiniMax-H3 always places the generated target audio segment immediately before
    the generated target video segment.  Everything before target audio is the packed
    conditioning prefix: text plus any keyframe/reference video or audio rows.  The
    function deliberately derives these boundaries from ``layout.segments`` rather
    than guessing from modality names or token counts.
    """
    segments = tuple(layout.segments)
    audio = [segment for segment in segments if segment[2] == "audio"]
    video = [segment for segment in segments if segment[2] == "video"]
    if len(audio) != 1 or len(video) != 1:
        raise RuntimeError(
            "MiniMax-H3 drift diagnostics require exactly one target audio and one "
            f"target video segment, got audio={audio!r} video={video!r}"
        )
    aa, ab, _ = audio[0]
    va, vb, _ = video[0]
    seq_len = int(layout.seq_len)
    if not (0 <= aa < ab == va < vb == seq_len):
        raise RuntimeError(
            "MiniMax-H3 target layout is not [prefix | target audio | target video]: "
            f"audio=[{aa},{ab}) video=[{va},{vb}) seq={seq_len}"
        )
    return {
        "prefix": (0, int(aa)),
        "audio": (int(aa), int(ab)),
        "video": (int(va), int(vb)),
    }


def split_qkv_rows(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Comfy's fused MiniMax-H3 qkv projection output into raw Q/K/V."""
    if tensor.ndim < 2 or tensor.shape[-1] % 3:
        raise ValueError(f"expected fused QKV last dimension divisible by three, got {tuple(tensor.shape)}")
    return tensor.chunk(3, dim=-1)


def pair_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> PairMetrics:
    """Exact aggregate relative-RMS, cosine and max-absolute delta for two tensors."""
    if reference.shape != candidate.shape:
        raise ValueError(
            f"drift tensors must have identical shape, got {tuple(reference.shape)} "
            f"and {tuple(candidate.shape)}"
        )
    ref = reference.detach().to(dtype=torch.float32)
    cand = candidate.detach().to(device=ref.device, dtype=torch.float32)
    if not torch.isfinite(ref).all() or not torch.isfinite(cand).all():
        raise RuntimeError("drift diagnostic received non-finite values")
    diff = cand - ref
    ref_norm = ref.square().mean().sqrt().clamp_min(1e-12)
    rel_rms = diff.square().mean().sqrt() / ref_norm
    dot = (ref * cand).sum()
    denom = ref.square().sum().sqrt() * cand.square().sum().sqrt()
    if float(denom) == 0.0:
        cosine = torch.tensor(1.0 if torch.equal(ref, cand) else 0.0)
    else:
        cosine = dot / denom
    return PairMetrics(
        rel_rms=float(rel_rms),
        cosine=float(cosine),
        max_abs=float(diff.abs().max()) if diff.numel() else 0.0,
    )


def cpu_snapshot(tensor: torch.Tensor) -> torch.Tensor:
    """Compact one-time snapshot for the tiny synthetic full-stack probe geometry."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected Tensor, got {type(tensor).__name__}")
    return tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()


__all__ = [
    "PairMetrics",
    "cpu_snapshot",
    "packed_target_ranges",
    "pair_metrics",
    "split_qkv_rows",
]
