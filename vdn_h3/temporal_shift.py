"""Single-pass BF16 temporal short-conv for the VDN linear branch.

``branch._temporal_shift`` evaluates the released 5-tap depthwise temporal
convolution as eager shift-multiply-add: one padded copy of the whole feature
volume, then a product and a running sum per tap, each a separate full-volume
BF16 tensor. For a [F, S, C] volume that is roughly 24 volume-sized reads and
writes and three live volume-sized temporaries per feature.

This kernel produces the same BF16 result bit for bit in one pass: each output
element is ``round(...round(round(x[f-2]*w0) + round(x[f-1]*w1))...)`` with
round-to-nearest-even after every multiply and every add, exactly the rounding
sequence of the eager tensor ops (BF16 products of BF16 operands are exact in
FP32 before rounding, and FP32 addition is what PyTorch uses as BF16 opmath).
Zero padding is reproduced with +0 operands, so even signed zeros match. The
equality claim covers finite normal-range values, infinities and overflow;
BF16 subnormals (magnitude below ~1.2e-38) can flush to zero on backends that
flush FP32 denormals, which changes nothing at feature scale. Every
program keeps a sliding window of five tiles in registers and walks the frame
axis, so each input element is read once and each output written once.

Only CUDA BF16 inputs with a 5-tap kernel use it. The first launch in a process
is checked bitwise against the eager path on a small slice; any mismatch,
compile failure or launch failure disables the kernel for the process and the
eager path is used. ``VDN_H3_FUSED_TEMPORAL=0`` disables it.
"""

from __future__ import annotations

import logging
import os
import threading

import torch

try:  # Triton ships with CUDA PyTorch builds; CPU-only installs fall back to eager.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on the installed PyTorch build
    triton = None
    tl = None

_log = logging.getLogger("comfy.vdn")

ENV_FUSED_TEMPORAL = "VDN_H3_FUSED_TEMPORAL"
TAPS = 5
BLOCK_ROWS = 16
BLOCK_CHANNELS = 128

_state_lock = threading.Lock()
_verified = False
_disabled_reason: str | None = None


def eager_temporal_shift(x, w, kernel):
    """The released shift-multiply-add arithmetic (reference for the kernel)."""
    pad = kernel // 2
    xp = torch.nn.functional.pad(x, (0, 0, 0, 0, pad, pad))
    out = None
    for dt in range(kernel):
        part = xp[dt:dt + x.shape[0]] * w[:, dt].view(1, 1, -1)
        out = part if out is None else out + part
    return out


if triton is not None:

    @triton.jit
    def _round_bf16(value):
        """Round FP32 to the nearest BF16 (ties to even), kept in FP32. NaN passes through."""
        bits = value.to(tl.int32, bitcast=True)
        bits = bits + 0x7FFF + ((bits >> 16) & 1)
        bits = (bits >> 16) << 16
        return tl.where(value != value, value, bits.to(tl.float32, bitcast=True))

    @triton.jit
    def _load_frame(x_ptr, frame, frames, frame_elements, offsets, mask):
        valid = (frame >= 0) & (frame < frames)
        safe = tl.minimum(tl.maximum(frame, 0), frames - 1).to(tl.int64)
        values = tl.load(x_ptr + safe * frame_elements + offsets, mask=mask & valid, other=0.0)
        return values.to(tl.float32)

    @triton.jit
    def _temporal_shift_bf16_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        frames,
        rows,
        channels,
        frame_elements,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
        channel = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        channel_mask = channel < channels
        mask = (row[:, None] < rows) & channel_mask[None, :]
        offsets = row[:, None].to(tl.int64) * channels + channel[None, :]
        first = tl.program_id(0) * 0

        w0 = tl.load(w_ptr + channel * 5 + 0, mask=channel_mask, other=0.0).to(tl.float32)[None, :]
        w1 = tl.load(w_ptr + channel * 5 + 1, mask=channel_mask, other=0.0).to(tl.float32)[None, :]
        w2 = tl.load(w_ptr + channel * 5 + 2, mask=channel_mask, other=0.0).to(tl.float32)[None, :]
        w3 = tl.load(w_ptr + channel * 5 + 3, mask=channel_mask, other=0.0).to(tl.float32)[None, :]
        w4 = tl.load(w_ptr + channel * 5 + 4, mask=channel_mask, other=0.0).to(tl.float32)[None, :]

        back2 = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
        back1 = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
        here = _load_frame(x_ptr, first, frames, frame_elements, offsets, mask)
        ahead1 = _load_frame(x_ptr, first + 1, frames, frame_elements, offsets, mask)
        ahead2 = _load_frame(x_ptr, first + 2, frames, frame_elements, offsets, mask)
        frame = first
        for _ in range(0, frames):
            acc = _round_bf16(back2 * w0)
            acc = _round_bf16(acc + _round_bf16(back1 * w1))
            acc = _round_bf16(acc + _round_bf16(here * w2))
            acc = _round_bf16(acc + _round_bf16(ahead1 * w3))
            acc = _round_bf16(acc + _round_bf16(ahead2 * w4))
            tl.store(
                out_ptr + frame.to(tl.int64) * frame_elements + offsets,
                acc.to(out_ptr.dtype.element_ty),
                mask=mask,
            )
            back2 = back1
            back1 = here
            here = ahead1
            ahead1 = ahead2
            ahead2 = _load_frame(x_ptr, frame + 3, frames, frame_elements, offsets, mask)
            frame += 1


def _launch(x, w, out):
    frames, rows, channels = x.shape
    grid = (triton.cdiv(rows, BLOCK_ROWS), triton.cdiv(channels, BLOCK_CHANNELS))
    _temporal_shift_bf16_kernel[grid](
        x, w, out, frames, rows, channels, rows * channels, BLOCK_R=BLOCK_ROWS, BLOCK_C=BLOCK_CHANNELS, num_warps=4,
    )
    return out


def fused_temporal_shift(x, w):
    """Run the kernel unconditionally (x: [F, S, C] BF16 contiguous; w: [C, 5] BF16)."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if x.dim() != 3 or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise RuntimeError("fused temporal shift requires BF16 [F, S, C] features and BF16 [C, 5] taps")
    if tuple(w.shape) != (x.shape[2], TAPS):
        raise RuntimeError("fused temporal shift taps do not match the feature channels")
    x = x.contiguous()
    w = w.contiguous()
    return _launch(x, w, torch.empty_like(x, memory_format=torch.contiguous_format))


def fused_temporal_status():
    return {"verified": _verified, "disabled_reason": _disabled_reason}


def _compiling():
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    return bool(is_compiling()) if callable(is_compiling) else False


def _eligible(x, w, kernel):
    return (
        triton is not None
        and _disabled_reason is None
        and os.environ.get(ENV_FUSED_TEMPORAL, "1").strip().lower() not in {"0", "false", "no", "off"}
        and kernel == TAPS
        and x.is_cuda
        and x.dim() == 3
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and tuple(w.shape) == (x.shape[2], TAPS)
        and x.numel() > 0
        and not _compiling()
    )


def _disable(reason):
    global _disabled_reason
    _disabled_reason = reason
    _log.warning("[vdn] fused temporal short-conv disabled (%s); using the eager path", reason)


def _verify_once(x, w):
    """Bitwise canary on a bounded slice; one host sync per process."""
    global _verified
    with _state_lock:
        if _verified or _disabled_reason is not None:
            return _disabled_reason is None
        probe = x[: min(int(x.shape[0]), 7), : min(int(x.shape[1]), 64)].contiguous()
        expected = eager_temporal_shift(probe, w, TAPS)
        actual = fused_temporal_shift(probe, w)
        if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
            _disable("bitwise canary mismatch against the eager shift-multiply-add")
            return False
        _verified = True
        return True


def temporal_shift(x, w, kernel):
    """Released temporal short-conv; single-pass on CUDA BF16, eager otherwise."""
    if not _eligible(x, w, kernel):
        return eager_temporal_shift(x, w, kernel)
    try:
        if not _verified and not _verify_once(x, w):
            return eager_temporal_shift(x, w, kernel)
        return fused_temporal_shift(x, w)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:
        _disable(f"{type(exc).__name__}: {exc}")
        return eager_temporal_shift(x, w, kernel)


__all__ = [
    "ENV_FUSED_TEMPORAL",
    "eager_temporal_shift",
    "fused_temporal_shift",
    "fused_temporal_status",
    "temporal_shift",
]
