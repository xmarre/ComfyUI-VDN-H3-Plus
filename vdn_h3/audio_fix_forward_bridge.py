"""Inference-exact forward bridges for the standalone INT8 audio-fix trainer.

Comfy's global ``model_management.in_training`` switch changes two MiniMax-H3
forward primitives that materially affect the 50-block trajectory:

* H3 uses functional RMS/RoPE in training but an in-place fused kernel in inference.
* ``linear_input_act`` disables the fused INT8 activation/down-projection path while
  ``in_training`` is true.

Current MiniMax-H3 also deliberately uses in-place packed-row modulation/residual
updates in ``_mod_scale_shift`` and ``_mod_gate``.  Those are excellent inference
optimizations, but checkpointed autograd cannot allow the saved block inputs / norm
outputs to be mutated after they have been captured for backward.

The audio-fix trainer therefore needs both numerical and aliasing bridges:

* exact production forward values for RMS/RoPE and fused INT8 fc2;
* the exact same H3 modulation arithmetic, but applied to a clone so the caller's
  autograd-visible tensor is not modified in place.

All patches are process-local and scoped to the standalone training context. Normal
ComfyUI inference is untouched.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


def _straight_through(exact: torch.Tensor, surrogate: torch.Tensor) -> torch.Tensor:
    """Return ``exact`` numerically while differentiating as ``surrogate``."""
    surrogate = surrogate.to(device=exact.device, dtype=exact.dtype)
    return exact + (surrogate - surrogate.detach())


def _clone_then_inplace(fn):
    """Preserve an inference in-place kernel's arithmetic without mutating its input."""
    def wrapped(x, *args, **kwargs):
        return fn(x.clone(), *args, **kwargs)
    return wrapped


@contextmanager
def inference_exact_training_primitives():
    """Make H3 training forward production-exact while keeping autograd valid.

    This context assumes the caller owns the process-level Comfy training state.  It
    intentionally does not toggle ``model_management.in_training`` globally except for
    the tiny no-grad call used to obtain the fused ``linear_input_act`` forward value.
    """
    import comfy.model_management as mm
    import comfy.ops
    import comfy.quant_ops
    import comfy.ldm.minimax.model as minimax_model

    ck = comfy.quant_ops.ck
    functional_rms_rope = ck.rms_rope_split_half
    inplace_rms_rope = ck.rms_rope_split_half_
    linear_input_act = comfy.ops.linear_input_act
    mod_scale_shift = minimax_model._mod_scale_shift
    mod_gate = minimax_model._mod_gate

    def exact_forward_rms_rope(q, k, rope, qw, kw, **kwargs):
        # Build the supported differentiable graph first.  The production fused kernel
        # is then evaluated on detached clones so it cannot mutate q/k or enter autograd.
        q_surrogate, k_surrogate = functional_rms_rope(
            q, k, rope, qw, kw, **kwargs
        )
        with torch.no_grad():
            q_exact = q.detach().clone()
            k_exact = k.detach().clone()
            inplace_rms_rope(q_exact, k_exact, rope, qw, kw, **kwargs)
        return (
            _straight_through(q_exact, q_surrogate),
            _straight_through(k_exact, k_surrogate),
        )

    def exact_forward_linear_input_act(linear, x, input_act):
        # Production H3 fuses SwiGLU into the INT8 fc2 kernel only when the global
        # training flag is false.  Evaluate that exact path without autograd, then use
        # Comfy's normal training/eager path solely to carry d(output)/d(input).
        old_training = mm.in_training
        try:
            mm.in_training = False
            with torch.no_grad():
                exact = linear_input_act(linear, x.detach(), input_act)
        finally:
            mm.in_training = old_training

        if not torch.is_grad_enabled() or not x.requires_grad:
            return exact

        surrogate = linear_input_act(linear, x, input_act)
        return _straight_through(exact, surrogate)

    # H3's production helpers intentionally mutate their first argument segment by
    # segment.  With three packed modality segments, _mod_scale_shift bumps the saved
    # tensor version six times (mul_ + add_ for each segment), which breaks checkpoint
    # backward. Clone first, then execute the original production helper unchanged on
    # that clone.  This changes aliasing only; the arithmetic and result values are the
    # same as inference for the same input.
    minimax_model._mod_scale_shift = _clone_then_inplace(mod_scale_shift)
    minimax_model._mod_gate = _clone_then_inplace(mod_gate)
    ck.rms_rope_split_half = exact_forward_rms_rope
    comfy.ops.linear_input_act = exact_forward_linear_input_act
    try:
        yield
    finally:
        comfy.ops.linear_input_act = linear_input_act
        ck.rms_rope_split_half = functional_rms_rope
        minimax_model._mod_gate = mod_gate
        minimax_model._mod_scale_shift = mod_scale_shift
