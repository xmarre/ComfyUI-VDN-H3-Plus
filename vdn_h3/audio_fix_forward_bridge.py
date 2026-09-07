"""Inference-exact forward bridges for the standalone INT8 audio-fix trainer.

Comfy's global ``model_management.in_training`` switch changes two MiniMax-H3
forward primitives that materially affect the 50-block trajectory:

* H3 uses functional RMS/RoPE in training but an in-place fused kernel in inference.
* ``linear_input_act`` disables the fused INT8 activation/down-projection path while
  ``in_training`` is true.

The audio-fix trainer needs ``in_training=True`` for autograd-safe H3/VDN execution,
but it must optimize against the production inference forward.  These bridges keep
the exact inference value in the forward pass and attach the supported functional
training path only as the backward surrogate.  The patch is process-local and scoped
to the standalone training context.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


def _straight_through(exact: torch.Tensor, surrogate: torch.Tensor) -> torch.Tensor:
    """Return ``exact`` numerically while differentiating as ``surrogate``."""
    surrogate = surrogate.to(device=exact.device, dtype=exact.dtype)
    return exact + (surrogate - surrogate.detach())


@contextmanager
def inference_exact_training_primitives():
    """Make H3 training forward match production primitives without losing gradients.

    This context assumes the caller owns the process-level Comfy training state.  It
    intentionally does not toggle ``model_management.in_training`` globally except for
    the tiny no-grad call used to obtain the fused ``linear_input_act`` forward value.
    """
    import comfy.model_management as mm
    import comfy.ops
    import comfy.quant_ops

    ck = comfy.quant_ops.ck
    functional_rms_rope = ck.rms_rope_split_half
    inplace_rms_rope = ck.rms_rope_split_half_
    linear_input_act = comfy.ops.linear_input_act

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

    ck.rms_rope_split_half = exact_forward_rms_rope
    comfy.ops.linear_input_act = exact_forward_linear_input_act
    try:
        yield
    finally:
        comfy.ops.linear_input_act = linear_input_act
        ck.rms_rope_split_half = functional_rms_rope
