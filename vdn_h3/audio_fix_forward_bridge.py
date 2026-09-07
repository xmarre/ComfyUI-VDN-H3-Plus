"""Inference-exact forward bridges for the standalone INT8 audio-fix trainer.

The production MiniMax-H3 path uses fused/in-place kernels that either have no
registered autograd formula or are unsuitable for checkpointed training.  The
standalone audio-fix trainer keeps production forward values, but supplies explicit
training-only backward paths.

Two large operators need special handling at production geometry:

* RMSNorm + split-half RoPE: comfy-kitchen's functional custom op has no autograd
  formula.  The exact production in-place kernel is evaluated in forward; backward
  recomputes the ordinary-PyTorch reference lazily, one Q/K tensor at a time.
* fused INT8 fc2/SwiGLU: the exact inference kernel is evaluated in forward; backward
  lazily recomputes only the eager training surrogate when its input gradient is
  actually requested.

Deferring those surrogate graphs to operator backward is important.  Building them
while a checkpointed H3 block is being recomputed keeps several multi-GiB activation
graphs alive simultaneously and can exhaust a 96-GiB card near the end of backward.

MiniMax-H3's in-place packed-row modulation helpers are also functionalized by cloning
their first argument before executing the original arithmetic.  Normal ComfyUI
inference is untouched; every patch in this module is scoped to the standalone trainer.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F


def _clone_then_inplace(fn):
    """Preserve an inference in-place helper's arithmetic without mutating its input."""
    def wrapped(x, *args, **kwargs):
        return fn(x.clone(), *args, **kwargs)
    return wrapped


def _apply_rope_split_half_surrogate(x: torch.Tensor,
                                     freqs_cis: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch split-half RoPE matching comfy-kitchen eager semantics."""
    t = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
    t = t.to(freqs_cis.dtype)
    out = freqs_cis[..., 0] * t[..., 0] + freqs_cis[..., 1] * t[..., 1]
    return out.movedim(-1, -2).reshape(*x.shape).type_as(x)


def _rms_rope_one_surrogate(x: torch.Tensor,
                            freqs_cis: torch.Tensor,
                            scale: torch.Tensor,
                            epsilon: float = 1e-6,
                            rot_dim: int = 0) -> torch.Tensor:
    """Ordinary-PyTorch RMSNorm + partial split-half RoPE for one Q/K tensor."""
    x_norm = F.rms_norm(
        x,
        (x.shape[-1],),
        weight=scale,
        eps=float(epsilon),
    )
    rd = int(rot_dim)
    if rd and rd != x.shape[-1]:
        if rd <= 0 or rd > x.shape[-1] or rd % 2:
            raise ValueError(
                f"invalid split-half rotary dimension {rd} for width {x.shape[-1]}")
        rotated = _apply_rope_split_half_surrogate(x_norm[..., :rd], freqs_cis)
        return torch.cat((rotated, x_norm[..., rd:]), dim=-1)
    return _apply_rope_split_half_surrogate(x_norm, freqs_cis)


def _rms_rope_split_half_surrogate(q: torch.Tensor,
                                   k: torch.Tensor,
                                   freqs_cis: torch.Tensor,
                                   q_scale: torch.Tensor,
                                   k_scale: torch.Tensor | None = None,
                                   epsilon: float = 1e-6,
                                   rot_dim: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference helper retained for parity tests; the production bridge is lazy."""
    if k_scale is None:
        k_scale = q_scale
    return (
        _rms_rope_one_surrogate(q, freqs_cis, q_scale, epsilon, rot_dim),
        _rms_rope_one_surrogate(k, freqs_cis, k_scale, epsilon, rot_dim),
    )


def _lazy_input_grad(forward_fn, x: torch.Tensor, grad_output: torch.Tensor | None):
    """Recompute one surrogate only when its input gradient is requested."""
    if grad_output is None:
        return None
    with torch.enable_grad():
        probe = x.detach().requires_grad_(True)
        surrogate = forward_fn(probe)
        (grad_x,) = torch.autograd.grad(
            surrogate,
            probe,
            grad_outputs=grad_output.to(dtype=surrogate.dtype),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
    return grad_x


@contextmanager
def inference_exact_training_primitives():
    """Make H3 training forward production-exact while keeping autograd valid."""
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

    class _ExactRmsRope(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, rope, qw, kw, epsilon, rot_dim):
            kw_effective = qw if kw is None else kw
            if rope.requires_grad or qw.requires_grad or kw_effective.requires_grad:
                raise RuntimeError(
                    "audio-fix RMS/RoPE bridge expects frozen rope/norm parameters")
            ctx.save_for_backward(q, k, rope, qw, kw_effective)
            ctx.epsilon = float(epsilon)
            ctx.rot_dim = int(rot_dim)

            # Autograd.Function.forward already executes without recording a graph.
            # Clone because the production kernel mutates Q/K in place.
            q_exact = q.detach().clone()
            k_exact = k.detach().clone()
            inplace_rms_rope(
                q_exact,
                k_exact,
                rope,
                qw,
                kw_effective,
                epsilon=ctx.epsilon,
                rot_dim=ctx.rot_dim,
            )
            return q_exact, k_exact

        @staticmethod
        def backward(ctx, grad_q, grad_k):
            q, k, rope, qw, kw = ctx.saved_tensors
            # Recompute Q and K separately.  The old straight-through implementation
            # built both large surrogate graphs during checkpoint recomputation and
            # kept them alive together until block backward.
            dq = _lazy_input_grad(
                lambda probe: _rms_rope_one_surrogate(
                    probe, rope, qw, ctx.epsilon, ctx.rot_dim),
                q,
                grad_q,
            )
            dk = _lazy_input_grad(
                lambda probe: _rms_rope_one_surrogate(
                    probe, rope, kw, ctx.epsilon, ctx.rot_dim),
                k,
                grad_k,
            )
            return dq, dk, None, None, None, None, None

    class _ExactLinearInputAct(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, linear, input_act):
            ctx.linear = linear
            ctx.input_act = input_act
            ctx.save_for_backward(x)
            old_training = mm.in_training
            try:
                mm.in_training = False
                exact = linear_input_act(linear, x.detach(), input_act)
            finally:
                mm.in_training = old_training
            return exact

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors

            def surrogate_forward(probe):
                # The original helper dispatches the normal eager/training path while
                # the global training flag is true.  Keep that graph local to this
                # operator backward instead of retaining it for the whole H3 block.
                old_training = mm.in_training
                try:
                    mm.in_training = True
                    return linear_input_act(ctx.linear, probe, ctx.input_act)
                finally:
                    mm.in_training = old_training

            dx = _lazy_input_grad(surrogate_forward, x, grad_output)
            return dx, None, None

    def exact_forward_rms_rope(q, k, rope, qw, kw=None, **kwargs):
        epsilon = float(kwargs.pop("epsilon", 1e-6))
        rot_dim = int(kwargs.pop("rot_dim", 0))
        if kwargs:
            raise TypeError(
                "unexpected RMS/RoPE training bridge kwargs: "
                + ", ".join(sorted(kwargs)))
        return _ExactRmsRope.apply(q, k, rope, qw, kw, epsilon, rot_dim)

    def exact_forward_linear_input_act(linear, x, input_act):
        if not torch.is_grad_enabled() or not x.requires_grad:
            old_training = mm.in_training
            try:
                mm.in_training = False
                with torch.no_grad():
                    return linear_input_act(linear, x.detach(), input_act)
            finally:
                mm.in_training = old_training
        return _ExactLinearInputAct.apply(x, linear, input_act)

    # Preserve production modulation arithmetic, but eliminate caller-visible aliases.
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


__all__ = [
    "inference_exact_training_primitives",
]
