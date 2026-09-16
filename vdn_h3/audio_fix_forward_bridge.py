"""Inference-exact, memory-bounded forward bridges for INT8 audio-fix training.

Production MiniMax-H3 uses fused/in-place kernels that are either inference-only or
lack a registered autograd formula.  The audio-fix trainer must preserve their exact
forward values while still differentiating through a frozen INT8/ConvRot H3.

The important memory rule is that surrogate graphs are *not* constructed during block
forward.  At production geometry those Q/K and SwiGLU/fc2 graphs are multi-GiB.  If
built while a checkpointed block is being recomputed, they remain live together until
that block's backward and can push a 96-GiB card into WDDM/shared-memory migration.

Instead:

* RMSNorm + split-half RoPE runs the exact production in-place fused kernel on detached
  clones in forward. Its custom backward lazily recomputes the ordinary-PyTorch Q and K
  references one at a time.
* fused INT8 fc2/SwiGLU runs the exact inference helper in forward. Its custom backward
  lazily recomputes only the eager training surrogate required for d(output)/d(input).
* MiniMax-H3's in-place packed-row modulation/residual helpers keep the exact same
  arithmetic but operate on a clone so checkpointed autograd never sees caller-visible
  version-counter mutations.

The patches are process-local and scoped to the standalone training context. Normal
ComfyUI inference is unchanged.
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


def _lazy_input_grad(forward_fn, x: torch.Tensor,
                     grad_output: torch.Tensor | None):
    """Recompute one surrogate graph only for the duration of this operator backward."""
    if grad_output is None:
        return None
    with torch.enable_grad():
        probe = x.detach().requires_grad_(True)
        surrogate = forward_fn(probe)
        (grad_x,) = torch.autograd.grad(
            surrogate,
            probe,
            grad_outputs=grad_output.to(device=surrogate.device, dtype=surrogate.dtype),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
    return grad_x


@contextmanager
def inference_exact_training_primitives():
    """Keep H3 forward production-exact while bounding training activation residency."""
    import comfy.ldm.minimax.model as minimax_model
    import comfy.model_management as mm
    import comfy.ops
    import comfy.quant_ops

    ck = comfy.quant_ops.ck
    functional_rms_rope = ck.rms_rope_split_half
    inplace_rms_rope = ck.rms_rope_split_half_
    linear_input_act = comfy.ops.linear_input_act
    mod_scale_shift = minimax_model._mod_scale_shift
    mod_gate = minimax_model._mod_gate

    def exact_rms_rope_no_grad(q, k, rope, qw, kw, epsilon, rot_dim):
        kw_effective = qw if kw is None else kw
        with torch.no_grad():
            q_exact = q.detach().clone()
            k_exact = k.detach().clone()
            inplace_rms_rope(
                q_exact,
                k_exact,
                rope,
                qw,
                kw_effective,
                epsilon=float(epsilon),
                rot_dim=int(rot_dim),
            )
        return q_exact, k_exact

    class _ExactRmsRope(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, rope, qw, kw, epsilon, rot_dim):
            kw_effective = qw if kw is None else kw
            if rope.requires_grad or qw.requires_grad or kw_effective.requires_grad:
                raise RuntimeError(
                    "audio-fix RMS/RoPE bridge expects frozen rope/norm parameters"
                )
            ctx.save_for_backward(q, k, rope, qw, kw_effective)
            ctx.epsilon = float(epsilon)
            ctx.rot_dim = int(rot_dim)
            return exact_rms_rope_no_grad(
                q, k, rope, qw, kw_effective, ctx.epsilon, ctx.rot_dim
            )

        @staticmethod
        def backward(ctx, grad_q, grad_k):
            q, k, rope, qw, kw = ctx.saved_tensors
            dq = None
            dk = None
            if ctx.needs_input_grad[0] and grad_q is not None:
                dq = _lazy_input_grad(
                    lambda probe: _rms_rope_one_surrogate(
                        probe, rope, qw, ctx.epsilon, ctx.rot_dim
                    ),
                    q,
                    grad_q,
                )
            if ctx.needs_input_grad[1] and grad_k is not None:
                dk = _lazy_input_grad(
                    lambda probe: _rms_rope_one_surrogate(
                        probe, rope, kw, ctx.epsilon, ctx.rot_dim
                    ),
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
                with torch.no_grad():
                    exact = linear_input_act(linear, x.detach(), input_act)
            finally:
                mm.in_training = old_training
            return exact

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors

            def surrogate_forward(probe):
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
                + ", ".join(sorted(kwargs))
            )
        if not torch.is_grad_enabled() or not (q.requires_grad or k.requires_grad):
            return exact_rms_rope_no_grad(q, k, rope, qw, kw, epsilon, rot_dim)
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
