from __future__ import annotations

import pytest
import torch

import comfy.model_management as mm
import comfy.ops
import comfy.quant_ops
import comfy.ldm.minimax.model as minimax_model

from vdn_h3.audio_fix_forward_bridge import _rms_rope_split_half_surrogate
from vdn_h3.audio_fix_train import comfy_quant_training_mode


def _identity_split_half_rope(seq: int, rot_dim: int, dtype=torch.float32):
    eye = torch.eye(2, dtype=dtype)
    return eye.view(1, 1, 1, 1, 2, 2).expand(
        1, int(seq), 1, int(rot_dim) // 2, 2, 2
    ).contiguous()


def test_rms_rope_bridge_uses_inference_value_and_pytorch_surrogate_gradient(monkeypatch):
    calls = {"functional": 0, "inplace": 0}

    def unsupported_functional(*args, **kwargs):
        del args, kwargs
        calls["functional"] += 1
        raise RuntimeError("custom-op autograd path must not be used")

    def inplace(q, k, rope, qw, kw, **kwargs):
        del rope, qw, kw, kwargs
        calls["inplace"] += 1
        q.add_(10.0)
        k.sub_(5.0)

    monkeypatch.setattr(
        comfy.quant_ops.ck, "rms_rope_split_half", unsupported_functional
    )
    monkeypatch.setattr(comfy.quant_ops.ck, "rms_rope_split_half_", inplace)

    seq, heads, width, rot_dim = 3, 2, 8, 6
    rope = _identity_split_half_rope(seq, rot_dim)
    q_scale = torch.linspace(0.8, 1.2, width)
    k_scale = torch.linspace(1.1, 0.7, width)
    q = torch.randn(1, seq, heads, width, requires_grad=True)
    k = torch.randn(1, seq, heads, width, requires_grad=True)
    q_before = q.detach().clone()
    k_before = k.detach().clone()

    # Compute the gradient oracle from the local ordinary-PyTorch reference itself.
    q_ref = q_before.clone().requires_grad_(True)
    k_ref = k_before.clone().requires_grad_(True)
    q_sur, k_sur = _rms_rope_split_half_surrogate(
        q_ref,
        k_ref,
        rope,
        q_scale,
        k_scale,
        epsilon=1e-6,
        rot_dim=rot_dim,
    )
    (q_sur.sum() + k_sur.sum()).backward()
    q_grad_ref = q_ref.grad.detach().clone()
    k_grad_ref = k_ref.grad.detach().clone()

    with comfy_quant_training_mode():
        q_out, k_out = comfy.quant_ops.ck.rms_rope_split_half(
            q,
            k,
            rope,
            q_scale,
            k_scale,
            epsilon=1e-6,
            rot_dim=rot_dim,
        )
        assert torch.equal(q_out.detach(), q_before + 10.0)
        assert torch.equal(k_out.detach(), k_before - 5.0)
        # The inference-exact bridge must not mutate the actual autograd inputs.
        assert torch.equal(q.detach(), q_before)
        assert torch.equal(k.detach(), k_before)
        (q_out.sum() + k_out.sum()).backward()

    # The original comfy-kitchen functional custom op is deliberately never called;
    # its torch.library operator currently has no registered autograd formula.
    assert calls == {"functional": 0, "inplace": 1}
    assert torch.allclose(q.grad, q_grad_ref)
    assert torch.allclose(k.grad, k_grad_ref)
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()


def test_linear_input_act_bridge_uses_fused_value_and_training_gradient(monkeypatch):
    calls = {"inference": 0, "training": 0}

    def fake_linear_input_act(linear, x, input_act):
        del linear, input_act
        if mm.in_training:
            calls["training"] += 1
            return x * 3.0
        calls["inference"] += 1
        return x * 10.0 + 7.0

    monkeypatch.setattr(comfy.ops, "linear_input_act", fake_linear_input_act)
    x = torch.randn(4, 5, requires_grad=True)

    with comfy_quant_training_mode():
        assert mm.in_training is True
        out = comfy.ops.linear_input_act(object(), x, "swiglu")
        assert torch.equal(out.detach(), x.detach() * 10.0 + 7.0)
        out.sum().backward()

    assert calls == {"inference": 1, "training": 1}
    assert torch.equal(x.grad, torch.full_like(x, 3.0))


def test_h3_modulation_bridge_preserves_arithmetic_without_input_aliasing():
    x = torch.randn(8, 4, dtype=torch.float32, requires_grad=True)
    other = torch.randn(8, 4, dtype=torch.float32, requires_grad=True)
    scale = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    shift = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    gate = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    segments = ((0, 2, 0), (2, 5, 1), (5, 8, 2))

    x_before = x.detach().clone()
    expected_scaled = minimax_model._mod_scale_shift(
        x_before.clone(), shift.detach(), scale.detach(), segments)
    expected_gated = minimax_model._mod_gate(
        x_before.clone(), gate.detach(), other.detach(), segments)

    before_version = x._version
    original_scale_shift = minimax_model._mod_scale_shift
    original_gate = minimax_model._mod_gate
    with comfy_quant_training_mode():
        assert minimax_model._mod_scale_shift is not original_scale_shift
        assert minimax_model._mod_gate is not original_gate
        scaled = minimax_model._mod_scale_shift(x, shift, scale, segments)
        gated = minimax_model._mod_gate(x, gate, other, segments)
        assert x._version == before_version
        assert torch.equal(x.detach(), x_before)
        assert torch.equal(scaled.detach(), expected_scaled)
        assert torch.equal(gated.detach(), expected_gated)
        (scaled.square().mean() + gated.square().mean()).backward()

    assert minimax_model._mod_scale_shift is original_scale_shift
    assert minimax_model._mod_gate is original_gate
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert other.grad is not None and torch.isfinite(other.grad).all()
    assert scale.grad is not None and torch.isfinite(scale.grad).all()
    assert shift.grad is not None and torch.isfinite(shift.grad).all()
    assert gate.grad is not None and torch.isfinite(gate.grad).all()


def test_training_bridge_restores_global_primitives_and_flags_on_exception(monkeypatch):
    def functional(q, k, rope, qw, kw, **kwargs):
        del rope, qw, kw, kwargs
        return q, k

    def inplace(q, k, rope, qw, kw, **kwargs):
        del q, k, rope, qw, kw, kwargs

    def linear_input_act(linear, x, input_act):
        del linear, input_act
        return x

    monkeypatch.setattr(comfy.quant_ops.ck, "rms_rope_split_half", functional)
    monkeypatch.setattr(comfy.quant_ops.ck, "rms_rope_split_half_", inplace)
    monkeypatch.setattr(comfy.ops, "linear_input_act", linear_input_act)

    old_training = mm.in_training
    old_fp8_bwd = mm.training_fp8_bwd
    old_mod_scale_shift = minimax_model._mod_scale_shift
    old_mod_gate = minimax_model._mod_gate

    with pytest.raises(RuntimeError, match="sentinel"):
        with comfy_quant_training_mode():
            assert mm.in_training is True
            assert mm.training_fp8_bwd is False
            assert comfy.quant_ops.ck.rms_rope_split_half is not functional
            assert comfy.ops.linear_input_act is not linear_input_act
            assert minimax_model._mod_scale_shift is not old_mod_scale_shift
            assert minimax_model._mod_gate is not old_mod_gate
            raise RuntimeError("sentinel")

    assert mm.in_training is old_training
    assert mm.training_fp8_bwd is old_fp8_bwd
    assert comfy.quant_ops.ck.rms_rope_split_half is functional
    assert comfy.quant_ops.ck.rms_rope_split_half_ is inplace
    assert comfy.ops.linear_input_act is linear_input_act
    assert minimax_model._mod_scale_shift is old_mod_scale_shift
    assert minimax_model._mod_gate is old_mod_gate
