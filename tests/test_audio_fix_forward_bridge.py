from __future__ import annotations

import pytest
import torch

import comfy.model_management as mm
import comfy.ops
import comfy.quant_ops

from vdn_h3.audio_fix_train import comfy_quant_training_mode


def test_rms_rope_bridge_uses_inference_value_and_functional_gradient(monkeypatch):
    calls = {"functional": 0, "inplace": 0}

    def functional(q, k, rope, qw, kw, **kwargs):
        del rope, qw, kw, kwargs
        calls["functional"] += 1
        return q * 2.0, k * 3.0

    def inplace(q, k, rope, qw, kw, **kwargs):
        del rope, qw, kw, kwargs
        calls["inplace"] += 1
        q.add_(10.0)
        k.sub_(5.0)

    monkeypatch.setattr(comfy.quant_ops.ck, "rms_rope_split_half", functional)
    monkeypatch.setattr(comfy.quant_ops.ck, "rms_rope_split_half_", inplace)

    q = torch.randn(2, 3, requires_grad=True)
    k = torch.randn(2, 3, requires_grad=True)
    q_before = q.detach().clone()
    k_before = k.detach().clone()

    with comfy_quant_training_mode():
        q_out, k_out = comfy.quant_ops.ck.rms_rope_split_half(
            q, k, None, None, None
        )
        assert torch.equal(q_out.detach(), q_before + 10.0)
        assert torch.equal(k_out.detach(), k_before - 5.0)
        # The inference-exact bridge must not mutate the actual autograd inputs.
        assert torch.equal(q.detach(), q_before)
        assert torch.equal(k.detach(), k_before)
        (q_out.sum() + k_out.sum()).backward()

    assert calls == {"functional": 1, "inplace": 1}
    assert torch.equal(q.grad, torch.full_like(q, 2.0))
    assert torch.equal(k.grad, torch.full_like(k, 3.0))


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

    with pytest.raises(RuntimeError, match="sentinel"):
        with comfy_quant_training_mode():
            assert mm.in_training is True
            assert mm.training_fp8_bwd is False
            assert comfy.quant_ops.ck.rms_rope_split_half is not functional
            assert comfy.ops.linear_input_act is not linear_input_act
            raise RuntimeError("sentinel")

    assert mm.in_training is old_training
    assert mm.training_fp8_bwd is old_fp8_bwd
    assert comfy.quant_ops.ck.rms_rope_split_half is functional
    assert comfy.quant_ops.ck.rms_rope_split_half_ is inplace
    assert comfy.ops.linear_input_act is linear_input_act
