from __future__ import annotations

import torch
from torch import nn

import comfy.model_management
import comfy.model_patcher

from vdn_h3.apply import _PostForwardLoRA, apply_adapters


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)
        self.use_adaln_curves = False


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = Diffusion()
        self.device = torch.device("cpu")


def _base_patcher():
    torch.manual_seed(10)
    return comfy.model_patcher.ModelPatcher(
        ToyModel(), torch.device("cpu"), torch.device("cpu")
    )


def _converted(seed=20):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 8, generator=gen)
    b = torch.randn(8, 3, generator=gen)
    return {"default": {"linear": (a, b, 1.0)}}, a, b


def test_bypass_apply_uses_injection_and_never_weight_wrappers(monkeypatch):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )
    base = _base_patcher()
    module = base.model.diffusion_model.linear
    true_forward = module.forward
    converted, a, b = _converted()

    vdn = base.clone()
    report = apply_adapters(
        vdn, converted, 0.75, mode="bypass", stage_path=None
    )

    assert "vdn_lora" in vdn.injections
    assert vdn.weight_wrapper_patches == {}
    assert report["default"]["runtime_bypass_targets"] == 1
    assert report["default"]["runtime_weight_targets"] == 1
    runtime = report["runtime_lowvram"]
    assert runtime["mode"] == "post_forward_hook_bypass"
    assert runtime["forward_hooks"] == 1
    assert runtime["pytorch_forward_post_hooks"] == 1
    assert runtime["mutable_forward_wrappers"] == 0
    assert runtime["module_forward_untouched"] is True
    assert runtime["weight_wrappers"] == 0
    assert runtime["bias_wrappers"] == 0
    assert runtime["runtime_preloaded_on_inject"] is True
    assert runtime["managed_adapter_bytes"] == a.numel() * a.element_size() + b.numel() * b.element_size()
    assert runtime["owner_key"] is None
    assert runtime["stack_safe_cross_provider"] is True
    assert runtime["cross_provider_forward_chain_independent"] is True

    x = torch.randn(4, 8)
    base_out = true_forward(x)
    want = base_out + 0.75 * torch.nn.functional.linear(
        torch.nn.functional.linear(x, a), b
    )

    injection = vdn.injections["vdn_lora"][0]
    injection.inject(vdn)
    try:
        assert module.forward == true_forward
        got = module(x)
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)
    finally:
        injection.eject(vdn)

    assert module.forward == true_forward


def test_merge_apply_stays_on_normal_weight_patches():
    base = _base_patcher()
    converted, _, _ = _converted()
    merged = base.clone()

    report = apply_adapters(
        merged, converted, 1.0, mode="merge", stage_path=None
    )

    assert merged.injections == {}
    assert merged.weight_wrapper_patches == {}
    assert "diffusion_model.linear.weight" in merged.patches
    assert report["default"]["native_weight_patches"] == 1
    assert report["default"]["runtime_bypass_targets"] == 0


def test_bypass_no_longer_requires_weight_function_capability(monkeypatch):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )
    # Plain nn.Linear deliberately has no Comfy weight_function list. v1.5.0
    # rejected this class because its bypass path depended on add_weight_wrapper;
    # the VDN runtime path uses a PyTorch post-forward hook instead.
    base = _base_patcher()
    converted, _, _ = _converted()
    vdn = base.clone()
    apply_adapters(vdn, converted, 1.0, mode="bypass", stage_path=None)
    assert "vdn_lora" in vdn.injections
    assert vdn.weight_wrapper_patches == {}



def test_post_forward_bypass_reuses_fresh_output_storage_in_inference():
    torch.manual_seed(31)
    x = torch.randn(9, 8)
    output = torch.randn(9, 12)
    original = output.clone()
    down = torch.randn(3, 8)
    up = torch.randn(12, 3)
    plan = _PostForwardLoRA(((down, up, 0.75),))
    ptr = output.data_ptr()

    with torch.no_grad():
        got = plan(nn.Identity(), (x,), output)

    want = original + 0.75 * torch.nn.functional.linear(
        torch.nn.functional.linear(x, down), up
    )
    assert got is output
    assert got.data_ptr() == ptr
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)


def test_post_forward_bypass_keeps_out_of_place_autograd_semantics():
    torch.manual_seed(32)
    x = torch.randn(4, 8, requires_grad=True)
    output = torch.randn(4, 12, requires_grad=True)
    down = torch.randn(3, 8)
    up = torch.randn(12, 3)
    plan = _PostForwardLoRA(((down, up, 0.5),))
    ptr = output.data_ptr()

    got = plan(nn.Identity(), (x,), output)
    want = output + 0.5 * torch.nn.functional.linear(
        torch.nn.functional.linear(x, down), up
    )
    assert got.data_ptr() != ptr
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)
    got.sum().backward()
    assert x.grad is not None
    assert output.grad is not None
