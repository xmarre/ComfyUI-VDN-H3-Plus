"""Lifecycle regressions for VDN's non-mutating runtime bypass.

VDN must never replace ``module.forward``. Its low-residency LoRA residuals use
PyTorch forward post-hooks whose handles are owned by one PatcherInjection.
ModelPatcher clones share the inner model, so a new VDN generation replaces the
old registered handles and stale clone ejection cannot remove the newer set.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import comfy.model_management
import comfy.model_patcher
from comfy.weight_adapter.bypass import BypassForwardHook
from comfy.weight_adapter.lora import LoRAAdapter

from vdn_h3.apply import apply_adapters


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)
        self.use_adaln_curves = False


class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = Diffusion()
        self.device = torch.device("cpu")


def _base():
    torch.manual_seed(901)
    return comfy.model_patcher.ModelPatcher(
        Root(), torch.device("cpu"), torch.device("cpu")
    )


def _term(seed, rank=3):
    gen = torch.Generator().manual_seed(seed)
    down = torch.randn(rank, 8, generator=gen)
    up = torch.randn(8, rank, generator=gen)
    return down, up


def _apply(base, down, up, strength=1.0):
    patcher = base.clone()
    report = apply_adapters(
        patcher,
        {"default": {"linear": (down, up, 1.0)}},
        strength,
        mode="bypass",
        stage_path=None,
    )
    return patcher, report


def _delta(x, down, up, strength=1.0):
    return strength * torch.nn.functional.linear(
        torch.nn.functional.linear(x, down), up
    )


def test_vdn_bypass_never_replaces_module_forward(monkeypatch):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )
    base = _base()
    module = base.model.diffusion_model.linear
    true_forward = module.forward
    down, up = _term(902)
    vdn, report = _apply(base, down, up, 0.75)
    injection = vdn.injections["vdn_lora"][0]
    x = torch.randn(4, 8)
    want = true_forward(x) + _delta(x, down, up, 0.75)

    assert report["runtime_bypass"]["mode"] == "post_forward_hook_bypass"
    assert report["runtime_bypass"]["mutable_forward_wrappers"] == 0
    assert report["runtime_bypass"]["module_forward_untouched"] is True
    assert module.forward == true_forward

    injection.inject(vdn)
    try:
        assert module.forward == true_forward
        assert torch.allclose(module(x), want, atol=1e-5, rtol=1e-5)
    finally:
        injection.eject(vdn)
    assert module.forward == true_forward


def test_clone_replacement_and_stale_eject_do_not_accumulate(monkeypatch):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )
    base = _base()
    module = base.model.diffusion_model.linear
    true_forward = module.forward
    down1, up1 = _term(903)
    down2, up2 = _term(904)
    first, _ = _apply(base, down1, up1, 1.0)
    second, _ = _apply(base, down2, up2, 0.5)
    inj1 = first.injections["vdn_lora"][0]
    inj2 = second.injections["vdn_lora"][0]
    x = torch.randn(3, 8)

    inj1.inject(first)
    assert module.forward == true_forward
    assert torch.allclose(
        module(x), true_forward(x) + _delta(x, down1, up1), atol=1e-5
    )

    # Same shared model, newer clone: this must replace, not stack.
    inj2.inject(second)
    want2 = true_forward(x) + _delta(x, down2, up2, 0.5)
    assert module.forward == true_forward
    assert torch.allclose(module(x), want2, atol=1e-5)

    # Ejecting the stale generation must not tear down the current one.
    inj1.eject(first)
    assert module.forward == true_forward
    assert torch.allclose(module(x), want2, atol=1e-5)

    inj2.eject(second)
    assert module.forward == true_forward
    assert torch.allclose(module(x), true_forward(x), atol=1e-6)


def _external_hook(module, down, up, strength=1.0):
    # Core Comfy LoRA bypass hook: this deliberately DOES mutate module.forward.
    # VDN must coexist without becoming part of this linked list.
    alpha = torch.tensor(float(down.shape[0]))
    adapter = LoRAAdapter(set(), (up, down, alpha, None, None, None))
    return BypassForwardHook(module, adapter, multiplier=float(strength))


def _run_cross_provider(monkeypatch, external_first):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )
    base = _base()
    module = base.model.diffusion_model.linear
    true_forward = module.forward
    vd, vu = _term(905)
    ed, eu = _term(906)
    vdn, _ = _apply(base, vd, vu, 0.6)
    injection = vdn.injections["vdn_lora"][0]
    external = _external_hook(module, ed, eu, 0.4)
    x = torch.randn(2, 8)
    want = (
        true_forward(x)
        + _delta(x, ed, eu, 0.4)
        + _delta(x, vd, vu, 0.6)
    )

    if external_first:
        external.inject()
        external_forward = module.forward
        injection.inject(vdn)
    else:
        injection.inject(vdn)
        assert module.forward == true_forward
        external.inject()
        external_forward = module.forward

    try:
        # VDN registration never changes the external provider's forward object.
        assert module.forward == external_forward
        assert torch.allclose(module(x), want, atol=1e-5, rtol=1e-5)

        injection.eject(vdn)
        # External provider is still live and exact after VDN removal.
        assert module.forward == external_forward
        external_only = true_forward(x) + _delta(x, ed, eu, 0.4)
        assert torch.allclose(module(x), external_only, atol=1e-5, rtol=1e-5)
    finally:
        # idempotent VDN stale eject is harmless
        injection.eject(vdn)
        external.eject()

    assert module.forward == true_forward


def test_cross_provider_external_first(monkeypatch):
    _run_cross_provider(monkeypatch, external_first=True)


def test_cross_provider_vdn_first(monkeypatch):
    _run_cross_provider(monkeypatch, external_first=False)


def test_repeated_pseudo_continuum_chunks_do_not_accumulate(monkeypatch):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )
    base = _base()
    module = base.model.diffusion_model.linear
    true_forward = module.forward
    down, up = _term(907)
    x = torch.randn(3, 8)
    want = true_forward(x) + _delta(x, down, up)

    for chunk in range(12):
        vdn, _ = _apply(base, down, up, 1.0)
        injection = vdn.injections["vdn_lora"][0]
        injection.inject(vdn)
        try:
            assert module.forward == true_forward, chunk
            assert torch.allclose(module(x), want, atol=1e-5), chunk
        finally:
            injection.eject(vdn)
        assert module.forward == true_forward, chunk
        assert torch.allclose(module(x), true_forward(x), atol=1e-6), chunk


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_large_post_forward_delta_is_tiled_without_a_full_residual(monkeypatch, dtype):
    from vdn_h3 import apply
    down = torch.randn(5, 8, generator=torch.Generator().manual_seed(1001)).to(dtype)
    up = torch.randn(16, 5, generator=torch.Generator().manual_seed(1002)).to(dtype)
    bias = torch.randn(16, generator=torch.Generator().manual_seed(1003)).to(dtype)
    x = torch.randn(2, 11, 8, generator=torch.Generator().manual_seed(1004)).to(dtype)
    weight = torch.randn(16, 8, generator=torch.Generator().manual_seed(1005)).to(dtype)
    base = torch.nn.functional.linear(x, weight)
    expected = base + (_delta(x, down, up) + bias)
    plan = apply._PostForwardLoRA(((down, up, 1.0),), ((bias, 1.0),))
    module = nn.Linear(8, 16, bias=False, dtype=dtype)
    module.weight.data.copy_(weight)
    handle = module.register_forward_hook(plan)
    monkeypatch.setattr(apply, "_POST_FORWARD_DELTA_TILE_BYTES", 256)
    linear = torch.nn.functional.linear
    shapes = []

    def inspected(inputs, weights, bias=None):
        if weights.shape in (down.shape, up.shape):
            shapes.append(tuple(inputs.shape))
        return linear(inputs, weights, bias)

    monkeypatch.setattr(apply.F, "linear", inspected)
    try:
        with torch.no_grad():
            actual = module(x)
    finally:
        handle.remove()
    torch.testing.assert_close(actual, expected, rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
                               atol=0.05 if dtype == torch.bfloat16 else 1e-5)
    # The tile budget covers all batch dimensions and output channels.
    # Every pair of projections is restricted to at most 2 token rows.
    assert len(shapes) > 2
    assert max(shape[-2] for shape in shapes) <= (4 if dtype == torch.bfloat16 else 2)


def test_post_forward_large_delta_oom_retries_only_uncommitted_rows(monkeypatch):
    from vdn_h3 import apply
    down, up = _term(1006)
    x = torch.randn(1, 13, 8, generator=torch.Generator().manual_seed(1007))
    base = torch.randn(1, 13, 8, generator=torch.Generator().manual_seed(1008))
    expected = base + _delta(x, down, up)
    plan = apply._PostForwardLoRA(((down, up, 1.0),))
    module = nn.Linear(8, 8, bias=False)
    handle = module.register_forward_hook(plan)
    monkeypatch.setattr(apply, "_POST_FORWARD_DELTA_TILE_BYTES", 128)
    linear = torch.nn.functional.linear
    retries = []

    def synthetic_allocator_pressure(inputs, weights, bias=None):
        if inputs.shape[-2] > 2:
            retries.append(inputs.shape[-2])
            raise torch.OutOfMemoryError("synthetic tile workspace failure")
        return linear(inputs, weights, bias)

    monkeypatch.setattr(apply.F, "linear", synthetic_allocator_pressure)
    try:
        with torch.no_grad():
            result = plan(module, (x,), base)
    finally:
        handle.remove()
    assert retries
    assert result is base
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_post_forward_gradient_mode_keeps_functional_result(monkeypatch):
    from vdn_h3 import apply
    down, up = _term(1009)
    plan = apply._PostForwardLoRA(((down, up, 1.0),))
    monkeypatch.setattr(apply, "_POST_FORWARD_DELTA_TILE_BYTES", 8)
    x = torch.randn(12, 8, requires_grad=True)
    output = x @ torch.randn(8, 8)
    original = output.clone()
    result = plan(None, (x,), output)
    assert result is not output
    assert torch.equal(output, original)
    result.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
