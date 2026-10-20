"""Large residuals must preserve tensors retained by other providers."""
import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.module import register_module_forward_hook
from torch.utils._python_dispatch import TorchDispatchMode

import comfy.ops

from vdn_h3 import apply


def setup(monkeypatch, module=None):
    monkeypatch.setattr(apply, "_POST_FORWARD_DELTA_TILE_BYTES", 64)
    module = nn.Linear(4, 6) if module is None else module
    plan = apply._PostForwardLoRA(((torch.randn(2, 4), torch.randn(6, 2), 0.5),))
    x = torch.randn(2, 19, 4)
    return module, plan, x


@pytest.mark.parametrize("kind", ["earlier", "later", "kwargs", "global"])
def test_competing_hooks_preserve_captured_activation(monkeypatch, kind):
    module, plan, x = setup(monkeypatch)
    captured = []
    handles = []
    with torch.no_grad():
        base = module(x)
        down, up, _ = plan._weights_for(x)
        expected = base + F.linear(F.linear(x, down), up)

    def capture(_module, _inputs, output):
        captured.append((output, output.clone()))

    if kind == "later":
        handles.append(module.register_forward_hook(plan))
    if kind == "global":
        handles.append(register_module_forward_hook(capture))
    elif kind == "kwargs":
        handles.append(module.register_forward_hook(
            lambda m, args, kwargs, out: capture(m, args, out), with_kwargs=True))
    else:
        handles.append(module.register_forward_hook(capture))
    if kind != "later":
        handles.append(module.register_forward_hook(plan))
    try:
        with torch.no_grad():
            actual = module(x)
        torch.testing.assert_close(actual, expected)
        assert captured
        for tensor, snapshot in captured:
            assert torch.equal(tensor, snapshot)
        assert not plan._bounded_reported
    finally:
        for handle in reversed(handles):
            handle.remove()


@pytest.mark.parametrize("kind", ["instance", "class", "cast"])
def test_caching_forward_wrapper_preserves_its_output(monkeypatch, kind):
    saved = []

    class CachingLinear(nn.Linear):
        def forward(self, x):
            output = super().forward(x)
            saved.append((output, output.clone()))
            return output

    module = CachingLinear(4, 6) if kind == "class" else nn.Linear(4, 6)
    if kind == "cast":
        module = comfy.ops.manual_cast.Linear(4, 6)
        with torch.no_grad():
            module.weight.normal_()
            module.bias.zero_()
    module, plan, x = setup(monkeypatch, module)
    handle = module.register_forward_hook(plan)
    # Install after VDN so a qualification cached at injection would be stale.
    if kind in ("instance", "cast"):
        name = "forward" if kind == "instance" else "forward_comfy_cast_weights"
        original = getattr(module, name)

        def wrapped(x):
            output = original(x)
            saved.append((output, output.clone()))
            return output

        setattr(module, name, wrapped)
    try:
        with torch.no_grad():
            actual = module(x)
        assert saved
        for tensor, snapshot in saved:
            assert torch.equal(tensor, snapshot)
            assert actual.data_ptr() != tensor.data_ptr()
        assert not plan._bounded_reported
    finally:
        handle.remove()


@pytest.mark.parametrize("provider", ["torch", "manual", "mixed"])
def test_unwrapped_linear_contract_enables_bounded_path(monkeypatch, provider):
    cls = {"torch": nn.Linear, "manual": comfy.ops.manual_cast.Linear,
           "mixed": comfy.ops.mixed_precision_ops(compute_dtype=torch.float32).Linear}[provider]
    module = cls(4, 6)
    module.weight = nn.Parameter(torch.randn(6, 4), requires_grad=False)
    module.bias = nn.Parameter(torch.randn(6), requires_grad=False)
    module, plan, x = setup(monkeypatch, module)
    with torch.no_grad():
        base = module(x)
        down, up, _ = plan._weights_for(x)
        expected = base + F.linear(F.linear(x, down), up)
    handle = module.register_forward_hook(plan)
    try:
        with torch.no_grad():
            actual = module(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        assert plan._bounded_reported
        # A subsequently added observer must disable mutation on the next call.
        captured = []
        observer = module.register_forward_hook(
            lambda m, args, out: captured.append((out, out.clone())), prepend=True)
        try:
            with torch.no_grad():
                module(x)
            assert torch.equal(*captured[0])
            assert plan._bounded_fallback_reported
        finally:
            observer.remove()
    finally:
        handle.remove()


def test_fatal_oom_diagnostics_preserve_original_error(monkeypatch):
    module, plan, x = setup(monkeypatch)
    output = module(x)
    error = torch.OutOfMemoryError("original allocation failure")
    calls = []

    def fail(*args):
        raise error

    monkeypatch.setattr(plan, "_apply", fail)
    monkeypatch.setattr(apply, "_log_post_forward_oom", lambda *args: calls.append(args))
    for _ in range(2):
        with pytest.raises(torch.OutOfMemoryError) as caught:
            plan(module, (x,), output)
        assert caught.value is error
    assert len(calls) == 1


def test_allocator_diagnostics_report_backend_and_stats(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "cudaMallocAsync")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (123, 456))
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda device: {"allocated_bytes.all.current": 42})
    monkeypatch.setattr(torch.cuda, "memory_summary", lambda device: "allocator summary")
    with caplog.at_level(logging.WARNING, logger="comfy.vdn"):
        apply._log_post_forward_oom(SimpleNamespace(device=torch.device("cuda:0")), torch.empty(2, 6))
    assert "cudaMallocAsync" in caplog.text
    assert "allocated_bytes.all.current" in caplog.text
    assert "allocator summary" in caplog.text


def test_unavailable_summary_keeps_other_allocator_evidence(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "cudaMallocAsync")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (123, 456))
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda device: {"active_bytes.all.current": 42})

    def unavailable(device):
        raise KeyError("unsupported counter")

    monkeypatch.setattr(torch.cuda, "memory_summary", unavailable)
    with caplog.at_level(logging.WARNING, logger="comfy.vdn"):
        apply._log_post_forward_oom(SimpleNamespace(device=torch.device("cuda:0")), torch.empty(2, 6))
    assert "cudaMallocAsync" in caplog.text
    assert "active_bytes.all.current" in caplog.text
    assert "unavailable" in caplog.text


def test_up_projection_oom_after_committed_tile_does_not_double_apply(monkeypatch):
    module, plan, x = setup(monkeypatch)
    with torch.no_grad():
        expected = module(x)
        down, up, _ = plan._weights_for(x)
        expected = expected + F.linear(F.linear(x, down), up)
    linear = F.linear
    up_calls = []

    def pressured(inputs, weight, bias=None):
        if weight is up:
            up_calls.append(inputs.shape[-2])
            if len(up_calls) == 2:
                raise torch.OutOfMemoryError("up projection allocation failed")
        return linear(inputs, weight, bias)

    # Two rows per initial tile; fail on the second tile's up projection.
    monkeypatch.setattr(apply, "_POST_FORWARD_DELTA_TILE_BYTES", 96)
    monkeypatch.setattr(apply.F, "linear", pressured)
    handle = module.register_forward_hook(plan)
    try:
        with torch.no_grad():
            actual = module(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        assert up_calls[:3] == [2, 2, 1]
        assert not plan._oom_reported
    finally:
        handle.remove()


def test_dispatch_observer_preserves_base_projection(monkeypatch):
    module, plan, x = setup(monkeypatch)
    captured = []

    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
            result = operation(*args, **(kwargs or {}))
            # Retain the base GEMM result, before VDN computes its owned delta.
            if not captured and isinstance(result, torch.Tensor) and result.numel() == 2 * 19 * 6:
                captured.append((result, result.clone()))
            return result

    handle = module.register_forward_hook(plan)
    try:
        with torch.no_grad(), Capture():
            module(x)
        assert captured
        for tensor, snapshot in captured:
            assert torch.equal(tensor, snapshot)
        assert not plan._bounded_reported
    finally:
        handle.remove()


def test_custom_module_dispatch_does_not_qualify(monkeypatch):
    class CustomCall(nn.Linear):
        def __call__(self, *args, **kwargs):
            return super().__call__(*args, **kwargs)

    module, plan, x = setup(monkeypatch, CustomCall(4, 6))
    handle = module.register_forward_hook(plan)
    try:
        with torch.no_grad():
            module(x)
        assert not plan._bounded_reported
    finally:
        handle.remove()
