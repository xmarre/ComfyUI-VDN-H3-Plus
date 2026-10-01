"""Owned adapter output storage must not overwrite upstream or cached tensors."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from comfy.weight_adapter.bypass import BypassForwardHook
from comfy.weight_adapter.lora import LoRAAdapter

from vdn_h3.apply import _PostForwardLoRA


def make_hook(dtype=torch.float32, bias=False):
    generator = torch.Generator().manual_seed(137)
    terms = [
        (torch.randn(2, 4, generator=generator, dtype=dtype),
         torch.randn(6, 2, generator=generator, dtype=dtype), -0.75),
        (torch.randn(3, 4, generator=generator, dtype=dtype),
         torch.randn(6, 3, generator=generator, dtype=dtype), 0.5),
    ]
    offsets = [(torch.randn(6, generator=generator, dtype=dtype), 0.25)] if bias else []
    return _PostForwardLoRA(terms, offsets)


def reference(hook, x, output):
    down, up, bias = hook._weights_for(x)
    delta = F.linear(F.linear(x, down), up) if down is not None else None
    if bias is not None:
        delta = bias if delta is None else delta + bias
    return output if delta is None else output + delta


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_inference_preserves_exact_arithmetic_and_cached_factors(dtype, bias):
    hook = make_hook(dtype, bias)
    for shape in ((19, 4), (2, 19, 4)):
        x = torch.randn(shape, dtype=dtype)
        output = torch.randn((*shape[:-1], 6), dtype=dtype)
        before = output.clone()
        weights = hook._weights_for(x)
        saved = [tensor.clone() if tensor is not None else None for tensor in weights]
        with torch.no_grad():
            expected = reference(hook, x, output)
            actual = hook(nn.Identity(), (x,), output)
        assert torch.equal(actual, expected)
        assert actual.dtype == expected.dtype and actual.stride() == expected.stride()
        assert actual.data_ptr() != output.data_ptr()
        assert torch.equal(output, before)
        for tensor, previous in zip(weights, saved):
            if tensor is not None:
                assert torch.equal(tensor, previous)
    assert len(hook._cache) == 1


class OutputStorage(TorchDispatchMode):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.pointers = set()

    def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
        result = operation(*args, **(kwargs or {}))
        if isinstance(result, torch.Tensor) and result.shape == self.shape:
            self.pointers.add(result.data_ptr())
        return result


def test_post_hook_uses_only_base_and_projection_output_storage():
    module = nn.Linear(4, 6, bias=False)
    hook = make_hook()
    hook.prepare(module)
    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad(), OutputStorage((19, 6)) as storage:
            actual = module(torch.ones(19, 4))
        assert len(storage.pointers) == 2
        assert actual.data_ptr() in storage.pointers
    finally:
        handle.remove()
        hook.clear()


def test_input_alias_and_earlier_post_hook_output_remain_unchanged():
    module = nn.Identity()
    hook = _PostForwardLoRA([(torch.randn(2, 6), torch.randn(6, 2), 0.5)])
    observed = []
    first = module.register_forward_hook(lambda _module, _inputs, output: observed.append(output))
    second = module.register_forward_hook(hook)
    x = torch.randn(19, 6)
    before = x.clone()
    try:
        with torch.no_grad():
            expected = reference(hook, x, x)
            actual = module(x)
        assert torch.equal(actual, expected)
        assert len(observed) == 1 and observed[0] is x
        assert torch.equal(x, before)
        assert actual.data_ptr() != x.data_ptr()
    finally:
        second.remove()
        first.remove()


def test_gradients_preserve_out_of_place_arithmetic():
    module = nn.Linear(4, 6)
    hook = make_hook(bias=True)
    x = torch.randn(19, 4, requires_grad=True)
    output = module(x)
    expected = reference(hook, x, output)
    expected_gradients = torch.autograd.grad(expected.square().sum(), (x, module.weight, module.bias), retain_graph=True)
    actual = hook(module, (x,), output)
    actual_gradients = torch.autograd.grad(actual.square().sum(), (x, module.weight, module.bias))
    assert torch.equal(actual, expected)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.equal(actual_gradient, expected_gradient)


@pytest.mark.parametrize("case", ["promotion", "layout", "broadcast"])
def test_fallback_preserves_dtype_broadcast_and_layout(case):
    hook = make_hook()
    x = torch.randn(19, 4)
    output = {
        "promotion": lambda: torch.randn(19, 6, dtype=torch.float64),
        "layout": lambda: torch.randn(6, 19).t(),
        "broadcast": lambda: torch.randn(1, 6),
    }[case]()
    before = output.clone()
    with torch.no_grad():
        expected = reference(hook, x, output)
        actual = hook(nn.Identity(), (x,), output)
    assert torch.equal(actual, expected)
    assert actual.dtype == expected.dtype and actual.stride() == expected.stride()
    assert torch.equal(output, before)


def test_bias_only_hook_does_not_overwrite_shared_bias():
    bias = torch.randn(6)
    hook = _PostForwardLoRA([], [(bias, 1.0)])
    before = bias.clone()
    for rows in (19, 23, 19):
        x = torch.randn(rows, 6)
        with torch.no_grad():
            expected = x + bias
            actual = hook(nn.Identity(), (x,), x)
        assert torch.equal(actual, expected)
        assert torch.equal(bias, before)
        assert hook._weights_for(x)[2].data_ptr() == bias.data_ptr()


def test_tensor_subclass_preserves_dispatch():
    class Activation(torch.Tensor):
        pass

    hook = make_hook()
    x = torch.randn(19, 4).as_subclass(Activation)
    output = torch.randn(19, 6).as_subclass(Activation)
    before = output.clone()
    with torch.no_grad():
        expected = reference(hook, x, output)
        actual = hook(nn.Identity(), (x,), output)
    assert type(actual) is type(expected) is Activation
    assert torch.equal(actual, expected) and torch.equal(output, before)


@pytest.mark.parametrize("external_first", [False, True])
def test_post_hook_preserves_external_core_bypass_order(external_first):
    module = nn.Linear(4, 6, bias=False)
    original = module.forward
    down, up = torch.randn(2, 4), torch.randn(6, 2)
    adapter = LoRAAdapter(set(), (up, down, 2.0, None, None, None))
    external = BypassForwardHook(module, adapter, multiplier=0.4)
    hook = make_hook()
    if external_first:
        external.inject()
        handle = module.register_forward_hook(hook)
    else:
        handle = module.register_forward_hook(hook)
        external.inject()
    x = torch.randn(19, 4)
    try:
        with torch.no_grad():
            core_output = original(x) + F.linear(F.linear(x, down), up) * 0.4
            expected = reference(hook, x, core_output)
            assert torch.equal(module(x), expected)
        handle.remove()
        with torch.no_grad():
            assert torch.equal(module(x), core_output)
    finally:
        handle.remove()
        external.eject()
    assert module.forward == original
