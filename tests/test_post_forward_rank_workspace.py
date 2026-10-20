"""Reuse small rank activations while preserving output tiling and OOM recovery."""
import pytest
import torch
import weakref
from torch import nn
from torch.nn import functional as F

from vdn_h3 import apply


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("failure", ["none", "down", "up_after_commit", "fatal"])
def test_rank_workspace_reuse_and_memory_pressure(monkeypatch, dtype, failure):
    generator = torch.Generator().manual_seed(2031)
    x = torch.randn(1, 13, 8, generator=generator).to(dtype)
    down = torch.randn(2, 8, generator=generator).to(dtype)
    up = torch.randn(8, 2, generator=generator).to(dtype)
    bias = torch.randn(8, generator=generator).to(dtype)
    output = torch.randn(1, 13, 8, generator=generator).to(dtype)
    expected = output + (F.linear(F.linear(x, down), up) + bias)
    plan = apply._PostForwardLoRA(((down, up, 1.0),), ((bias, 1.0),))
    module = nn.Linear(8, 8, dtype=dtype)
    handle = module.register_forward_hook(plan)
    down, up, _ = plan._weights_for(x)
    monkeypatch.setattr(apply, "_POST_FORWARD_DELTA_TILE_BYTES", 128)
    linear = F.linear
    calls = []
    up_count = 0
    rank_bytes = x.shape[-2] * down.shape[0] * x.element_size()
    full_rank = []

    def pressured(inputs, weight, bias=None):
        nonlocal up_count
        kind = "down" if weight is down else "up"
        calls.append((kind, inputs.shape[-2]))
        if kind == "down" and inputs.shape[-2] == 1 and failure == "up_after_commit":
            assert full_rank[-1]() is None
        if kind == "down" and failure == "down" and inputs.shape[-2] == 13:
            raise torch.OutOfMemoryError("rank workspace unavailable")
        if kind == "up":
            up_count += 1
            if failure == "fatal":
                raise torch.OutOfMemoryError("one row unavailable")
            if (failure == "up_after_commit" and up_count > 1
                    and inputs.untyped_storage().nbytes() == rank_bytes):
                raise torch.OutOfMemoryError("retained workspace pressure")
            assert inputs.shape[-2] * up.shape[0] * inputs.element_size() <= 128
        result = linear(inputs, weight, bias)
        if kind == "down" and inputs.shape[-2] == 13:
            full_rank.append(weakref.ref(result))
        return result

    monkeypatch.setattr(apply.F, "linear", pressured)
    try:
        with torch.no_grad():
            if failure == "fatal":
                with pytest.raises(torch.OutOfMemoryError, match="one row unavailable"):
                    plan(module, (x,), output)
                # The final attempt reprojects an uncommitted single row after
                # releasing the full rank workspace.
                assert calls[-2:] == [("down", 1), ("up", 1)]
                return
            actual = plan(module, (x,), output)
        torch.testing.assert_close(actual, expected,
                                   rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
                                   atol=0.05 if dtype == torch.bfloat16 else 1e-5)
        assert actual is output
        down_calls = [rows for kind, rows in calls if kind == "down"]
        assert down_calls[0] == 13
        if failure == "none":
            assert down_calls == [13]
            assert up_count > 1
        else:
            assert len(down_calls) > 1
            assert max(down_calls[1:]) <= (8 if dtype == torch.bfloat16 else 4)
        if failure == "up_after_commit":
            assert ("down", 1) in calls
    finally:
        handle.remove()
