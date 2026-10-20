"""Reuse small rank activations while preserving output tiling and OOM recovery."""
import pytest
import torch
import weakref
from torch import nn
from torch.nn import functional as F

from vdn_h3 import apply


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias_enabled", [False, True])
def test_direct_accumulation_reuses_private_output_without_up_tiles(
    monkeypatch, dtype, bias_enabled
):
    generator = torch.Generator().manual_seed(9152)
    x = torch.randn((25, 8), generator=generator).to(dtype)
    down = torch.randn((3, 8), generator=generator).to(dtype)
    up = torch.randn((12, 3), generator=generator).to(dtype)
    bias = torch.randn((12,), generator=generator).to(dtype) if bias_enabled else None
    original = torch.randn((25, 12), generator=generator).to(dtype)
    expected = original + F.linear(F.linear(x, down), up)
    if bias_enabled:
        expected += bias
    plan = apply._PostForwardLoRA(((down, up, 1.0),),
                                  ((bias, 1.0),) if bias_enabled else ())
    module = nn.Linear(8, 12, bias=False, dtype=dtype)
    handle = module.register_forward_hook(plan)
    down, up, _ = plan._weights_for(x)
    monkeypatch.setattr(apply, "_POST_FORWARD_DELTA_TILE_BYTES", 512)
    monkeypatch.setattr(apply, "_POST_FORWARD_DIRECT_GEMM", True)
    linear = F.linear
    addmm = torch.addmm
    calls = {"up": 0, "addmm": 0}

    def watched_linear(inputs, weight, bias=None):
        if weight is up:
            calls["up"] += 1
        return linear(inputs, weight, bias)

    def watched_addmm(input, mat1, mat2, **kwargs):
        assert input is original
        assert kwargs["out"] is original
        assert kwargs["beta"] == 1.0
        calls["addmm"] += 1
        return addmm(input, mat1, mat2, **kwargs)

    monkeypatch.setattr(apply.F, "linear", watched_linear)
    monkeypatch.setattr(apply.torch, "addmm", watched_addmm)
    try:
        with torch.no_grad():
            actual = plan(module, (x,), original)
        assert actual is original
        assert calls == {"up": 0, "addmm": 1}
        torch.testing.assert_close(
            actual, expected,
            rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
            atol=0.05 if dtype == torch.bfloat16 else 1e-5,
        )
    finally:
        handle.remove()


def test_direct_gemm_failure_never_retries_uncommitted_aliased_output(monkeypatch):
    generator = torch.Generator().manual_seed(66)
    x = torch.randn((13, 8), generator=generator)
    down = torch.randn((3, 8), generator=generator)
    up = torch.randn((9, 3), generator=generator)
    output = torch.randn((13, 9), generator=generator)
    before = output.clone()
    plan = apply._PostForwardLoRA(((down, up, 1.0),))
    module = nn.Linear(8, 9, bias=False)
    handle = module.register_forward_hook(plan)
    monkeypatch.setattr(apply, "_POST_FORWARD_DELTA_TILE_BYTES", 256)
    monkeypatch.setattr(apply, "_POST_FORWARD_DIRECT_GEMM", True)
    called = []

    def failed_addmm(input, mat1, mat2, **kwargs):
        called.append("direct")
        raise torch.OutOfMemoryError("direct GEMM unavailable")

    monkeypatch.setattr(apply.torch, "addmm", failed_addmm)
    try:
        with torch.no_grad(), pytest.raises(torch.OutOfMemoryError, match="direct GEMM unavailable"):
            plan(module, (x,), output)
        assert called == ["direct"]
        assert torch.equal(before, output)
    finally:
        handle.remove()

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("failure", ["none", "down", "up_after_commit", "fatal"])
@pytest.mark.parametrize("batched", [False, True])
def test_rank_workspace_reuse_and_memory_pressure(monkeypatch, dtype, failure, batched):
    generator = torch.Generator().manual_seed(2031)
    shape = (1, 13, 8) if batched else (13, 8)
    x = torch.randn(shape, generator=generator).to(dtype)
    down = torch.randn(2, 8, generator=generator).to(dtype)
    up = torch.randn(8, 2, generator=generator).to(dtype)
    bias = torch.randn(8, generator=generator).to(dtype)
    output = torch.randn(shape, generator=generator).to(dtype)
    expected = output + (F.linear(F.linear(x, down), up) + bias)
    plan = apply._PostForwardLoRA(((down, up, 1.0),), ((bias, 1.0),))
    module = nn.Linear(8, 8, dtype=dtype)
    handle = module.register_forward_hook(plan)
    down, up, _ = plan._weights_for(x)
    monkeypatch.setattr(apply, "_POST_FORWARD_DELTA_TILE_BYTES", 128)
    # Keep exercising OOM retry and bounded tiles separately from direct GEMM.
    monkeypatch.setattr(apply, "_POST_FORWARD_DIRECT_GEMM", False)
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
