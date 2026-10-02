"""Numerical contracts for the optional compile-fused branch fast paths.

Inductor can fuse several BF16 pointwise/reduction operations and round once at the
store where eager PyTorch rounds between operations. The fast path must preserve the
same algorithm and stay within a small BF16 error budget; bitwise identity is neither
expected nor claimed.
"""
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vdn_h3.branch import linear_epilogue
from vdn_h3 import branch as B


BF16_ATOL = 5e-3
BF16_RTOL = 1e-2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_linear_norm_epsilon_keeps_exact_dtype_rounding_on_cpu(monkeypatch, dtype):
    expected = torch.empty((), dtype=dtype).fill_(1e-6).item()
    B.linear_norm_epsilon.cache_clear()
    make_tensor = torch.tensor
    calls = []

    def cpu_constant(*args, **kwargs):
        assert kwargs["device"] == "cpu"
        calls.append(kwargs["dtype"])
        return make_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_constant)
    assert B.linear_norm_epsilon(dtype) == expected
    assert B.linear_norm_epsilon(dtype) == expected
    assert calls == [dtype]


def test_epilogue_parity():
    torch.manual_seed(0)
    frames, heads, per_frame, dim = 4, 3, 5, 8
    readout = torch.randn(frames, heads, per_frame, dim).to(torch.bfloat16)
    weight = torch.randn(dim).to(torch.bfloat16)
    gate = torch.rand(frames * per_frame, heads * dim).to(torch.bfloat16)

    eager = linear_epilogue(readout, weight, gate, 1e-6, fuse=False)
    fused = linear_epilogue(readout, weight, gate, 1e-6, fuse=True)
    assert torch.allclose(
        eager.float(), fused.float(), atol=BF16_ATOL, rtol=BF16_RTOL), (
        "compiled epilogue exceeded BF16 rounding budget")
    assert eager.shape == (frames * per_frame, heads * dim)


def test_q_fhsd_store():
    """The frame-major compiled q store replaces eager activation+view+permute.

    Values may differ by roughly one BF16 ulp because Inductor can fuse the
    normalization/reordering and change where rounding occurs.
    """
    torch.manual_seed(1)
    frames, per_frame, heads, dim = 4, 5, 3, 8
    x = torch.randn(frames * per_frame, heads, dim).to(torch.bfloat16)
    want = B._activate(x, True).view(frames, per_frame, heads, dim) \
        .permute(0, 2, 1, 3).contiguous()
    got = B._run_compiled(("test_act_fhsd", True), B._activate_fhsd_body,
                          x, True, frames, per_frame)
    assert got.shape == (frames, heads, per_frame, dim) and got.is_contiguous()
    assert torch.allclose(
        got.float(), want.float(), atol=BF16_ATOL, rtol=BF16_RTOL), (
        "compiled frame-major q store exceeded BF16 rounding budget")


def test_readout_parity():
    """End to end: LinearBranch._readout under fast_kernels (fused epilogue + fused
    gather + frame-major q store) must match the default eager path at FP32 test
    precision. This fixture uses FP32 activations, so the tighter budget is expected.
    """
    torch.manual_seed(2)
    frames, per_frame, heads, dim, hidden = 6, 4, 2, 4, 16
    channels = heads * dim
    w = {
        "beta_proj.weight": torch.randn(heads, hidden),
        "alpha.down.weight": torch.randn(dim, hidden),
        "alpha.up.weight": torch.randn(heads * dim, dim),
        "alpha.dt_bias": torch.randn(heads * dim),
        "alpha.A_log": torch.randn(heads),
        "output_gate.down.weight": torch.randn(dim, hidden),
        "output_gate.up.weight": torch.randn(heads * dim, dim),
        "output_gate.up.bias": torch.randn(heads * dim),
        "norm.weight": torch.randn(dim),
        "short_conv.k_sp.weight": torch.randn(channels, 1, 5, 5) * 0.2,
        "short_conv.k_tm.weight": torch.randn(channels, 1, 5) * 0.2,
        "short_conv.v_sp.weight": torch.randn(channels, 1, 5, 5) * 0.2,
        "short_conv.v_tm.weight": torch.randn(channels, 1, 5) * 0.2,
    }
    rows = frames * per_frame
    xv = torch.randn(rows, hidden)
    q_raw = torch.randn(rows, heads, dim)
    k_raw = torch.randn(rows, heads, dim)
    v_raw = torch.randn(rows, heads, dim)
    from vdn_h3.window import window_bounds
    bounds = window_bounds(frames, 1, 2)

    branch = B.LinearBranch(w, heads, dim, delta_rule="vdn_solve", bridge="alpha",
                            a_fp32=True, short_conv=("k", "v"),
                            enable_text_state=False)
    eager = branch.readout(w, xv, q_raw, k_raw, v_raw, frames, per_frame, bounds,
                           frame_size=(2, 2))
    branch.fuse_epilogue = True
    fused = branch.readout(w, xv, q_raw, k_raw, v_raw, frames, per_frame, bounds,
                           frame_size=(2, 2))
    err = (fused - eager).abs().max().item()
    assert err < 1e-4, f"fast_kernels readout differs: {err}"


if __name__ == "__main__":
    test_epilogue_parity()
    test_q_fhsd_store()
    test_readout_parity()
    print("ALL PASS")
