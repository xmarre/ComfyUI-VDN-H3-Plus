"""Regressions adopted from upstream v1.5.0/v1.5.1 after Plus-specific review."""
from __future__ import annotations

import weakref

import pytest
import torch
import torch.nn.functional as F

from vdn_h3 import branch, hybrid, retained


@pytest.mark.parametrize("frame_major", [False, True])
def test_query_short_conv_is_applied(frame_major):
    torch.manual_seed(13)
    q = torch.randn(12, 2, 4)
    # A zero q convolution must produce zero Q features, not raw SiLU(Q).
    weights = {
        "short_conv.q_sp.weight": torch.zeros(8, 1, 5, 5),
        "short_conv.q_tm.weight": torch.ones(8, 1, 5),
    }
    linear = branch.LinearBranch(weights, 2, 4, short_conv=("q",))
    got, _, _ = linear._features(
        weights, q, q, q, 3, (2, 2), q_fhsd=frame_major)
    assert torch.count_nonzero(got) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("a_fp32", [False, True])
def test_frame_statistics_batches_preserve_results(dtype, a_fp32, monkeypatch):
    torch.manual_seed(15)
    k = torch.randn(7, 23, 3, 16, dtype=dtype).permute(0, 2, 1, 3)
    v = torch.randn_like(k)
    beta = torch.rand(7, 3, 23, dtype=dtype)
    reference = branch._frame_statistics_chunk(k, v, beta, a_fp32)

    frames, heads, tokens, dim = k.shape
    per_frame = heads * tokens * (
        dim * (k.element_size() + (8 if a_fp32 else k.element_size()))
        + 2 * v.shape[-1] * v.element_size())
    monkeypatch.setattr(branch, "_STATISTICS_WORKSPACE_BYTES", 3 * per_frame)

    sizes = []
    original = branch._frame_statistics_chunk

    def chunk(k_chunk, v_chunk, beta_chunk, a_fp32_chunk):
        sizes.append(k_chunk.shape[0])
        return original(k_chunk, v_chunk, beta_chunk, a_fp32_chunk)

    monkeypatch.setattr(branch, "_frame_statistics_chunk", chunk)
    got = branch.frame_statistics(k, v, beta, a_fp32)

    assert sizes == [3, 3, 1]
    assert all(torch.equal(actual, expected)
               for actual, expected in zip(got, reference))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_readout_epsilon_preserves_checkpoint_dtype_rounding(dtype):
    expected = torch.tensor(1e-6, dtype=dtype, device="cpu").item()
    assert branch._readout_eps(dtype) == expected


def test_temporal_shift_preserves_rounding():
    torch.manual_seed(14)
    x = torch.randn(7, 15, 64)
    w = torch.randn(64, 5)
    padded = F.pad(x, (0, 0, 0, 0, 2, 2))
    expected = padded[:7] * w[:, 0]
    for tap in range(1, 5):
        expected = expected + padded[tap:tap + 7] * w[:, tap]
    got = branch._temporal_shift(x, w, 5)
    assert torch.equal(got, expected)


def _attention():
    return type("Attention", (), {
        "heads": 2,
        "head_dim": 4,
        "qkv_proj": torch.nn.Linear(8, 24, bias=False),
        "out_proj": torch.nn.Linear(8, 8, bias=False),
        "q_norm": torch.nn.RMSNorm(4, eps=1e-6),
        "k_norm": torch.nn.RMSNorm(4, eps=1e-6),
    })()


def test_hybrid_releases_projection_and_raw_branch_buffers(monkeypatch):
    attn = _attention()
    refs = {}
    attn.qkv_proj.register_forward_hook(
        lambda module, args, out: refs.update(qkv=weakref.ref(out)))
    attn.out_proj.register_forward_pre_hook(
        lambda module, args: refs.update(flat=weakref.ref(args[0])))

    def readout(weights, xv, *args, **kwargs):
        assert refs["qkv"]() is None, "RoPE views retained the fused QKV allocation"
        assert refs["flat"]() is None, "softmax projection input survived into branch"
        refs["raw"] = [weakref.ref(t) for t in args[:3]]
        return torch.zeros(xv.shape[0], 8)

    linear = type("Linear", (), {
        "enable_text_state": False,
        "_backend": None,
        "_backend_key": None,
        "readout": staticmethod(readout),
    })()
    cfg = {
        "enable_softmax_gate": True,
        "anchor_frames": "none",
        "linear_enabled": True,
    }
    state = hybrid.VDNState("test", cfg, [linear], 2, 4, retain_buffers=False)
    # text [0,2), target audio [2,3), video [3,25), one trailing global row.
    layout = hybrid.VDNLayout(
        3, 25, 2, 3, 11, 2, (1, 2), 0, 2, 26, 1, 5, "none")
    token = state._layout.set(layout)
    weights = {
        "softmax_gate.up.weight": torch.randn(2, 8),
        "softmax_gate.up.bias": torch.randn(2),
        "to_out_linear.weight": torch.randn(8, 8),
    }
    state.weights_on = lambda *args: weights

    monkeypatch.setattr(
        retained,
        "window_softmax_grouped_runtime",
        lambda q, k, v, *args, **kwargs: torch.zeros_like(q),
    )
    monkeypatch.setattr(
        hybrid.comfy.quant_ops.ck,
        "rms_rope_split_half_",
        lambda *args, **kwargs: None,
    )
    original_linear = F.linear

    def checked_linear(x, weight, *args, **kwargs):
        if weight is weights["to_out_linear.weight"]:
            assert all(ref() is None for ref in refs["raw"]), \
                "raw branch Q/K/V survived into final projection"
        return original_linear(x, weight, *args, **kwargs)

    monkeypatch.setattr(F, "linear", checked_linear)
    try:
        with torch.inference_mode():
            got = hybrid.make_vdn_forward(attn, state, 0)(
                torch.randn(26, 8), rope_freqs=torch.zeros(1, 1, 2))
        assert got.shape == (26, 8)
    finally:
        state._layout.reset(token)
