from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vdn_h3.drift_diagnostics import (
    cpu_snapshot,
    packed_target_ranges,
    pair_metrics,
    split_qkv_rows,
)


def test_packed_target_ranges_derive_prefix_audio_video_from_segments():
    layout = SimpleNamespace(
        seq_len=17,
        segments=[
            (0, 3, "text"),
            (3, 5, "ref_audio"),
            (5, 7, "ref_img"),
            (7, 11, "audio"),
            (11, 17, "video"),
        ],
    )
    assert packed_target_ranges(layout) == {
        "prefix": (0, 7),
        "audio": (7, 11),
        "video": (11, 17),
    }


def test_packed_target_ranges_fail_closed_when_target_order_is_not_h3_contract():
    layout = SimpleNamespace(
        seq_len=9,
        segments=[(0, 2, "text"), (2, 6, "video"), (6, 9, "audio")],
    )
    with pytest.raises(RuntimeError, match="target layout"):
        packed_target_ranges(layout)


def test_split_qkv_rows_preserves_projection_order():
    x = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    q, k, v = split_qkv_rows(x)
    assert torch.equal(q, x[:, :4])
    assert torch.equal(k, x[:, 4:8])
    assert torch.equal(v, x[:, 8:])
    with pytest.raises(ValueError, match="divisible by three"):
        split_qkv_rows(torch.zeros(2, 10))


def test_pair_metrics_identity_and_known_delta():
    ref = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    identity = pair_metrics(ref, ref.clone())
    assert identity.rel_rms == 0.0
    assert identity.cosine == pytest.approx(1.0)
    assert identity.max_abs == 0.0

    cand = ref + 1.0
    metrics = pair_metrics(ref, cand)
    expected_rel = torch.ones_like(ref).square().mean().sqrt() / ref.square().mean().sqrt()
    expected_cos = torch.nn.functional.cosine_similarity(ref.flatten(), cand.flatten(), dim=0)
    assert metrics.rel_rms == pytest.approx(float(expected_rel))
    assert metrics.cosine == pytest.approx(float(expected_cos))
    assert metrics.max_abs == 1.0


def test_cpu_snapshot_is_detached_compact_bfloat16():
    x = torch.randn(3, 4, requires_grad=True)
    snap = cpu_snapshot(x)
    assert snap.device.type == "cpu"
    assert snap.dtype == torch.bfloat16
    assert not snap.requires_grad
    assert snap.is_contiguous()
