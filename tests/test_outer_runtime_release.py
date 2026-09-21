from __future__ import annotations

from types import SimpleNamespace

import pytest

from vdn_h3.hybrid import (
    _allocator_pressure_requires_trim,
    _high_stage_block_trim_due,
    _maybe_trim_high_allocator,
    make_outer_release_wrapper,
)


class _Runtime:
    def __init__(self):
        self.releases = 0

    def release_retained(self):
        self.releases += 1
        return {"scan": 1}


class _State:
    retain_buffers = True

    def __init__(self):
        self.runtime = _Runtime()


class _Executor:
    def __init__(self, model_options, *, error=None):
        self.class_obj = SimpleNamespace(model_options=model_options)
        self.error = error

    def __call__(self, *args, **kwargs):
        del args, kwargs
        if self.error is not None:
            raise self.error
        return "ok"


def _options(stage_marker, stage=None):
    transformer = {}
    if stage_marker:
        transformer["h3_flow_stage"] = stage
    return {
        "h3_flow_partitioned_progressive_v1": object(),
        "transformer_options": transformer,
    }


@pytest.mark.parametrize(
    ("stage_marker", "stage", "expected"),
    [
        (True, "low", 0),
        (True, "probe", 1),
        (True, "high", 1),
        (False, None, 1),
    ],
)
def test_progressive_release_is_order_independent(monkeypatch, stage_marker, stage, expected):
    state = _State()
    trims = []
    monkeypatch.setattr(
        "vdn_h3.hybrid.comfy.model_management.soft_empty_cache",
        lambda: trims.append(True),
    )
    wrapped = make_outer_release_wrapper(state)
    assert wrapped(_Executor(_options(stage_marker, stage))) == "ok"
    assert state.runtime.releases == expected
    assert len(trims) == expected


def test_probe_release_allows_fresh_high_stage_pool(monkeypatch):
    state = _State()
    trims = []
    monkeypatch.setattr(
        "vdn_h3.hybrid.comfy.model_management.soft_empty_cache",
        lambda: trims.append(True),
    )
    wrapped = make_outer_release_wrapper(state)
    assert wrapped(_Executor(_options(True, "probe"))) == "ok"
    assert state.runtime.releases == 1
    assert trims == [True]
    assert wrapped(_Executor(_options(True, "high"))) == "ok"
    assert state.runtime.releases == 2
    assert trims == [True, True]


def test_failed_progressive_substage_releases_immediately(monkeypatch):
    state = _State()
    trims = []
    monkeypatch.setattr(
        "vdn_h3.hybrid.comfy.model_management.soft_empty_cache",
        lambda: trims.append(True),
    )
    wrapped = make_outer_release_wrapper(state)
    with pytest.raises(RuntimeError, match="boom"):
        wrapped(_Executor(_options(True, "low"), error=RuntimeError("boom")))
    assert state.runtime.releases == 1
    assert trims == [True]


def test_non_progressive_outer_sample_preserves_existing_retention_policy(monkeypatch):
    state = _State()
    trims = []
    monkeypatch.setattr(
        "vdn_h3.hybrid.comfy.model_management.soft_empty_cache",
        lambda: trims.append(True),
    )
    wrapped = make_outer_release_wrapper(state)
    options = {"transformer_options": {"h3_flow_stage": "high"}}
    assert wrapped(_Executor(options)) == "ok"
    assert state.runtime.releases == 0
    assert trims == []



def test_high_allocator_pressure_gate_matches_capacity_and_reclaimable_pool():
    assert _allocator_pressure_requires_trim(
        {
            "allocated_mib": 58483.0,
            "reserved_mib": 87168.0,
            "free_mib": 9000.0,
            "total_mib": 97886.0,
        }
    )
    assert not _allocator_pressure_requires_trim(
        {
            "allocated_mib": 53788.0,
            "reserved_mib": 56288.0,
            "free_mib": 41000.0,
            "total_mib": 97886.0,
        }
    )
    assert not _allocator_pressure_requires_trim(
        {
            "allocated_mib": 84000.0,
            "reserved_mib": 88000.0,
            "free_mib": 9000.0,
            "total_mib": 97886.0,
        }
    )


def test_high_stage_block_trim_sampling_is_sparse_and_stage_bounded():
    assert not _high_stage_block_trim_due({"h3_flow_stage": "low"}, 3)
    assert not _high_stage_block_trim_due({"h3_flow_stage": "high"}, 0)
    assert not _high_stage_block_trim_due({"h3_flow_stage": "high"}, 2)
    assert _high_stage_block_trim_due({"h3_flow_stage": "high"}, 3)
    assert _high_stage_block_trim_due({"h3_flow_stage": "high"}, 7)


def test_high_allocator_pressure_trim_is_stage_and_pressure_bounded(monkeypatch):
    state = _State()
    snapshots = iter(
        [
            {
                "allocated_mib": 58483.0,
                "reserved_mib": 87168.0,
                "free_mib": 9000.0,
                "total_mib": 97886.0,
            },
            {
                "allocated_mib": 58483.0,
                "reserved_mib": 60224.0,
                "free_mib": 37000.0,
                "total_mib": 97886.0,
            },
        ]
    )
    trims = []
    monkeypatch.setattr("vdn_h3.hybrid._cuda_allocator_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(
        "vdn_h3.hybrid.comfy.model_management.soft_empty_cache",
        lambda: trims.append(True),
    )
    assert _maybe_trim_high_allocator(state, {"h3_flow_stage": "high"}) is True
    assert trims == [True]


def test_high_allocator_pressure_trim_does_not_touch_low_or_healthy_high(monkeypatch):
    state = _State()
    trims = []
    healthy = {
        "allocated_mib": 54000.0,
        "reserved_mib": 65000.0,
        "free_mib": 32000.0,
        "total_mib": 97886.0,
    }
    monkeypatch.setattr("vdn_h3.hybrid._cuda_allocator_snapshot", lambda: healthy)
    monkeypatch.setattr(
        "vdn_h3.hybrid.comfy.model_management.soft_empty_cache",
        lambda: trims.append(True),
    )
    assert _maybe_trim_high_allocator(state, {"h3_flow_stage": "low"}) is False
    assert _maybe_trim_high_allocator(state, {"h3_flow_stage": "high"}) is False
    assert trims == []
