from __future__ import annotations

from types import SimpleNamespace

import pytest

from vdn_h3.hybrid import make_outer_release_wrapper


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
        (True, "probe", 0),
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
