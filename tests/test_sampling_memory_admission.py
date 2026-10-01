from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import comfy.model_management as management
import comfy.sampler_helpers
from comfy.model_patcher import ModelPatcher

from vdn_h3 import hybrid, policy

GIB = 1 << 30
RETAINED_SAMPLING_HEADROOM_BYTES = 10 * GIB


def _model(device="cuda"):
    model = ModelPatcher(nn.Linear(2, 2), torch.device(device), torch.device("cpu"))
    model.model_dtype = lambda: torch.bfloat16
    return model


@pytest.fixture
def budget(monkeypatch):
    monkeypatch.setattr(hybrid, "_cuda_allocator_snapshot", lambda: None)
    monkeypatch.setattr(management, "current_loaded_models", [])
    monkeypatch.setattr(management, "minimum_inference_memory", lambda: GIB)
    monkeypatch.setattr(management, "extra_reserved_memory", lambda: GIB // 4)
    monkeypatch.setattr(hybrid.comfy.sampler_helpers, "estimate_memory",
                        lambda *_args: (2 * GIB, GIB))
    monkeypatch.setattr(hybrid.comfy.sampler_helpers, "get_additional_models",
                        lambda *_args: ([], GIB // 2))
    return 2 * GIB + GIB // 2 + GIB // 4 + RETAINED_SAMPLING_HEADROOM_BYTES


@pytest.mark.parametrize("disable_smart_memory", [False, True])
def test_sufficient_headroom_preserves_unrelated_models(monkeypatch, caplog, budget,
                                                       disable_smart_memory):
    model, text, vae = _model(), _model(), _model()
    loaded = [management.LoadedModel(item) for item in (model, text, vae)]
    monkeypatch.setattr(management, "current_loaded_models", loaded)
    monkeypatch.setattr(management, "DISABLE_SMART_MEMORY", disable_smart_memory)
    monkeypatch.setattr(management, "get_free_memory", lambda _device: budget)
    monkeypatch.setattr(management, "free_memory", lambda *_args, **_kwargs: pytest.fail(
        "Admission must not request eviction when headroom already fits"))
    conds = {"positive": []}
    prepared = (model.model, conds, [])
    calls = []
    caplog.set_level("INFO", logger="comfy.vdn")

    def executor(*args, **kwargs):
        calls.append((args, kwargs))
        return prepared

    wrapped = hybrid.make_prepare_sampling_memory_wrapper(SimpleNamespace(retain_buffers=True))
    assert wrapped(executor, model, (1, 1, 1), conds) is prepared
    assert len(calls) == 1
    assert management.current_loaded_models is loaded
    assert "policy=bounded_headroom_v1" in caplog.text
    assert "eviction_requested=False" in caplog.text
    assert caplog.text.index("sampling preparation") < caplog.text.index("sampling admission")


@pytest.mark.parametrize("changed_weights", [False, True])
def test_pressure_preserves_prepared_models_clones_and_patch_backings(monkeypatch, caplog,
                                                                     budget, changed_weights):
    base = _model()
    resident, incoming = base.clone(), base.clone()
    if changed_weights:
        incoming.add_patches({"weight": (torch.ones_like(base.model.weight),)})
    branch, hook, backing, unrelated = (_model() for _ in range(4))
    hook.model_patches_models = lambda: [backing]
    incoming.model_dtype = lambda: torch.bfloat16
    cpu_clone = incoming.clone()
    cpu_clone.load_device = torch.device("cpu")
    loaded = [management.LoadedModel(item) for item in
              (resident, branch, hook, backing, unrelated, cpu_clone)]
    dead = SimpleNamespace(model=None)
    conds = {"positive": []}
    prepared_conds = {"positive": [{"prepared": True}]}
    prepared = (incoming.model, prepared_conds, [branch, hook])
    options = {"transformer_options": {"h3_flow_stage": "low"}}
    events = []
    monkeypatch.setattr(management, "get_free_memory", lambda _device: budget - 1)
    clock = iter([1.0, 1.5, 2.0, 2.2])
    monkeypatch.setattr(hybrid, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    caplog.set_level("INFO", logger="comfy.vdn")

    def estimate(model, shape, actual_conds):
        assert model is incoming and shape == (1, 2, 3)
        assert actual_conds is prepared_conds
        events.append("estimate")
        return 2 * GIB, GIB

    monkeypatch.setattr(hybrid.comfy.sampler_helpers, "estimate_memory", estimate)

    def executor(model, shape, actual_conds, **kwargs):
        assert model is incoming and shape == (1, 2, 3) and actual_conds is conds
        assert kwargs == {"model_options": options, "force_full_load": True,
                          "force_offload": True}
        # Core controls clone switching and weight-patch reconciliation.
        assert (incoming.patches_uuid != resident.patches_uuid) is changed_weights
        assert management.current_loaded_models == []
        monkeypatch.setattr(management, "current_loaded_models", [*loaded, dead])
        events.append("prepare")
        return prepared

    def free_memory(required, device, keep_loaded):
        assert required == budget and device == torch.device("cuda")
        assert keep_loaded == loaded[:4]
        assert loaded[0] != management.LoadedModel(incoming)
        assert incoming.is_clone(resident)
        events.append("evict")
        return [loaded[4]]

    monkeypatch.setattr(management, "free_memory", free_memory)
    wrapped = hybrid.make_prepare_sampling_memory_wrapper(SimpleNamespace(retain_buffers=True))
    assert wrapped(executor, incoming, (1, 2, 3), conds, model_options=options,
                   force_full_load=True, force_offload=True) is prepared
    assert events == ["prepare", "estimate", "evict"]
    assert "kept_resident_h3=1 kept_required=4 eviction_elapsed_ms=200.000" in caplog.text
    assert "stage=low success=True prepare_elapsed_ms=500.000" in caplog.text
    assert "eviction_requested=True unloaded_models=('Linear',)" in caplog.text


def test_core_partial_eviction_stops_at_finite_shortfall(monkeypatch, caplog, budget):
    model, branch, text = _model(), _model(), _model()
    loaded = [management.LoadedModel(item) for item in (model, branch, text)]
    memory = {"free": GIB}
    requests = []
    for item in loaded:
        item.is_dead = lambda: False
        item.model_offloaded_memory = lambda: 0
        item.model_memory = lambda: 32 * GIB
    loaded[0].model_unload = lambda *_args: pytest.fail("H3 was evicted")
    loaded[1].model_unload = lambda *_args: pytest.fail("Required branch was evicted")

    def unload(shortfall):
        requests.append(shortfall)
        memory["free"] += shortfall
        return False  # Core retains the registry entry after a partial unload.

    loaded[2].model_unload = unload
    monkeypatch.setattr(management, "current_loaded_models", loaded)
    monkeypatch.setattr(management, "get_free_memory", lambda *_args, **_kwargs: memory["free"])
    monkeypatch.setattr(management, "cleanup_models_gc", lambda: None)
    monkeypatch.setattr(management, "soft_empty_cache", lambda: None)
    monkeypatch.setattr(management, "vram_state", management.VRAMState.HIGH_VRAM)
    monkeypatch.setattr(management, "DISABLE_SMART_MEMORY", False)
    prepared = (model.model, {}, [branch])
    caplog.set_level("INFO", logger="comfy.vdn")
    wrapped = hybrid.make_prepare_sampling_memory_wrapper(SimpleNamespace(retain_buffers=True))
    assert wrapped(lambda *_args, **_kwargs: prepared, model, (1, 2, 3), {}) is prepared
    assert requests == [budget - GIB]
    assert memory["free"] == budget
    assert management.current_loaded_models == loaded
    assert "eviction_requested=True" in caplog.text
    assert "unloaded=0" in caplog.text


def test_core_minimum_inference_floor_is_retained(monkeypatch, budget):
    model = _model()
    monkeypatch.setattr(management, "minimum_inference_memory", lambda: 20 * GIB)
    monkeypatch.setattr(management, "get_free_memory", lambda _device: budget)
    requests = []
    monkeypatch.setattr(management, "free_memory",
                        lambda required, *_args, **_kwargs: requests.append(required) or [])
    prepared = (model.model, {}, [])
    wrapped = hybrid.make_prepare_sampling_memory_wrapper(SimpleNamespace(retain_buffers=True))
    assert wrapped(lambda *_args, **_kwargs: prepared, model, (1, 2, 3), {}) is prepared
    assert requests == [20 * GIB + RETAINED_SAMPLING_HEADROOM_BYTES]


@pytest.mark.parametrize("retained,device", [(False, "cuda"), (True, "cpu")])
def test_inactive_admission_delegates_without_memory_work(monkeypatch, retained, device):
    model = _model(device)
    for function in ("free_memory", "get_free_memory"):
        monkeypatch.setattr(management, function, lambda *_args, **_kwargs: pytest.fail(
            "Inactive admission must not inspect or evict GPU memory"))
    prepared = object()
    wrapped = hybrid.make_prepare_sampling_memory_wrapper(SimpleNamespace(retain_buffers=retained))
    assert wrapped(lambda *_args, **_kwargs: prepared, model, (1, 1, 1), {}) is prepared


def test_failed_core_preparation_propagates_without_admission(monkeypatch, caplog, budget):
    model = _model()
    monkeypatch.setattr(management, "free_memory", lambda *_args, **_kwargs: pytest.fail(
        "Admission must not run after failed Core preparation"))
    clock = iter([1.0, 1.25])
    monkeypatch.setattr(hybrid, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    caplog.set_level("INFO", logger="comfy.vdn")
    failure = RuntimeError("native prepare failed")

    def executor(*_args, **_kwargs):
        raise failure

    wrapped = hybrid.make_prepare_sampling_memory_wrapper(SimpleNamespace(retain_buffers=True))
    with pytest.raises(RuntimeError) as caught:
        wrapped(executor, model, (), {},
                model_options={"transformer_options": {"h3_flow_stage": "high"}})
    assert caught.value is failure
    assert "stage=high success=False prepare_elapsed_ms=250.000" in caplog.text
    assert "sampling admission" not in caplog.text


def test_admission_shares_the_auto_retention_allowance():
    assert policy.RETAINED_SAMPLING_HEADROOM_BYTES == RETAINED_SAMPLING_HEADROOM_BYTES


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_admission_failure_cleans_prepared_execution_state(monkeypatch, budget, cleanup_fails):
    model, branch = _model(), _model()
    conds = {"positive": []}
    prepared = (model.model, conds, [branch])
    failure = RuntimeError("admission failed")
    cleaned = []
    monkeypatch.setattr(management, "get_free_memory", lambda _device: 0)

    def free_memory(*_args, **_kwargs):
        raise failure

    def cleanup(actual_conds, models):
        assert actual_conds is conds and models is prepared[2]
        cleaned.append(True)
        if cleanup_fails:
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(management, "free_memory", free_memory)
    monkeypatch.setattr(hybrid.comfy.sampler_helpers, "cleanup_models", cleanup)
    wrapped = hybrid.make_prepare_sampling_memory_wrapper(SimpleNamespace(retain_buffers=True))
    with pytest.raises(RuntimeError) as caught:
        wrapped(lambda *_args, **_kwargs: prepared, model, (1, 2, 3), conds)
    assert caught.value is failure
    assert cleaned == [True]
