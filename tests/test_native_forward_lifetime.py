from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vdn_h3.hybrid import (
    VDNLayout,
    VDNState,
    make_prepare_sampling_memory_wrapper,
    make_vdn_forward,
)


class _TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 1
        self.head_dim = 2
        self.qkv_proj = nn.Linear(2, 6, bias=False)
        self.q_norm = nn.RMSNorm(2, eps=1e-6)
        self.k_norm = nn.RMSNorm(2, eps=1e-6)
        self.out_proj = nn.Identity()


class _RawQReadoutBranch:
    enable_text_state = False
    w = {}

    def readout(
        self,
        _weights,
        _xv,
        q_raw,
        _k_raw,
        _v_raw,
        _num_frames,
        _tokens_per_frame,
        _bounds,
        **_kwargs,
    ):
        return q_raw.reshape(q_raw.shape[0], -1)


def test_native_linear_complement_consumes_raw_views_without_activation_scratch(monkeypatch):
    torch.manual_seed(901)
    attn = _TinyAttention().requires_grad_(False)
    cfg = {
        "radius": 0,
        "chunk": 1,
        "anchor_frames": "none",
        "enable_softmax_gate": False,
        "linear_enabled": True,
    }
    state = VDNState(
        "native-lifetime-test",
        cfg,
        [_RawQReadoutBranch()],
        1,
        2,
        retain_buffers=True,
    )
    layout = VDNLayout(
        video_start=0,
        video_end=4,
        num_frames=4,
        tokens_per_frame=1,
        frame_size=(1, 1),
        text_start=0,
        text_len=0,
        seq_len=4,
        radius=0,
        chunk=1,
        anchor_frames="none",
    )
    events = []

    def weights_on(*_args, **kwargs):
        assert kwargs.get("prefetch_next") is False
        events.append("weights")
        return {"to_out_linear.weight": torch.eye(2)}

    def softmax(query, *_args, **_kwargs):
        events.append("softmax")
        return torch.zeros_like(query)

    state.weights_on = weights_on
    state.prefetch_next_weights = lambda *_args, **_kwargs: events.append("prefetch")
    monkeypatch.setattr(
        "vdn_h3.retained.window_softmax_grouped_runtime",
        softmax,
    )

    x = torch.randn(4, 2)
    with torch.no_grad():
        expected_q = attn.qkv_proj(x).split(2, dim=-1)[0].clone()

    with state.runtime.execution() as resources:
        # Simulate the retained raw-QKV payload left by the preceding partitioned
        # low/probe stage. Native high-grid entry must release it before qkv_proj
        # allocates the new target-grid projection.
        resources.activation_scratch(
            video_rows=3,
            text_rows=1,
            heads=1,
            head_dim=2,
            device=x.device,
            dtype=x.dtype,
        )
        assert resources.retained_counts()["activations"] == 1

        def forbidden_activation_scratch(*_args, **_kwargs):
            raise AssertionError(
                "native VDN forward must not preserve raw Q/K/V in activation scratch"
            )

        qkv_entry_activation_counts = []
        hook = attn.qkv_proj.register_forward_pre_hook(
            lambda *_args: qkv_entry_activation_counts.append(
                resources.retained_counts()["activations"]
            )
        )
        monkeypatch.setattr(resources, "activation_scratch", forbidden_activation_scratch)
        token = state._layout.set(layout)
        try:
            with torch.no_grad():
                got = make_vdn_forward(attn, state, 0)(x, transformer_options={})
        finally:
            state._layout.reset(token)
            hook.remove()

        assert qkv_entry_activation_counts == [0]
        assert resources.retained_counts()["activations"] == 0

    assert torch.equal(got, expected_q)
    assert events == ["weights", "softmax", "prefetch"]



def test_prepare_sampling_memory_admission_evicts_unrelated_models_before_executor(monkeypatch):
    state = SimpleNamespace(retain_buffers=True)
    model = SimpleNamespace(load_device=torch.device("cuda"))
    events = []

    class _Keep:
        def __init__(self, patcher):
            self.model = patcher

    monkeypatch.setattr("vdn_h3.hybrid._cuda_allocator_snapshot", lambda: {"ok": True})
    monkeypatch.setattr("comfy.model_management.LoadedModel", _Keep)

    def free_memory(required, device, keep_loaded):
        events.append(("free", required, device.type, keep_loaded[0].model is model))
        return ["text_encoder", "vae"]

    monkeypatch.setattr("comfy.model_management.free_memory", free_memory)

    def executor(*args, **kwargs):
        events.append((
            "executor",
            args[0] is model,
            kwargs["model_options"]["transformer_options"]["h3_flow_stage"],
        ))
        return "prepared"

    wrapped = make_prepare_sampling_memory_wrapper(state)
    result = wrapped(
        executor,
        model,
        (1, 1, 1),
        {},
        model_options={"transformer_options": {"h3_flow_stage": "high"}},
    )

    assert result == "prepared"
    assert events == [
        ("free", 1e30, "cuda", True),
        ("executor", True, "high"),
    ]


@pytest.mark.parametrize("retained,device", [(False, "cuda"), (True, "cpu")])
def test_prepare_sampling_memory_admission_is_inert_without_retained_buffers(monkeypatch, retained, device):
    state = SimpleNamespace(retain_buffers=retained)
    model = SimpleNamespace(load_device=torch.device(device))
    monkeypatch.setattr(
        "comfy.model_management.free_memory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("free_memory must not run when VDN buffers are not retained")
        ),
    )

    wrapped = make_prepare_sampling_memory_wrapper(state)
    assert wrapped(
        lambda *_args, **_kwargs: "prepared",
        model,
        (1, 1, 1),
        {},
        model_options={},
    ) == "prepared"


@pytest.mark.parametrize("changed_weights", [False, True])
def test_sampling_admission_keeps_resident_core_clones_for_native_switching(monkeypatch, caplog, changed_weights):
    from comfy.model_management import LoadedModel
    from comfy.model_patcher import ModelPatcher

    device = torch.device("cuda")
    base = ModelPatcher(nn.Linear(2, 2), load_device=device, offload_device=torch.device("cpu"))
    resident = base.clone()
    incoming = base.clone()
    if changed_weights:
        incoming.add_patches({"weight": (torch.ones_like(base.model.weight),)})
    unrelated = ModelPatcher(nn.Linear(2, 2), load_device=device, offload_device=torch.device("cpu"))
    loaded_resident = LoadedModel(resident)
    loaded_unrelated = LoadedModel(unrelated)
    cpu_clone = base.clone()
    cpu_clone.load_device = torch.device("cpu")
    loaded_cpu = LoadedModel(cpu_clone)
    # A dead weak reference must not become a keep-list entry.
    dead = SimpleNamespace(model=None)
    monkeypatch.setattr("comfy.model_management.current_loaded_models", [loaded_resident, loaded_unrelated, loaded_cpu, dead])
    monkeypatch.setattr("vdn_h3.hybrid._cuda_allocator_snapshot", lambda: None)
    clock = iter([1.0, 1.2, 2.0, 2.5])
    monkeypatch.setattr("vdn_h3.hybrid.time", SimpleNamespace(perf_counter=lambda: next(clock)))
    caplog.set_level("INFO", logger="comfy.vdn")

    assert incoming.is_clone(resident)
    assert loaded_resident != LoadedModel(incoming)
    assert (incoming.patches_uuid != resident.patches_uuid) is changed_weights
    events = []

    def free_memory(required, requested_device, keep_loaded):
        assert required == 1e30 and requested_device == device
        assert loaded_resident in keep_loaded
        assert loaded_unrelated not in keep_loaded and loaded_cpu not in keep_loaded and dead not in keep_loaded
        events.append("evict_unrelated")
        return [loaded_unrelated]

    monkeypatch.setattr("comfy.model_management.free_memory", free_memory)
    options = {"transformer_options": {"h3_flow_stage": "low"}}
    conds = {"positive": []}
    sentinel = object()

    def executor(model, shape, actual_conds, **kwargs):
        # Core remains responsible for applying differing patches and clone detach.
        assert model is incoming and actual_conds is conds and shape == (1, 2, 3)
        assert kwargs == {"model_options": options, "force_full_load": True, "force_offload": True}
        events.append("core_prepare")
        return sentinel

    wrapped = make_prepare_sampling_memory_wrapper(SimpleNamespace(retain_buffers=True))
    assert wrapped(executor, incoming, (1, 2, 3), conds, model_options=options,
                   force_full_load=True, force_offload=True) is sentinel
    assert events == ["evict_unrelated", "core_prepare"]
    assert "kept_resident_h3=1 eviction_elapsed_ms=200.000" in caplog.text
    assert "stage=low success=True prepare_elapsed_ms=500.000" in caplog.text


def test_sampling_preparation_failure_is_reported_and_propagated(monkeypatch, caplog):
    from comfy.model_patcher import ModelPatcher

    model = ModelPatcher(nn.Linear(2, 2), load_device=torch.device("cuda"),
                         offload_device=torch.device("cpu"))
    monkeypatch.setattr("comfy.model_management.current_loaded_models", [])
    monkeypatch.setattr("vdn_h3.hybrid._cuda_allocator_snapshot", lambda: None)
    monkeypatch.setattr("comfy.model_management.free_memory", lambda *_args, **_kwargs: [])
    clock = iter([1.0, 1.0, 2.0, 2.25])
    monkeypatch.setattr("vdn_h3.hybrid.time", SimpleNamespace(perf_counter=lambda: next(clock)))
    caplog.set_level("INFO", logger="comfy.vdn")
    failure = RuntimeError("native prepare failed")

    def executor(*_args, **_kwargs):
        raise failure

    wrapped = make_prepare_sampling_memory_wrapper(SimpleNamespace(retain_buffers=True))
    with pytest.raises(RuntimeError) as caught:
        wrapped(executor, model, (), {},
                model_options={"transformer_options": {"h3_flow_stage": "high"}})
    assert caught.value is failure
    assert "stage=high success=False prepare_elapsed_ms=250.000" in caplog.text
