from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import comfy.cli_args
import comfy.ldm.minimax.model as minimax_model
import comfy.model_base
import comfy.model_prefetch
from comfy.patcher_extension import WrappersMP

from vdn_h3 import compiler_guard, hybrid


@pytest.fixture(autouse=True)
def _reset_guard(monkeypatch):
    monkeypatch.setattr(compiler_guard, "_active_owned_guards", 0)
    monkeypatch.setattr(compiler_guard, "_warned", False)
    old = getattr(comfy.cli_args.args, "disable_comfy_compiler", None)
    if old is not None:
        comfy.cli_args.args.disable_comfy_compiler = False
    yield
    compiler_guard._active_owned_guards = 0
    if old is not None:
        comfy.cli_args.args.disable_comfy_compiler = old


def test_guard_is_noop_without_affected_stack(monkeypatch):
    monkeypatch.setattr(compiler_guard, "_compiler_stack_present", lambda: False)
    comfy.cli_args.args.disable_comfy_compiler = False
    with compiler_guard.disabled_for_vdn() as owns:
        assert owns is False
        assert comfy.cli_args.args.disable_comfy_compiler is False
    assert comfy.cli_args.args.disable_comfy_compiler is False


def test_guard_restores_after_exception(monkeypatch):
    monkeypatch.setattr(compiler_guard, "_compiler_stack_present", lambda: True)
    comfy.cli_args.args.disable_comfy_compiler = False
    with pytest.raises(RuntimeError, match="boom"):
        with compiler_guard.disabled_for_vdn() as owns:
            assert owns is True
            assert comfy.cli_args.args.disable_comfy_compiler is True
            raise RuntimeError("boom")
    assert comfy.cli_args.args.disable_comfy_compiler is False
    assert compiler_guard._active_owned_guards == 0


def test_guard_preserves_user_disabled_setting(monkeypatch):
    monkeypatch.setattr(compiler_guard, "_compiler_stack_present", lambda: True)
    comfy.cli_args.args.disable_comfy_compiler = True
    with compiler_guard.disabled_for_vdn() as owns:
        assert owns is False
        assert comfy.cli_args.args.disable_comfy_compiler is True
    assert comfy.cli_args.args.disable_comfy_compiler is True


def test_nested_owned_guards_restore_only_after_outer_exit(monkeypatch):
    monkeypatch.setattr(compiler_guard, "_compiler_stack_present", lambda: True)
    comfy.cli_args.args.disable_comfy_compiler = False
    with compiler_guard.disabled_for_vdn() as outer:
        assert outer is True
        assert compiler_guard._active_owned_guards == 1
        with compiler_guard.disabled_for_vdn() as inner:
            assert inner is True
            assert compiler_guard._active_owned_guards == 2
            assert comfy.cli_args.args.disable_comfy_compiler is True
        assert compiler_guard._active_owned_guards == 1
        assert comfy.cli_args.args.disable_comfy_compiler is True
    assert compiler_guard._active_owned_guards == 0
    assert comfy.cli_args.args.disable_comfy_compiler is False


def test_apply_model_wrapper_scopes_switch_to_one_evaluation(monkeypatch):
    monkeypatch.setattr(compiler_guard, "_compiler_stack_present", lambda: True)
    comfy.cli_args.args.disable_comfy_compiler = False
    seen = []

    def executor(*args, **kwargs):
        seen.append((args, kwargs, comfy.cli_args.args.disable_comfy_compiler))
        return "out"

    wrapper = compiler_guard.make_apply_model_wrapper()
    assert wrapper(executor, 1, 2, transformer_options={}) == "out"
    assert seen == [((1, 2), {"transformer_options": {}}, True)]
    assert comfy.cli_args.args.disable_comfy_compiler is False
    assert compiler_guard._active_owned_guards == 0


def test_apply_model_wrapper_restores_after_exception(monkeypatch):
    monkeypatch.setattr(compiler_guard, "_compiler_stack_present", lambda: True)
    comfy.cli_args.args.disable_comfy_compiler = False

    def executor(*args, **kwargs):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        compiler_guard.make_apply_model_wrapper()(executor)
    assert comfy.cli_args.args.disable_comfy_compiler is False
    assert compiler_guard._active_owned_guards == 0


def test_apply_model_wrapper_preserves_user_disabled_setting(monkeypatch):
    monkeypatch.setattr(compiler_guard, "_compiler_stack_present", lambda: True)
    comfy.cli_args.args.disable_comfy_compiler = True
    assert compiler_guard.make_apply_model_wrapper()(lambda: "out") == "out"
    assert comfy.cli_args.args.disable_comfy_compiler is True
    assert compiler_guard._active_owned_guards == 0


class _Patcher:
    """The ModelPatcher surface ``apply_vdn`` uses, with Core's wrapper registry."""

    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model
        self.object_patches = {}
        self.wrappers = {}

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.diffusion_model

    def add_object_patch(self, key, obj):
        self.object_patches[key] = obj

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)


def _vdn_patched_model():
    attn = SimpleNamespace(heads=1, head_dim=1, qkv_proj=object(), out_proj=object(),
                           q_norm=object(), k_norm=object())
    diffusion_model = SimpleNamespace(blocks=[SimpleNamespace(attn=attn)])
    patcher = _Patcher(diffusion_model)
    hybrid.apply_vdn(patcher, SimpleNamespace(branches=[None], cfg={}, retain_buffers=False))
    return patcher


def _run_core_apply_model(monkeypatch, apply_model_wrappers):
    """Drive Core's apply_model -> MiniMax-H3 forward and record the graph decision."""
    decisions = []
    opened = []

    def malloc_graph_enabled(device):
        enabled = not comfy.cli_args.args.disable_comfy_compiler
        decisions.append(enabled)
        return enabled

    monkeypatch.setattr(comfy.model_prefetch, "malloc_graph_enabled", malloc_graph_enabled)
    monkeypatch.setattr(comfy.model_prefetch, "malloc_graph_begin", lambda *a: opened.append(True))
    monkeypatch.setattr(comfy.model_prefetch, "malloc_graph_end", lambda *a: None)

    x = [torch.zeros(1, 1, 1, 1, 1), torch.zeros(1, 1, 1)]

    def _forward(x, timestep, context, transformer_options, **kwargs):
        return [x[0] + 1, x[1] + 1]

    diffusion_model = SimpleNamespace(_forward=_forward)

    def _apply_model(x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):
        # BaseModel._apply_model's call into the native MiniMax-H3 forward.
        return minimax_model.MiniMaxH3Model.forward(
            diffusion_model, x, t, None, transformer_options={})

    base_model = SimpleNamespace(_apply_model=_apply_model)
    options = {"wrappers": {WrappersMP.APPLY_MODEL: apply_model_wrappers}}
    out = comfy.model_base.BaseModel.apply_model(
        base_model, x, torch.ones(1), transformer_options=options)
    assert torch.equal(out[0], x[0] + 1) and torch.equal(out[1], x[1] + 1)
    return decisions, opened


def test_apply_vdn_registers_compiler_guard_ahead_of_graph_decision(monkeypatch):
    monkeypatch.setattr(compiler_guard, "_compiler_stack_present", lambda: True)
    comfy.cli_args.args.disable_comfy_compiler = False
    patcher = _vdn_patched_model()
    guard = patcher.wrappers[WrappersMP.APPLY_MODEL]
    assert list(guard) == ["vdn_h3_compiler_guard"]

    decisions, opened = _run_core_apply_model(monkeypatch, guard)

    assert decisions == [False]
    assert opened == []
    assert comfy.cli_args.args.disable_comfy_compiler is False
    assert compiler_guard._active_owned_guards == 0


def test_unguarded_core_path_opens_graph(monkeypatch):
    # Control for the test above: the same Core path opens a graph without VDN.
    comfy.cli_args.args.disable_comfy_compiler = False
    decisions, opened = _run_core_apply_model(monkeypatch, {})
    assert decisions == [True]
    assert opened == [True]


def test_compiler_guard_is_not_a_diffusion_model_wrapper():
    patcher = _vdn_patched_model()
    assert list(patcher.wrappers[WrappersMP.DIFFUSION_MODEL]) == ["vdn_h3"]
    assert not hasattr(hybrid.make_layout_wrapper, "_vdn_compiler_guard_installed")
