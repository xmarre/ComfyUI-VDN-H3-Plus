from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import comfy.cli_args

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


def test_apply_model_wrapper_encloses_outer_forward(monkeypatch):
    monkeypatch.setattr(compiler_guard, "_compiler_stack_present", lambda: True)
    comfy.cli_args.args.disable_comfy_compiler = False

    def execute(*args, **kwargs):
        assert comfy.cli_args.args.disable_comfy_compiler is True
        return "done"

    assert compiler_guard.apply_model_wrapper(execute, 1, value=2) == "done"
    assert comfy.cli_args.args.disable_comfy_compiler is False


def test_apply_vdn_registers_outer_compiler_and_inner_layout_wrappers():
    attn = SimpleNamespace(
        heads=2,
        head_dim=4,
        qkv_proj=torch.nn.Linear(8, 24, bias=False),
        out_proj=torch.nn.Linear(8, 8, bias=False),
        q_norm=torch.nn.RMSNorm(4, eps=1e-6),
        k_norm=torch.nn.RMSNorm(4, eps=1e-6),
    )
    dm = SimpleNamespace(blocks=[SimpleNamespace(attn=attn)])
    wrappers = {}
    patcher = SimpleNamespace(
        object_patches={},
        get_model_object=lambda key: dm,
        add_object_patch=lambda *args: None,
        add_wrapper_with_key=lambda kind, key, fn: wrappers.update({kind: fn}),
    )
    state = hybrid.VDNState("test", {}, [SimpleNamespace()], 2, 4)

    hybrid.apply_vdn(patcher, state)

    assert wrappers[hybrid.WrappersMP.APPLY_MODEL] is compiler_guard.apply_model_wrapper
    assert hybrid.WrappersMP.DIFFUSION_MODEL in wrappers
