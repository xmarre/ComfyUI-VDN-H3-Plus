from __future__ import annotations

import pytest

import comfy.cli_args

from vdn_h3 import compiler_guard


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


def test_outer_sample_wrapper_disables_for_the_wrapped_call(monkeypatch):
    monkeypatch.setattr(compiler_guard, "_compiler_stack_present", lambda: True)
    comfy.cli_args.args.disable_comfy_compiler = False
    seen = []

    def executor(*args, **kwargs):
        seen.append((args, kwargs,
                     comfy.cli_args.args.disable_comfy_compiler))
        return "out"

    wrapper = compiler_guard.make_outer_sample_wrapper()
    assert wrapper(executor, 1, 2, seed=3) == "out"
    assert seen == [((1, 2), {"seed": 3}, True)]
    assert comfy.cli_args.args.disable_comfy_compiler is False
    assert compiler_guard._active_owned_guards == 0


def test_outer_sample_wrapper_restores_after_exception(monkeypatch):
    monkeypatch.setattr(compiler_guard, "_compiler_stack_present", lambda: True)
    comfy.cli_args.args.disable_comfy_compiler = False

    def executor(*args, **kwargs):
        raise RuntimeError("boom")

    wrapper = compiler_guard.make_outer_sample_wrapper()
    with pytest.raises(RuntimeError, match="boom"):
        wrapper(executor)
    assert comfy.cli_args.args.disable_comfy_compiler is False
    assert compiler_guard._active_owned_guards == 0


def test_outer_sample_wrapper_preserves_user_disabled_setting(monkeypatch):
    monkeypatch.setattr(compiler_guard, "_compiler_stack_present", lambda: True)
    comfy.cli_args.args.disable_comfy_compiler = True

    wrapper = compiler_guard.make_outer_sample_wrapper()
    assert wrapper(lambda: "out") == "out"
    assert comfy.cli_args.args.disable_comfy_compiler is True
    assert compiler_guard._active_owned_guards == 0
