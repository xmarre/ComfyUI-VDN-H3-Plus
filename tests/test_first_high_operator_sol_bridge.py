from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from vdn_h3 import first_high_operator_sol_bridge as bridge


def _synthetic_sol_module(name: str, path: Path, calls: list[tuple]) -> ModuleType:
    path.write_text("# synthetic Sol W companion\n", encoding="utf-8")
    module = ModuleType(name)
    module.__file__ = str(path)

    def record_native_local(*args):
        calls.append(args)

    module.record_native_local = record_native_local
    return module


def test_bridge_resolves_loaded_comfyui_namespaced_sol_module_without_bare_import(monkeypatch, tmp_path):
    bare = bridge._SOL_DIAGNOSTIC_MODULE
    synthetic_name = f"custom_nodes.synthetic_sol.{bare}"
    calls: list[tuple] = []
    module = _synthetic_sol_module(synthetic_name, tmp_path / "first_high_operator_diagnostic.py", calls)

    original_bare = sys.modules.pop(bare, None)
    monkeypatch.setitem(sys.modules, synthetic_name, module)
    monkeypatch.setattr(bridge, "_SOL_RECORD_NATIVE_LOCAL", None)

    def deny_import(name: str):
        raise AssertionError(f"bare import must not be used for loaded namespaced Sol companion: {name}")

    monkeypatch.setattr(bridge.importlib, "import_module", deny_import)
    try:
        options = {"marker": object()}
        request = {"capture_id": "capture-w"}
        bridge._record_backend_route(options, 7, "vdn_local_native_window_w", request)
    finally:
        if original_bare is not None:
            sys.modules[bare] = original_bare

    assert calls == [(options, 7, "vdn_local_native_window_w", request)]


def test_bridge_fails_closed_on_distinct_namespaced_sol_sources(monkeypatch, tmp_path):
    bare = bridge._SOL_DIAGNOSTIC_MODULE
    first_name = f"custom_nodes.synthetic_sol_a.{bare}"
    second_name = f"custom_nodes.synthetic_sol_b.{bare}"
    first = _synthetic_sol_module(first_name, tmp_path / "a.py", [])
    second = _synthetic_sol_module(second_name, tmp_path / "b.py", [])

    original_bare = sys.modules.pop(bare, None)
    monkeypatch.setitem(sys.modules, first_name, first)
    monkeypatch.setitem(sys.modules, second_name, second)
    monkeypatch.setattr(bridge, "_SOL_RECORD_NATIVE_LOCAL", None)
    try:
        with pytest.raises(RuntimeError, match="resolves ambiguously"):
            bridge._resolve_sol_record_native_local()
    finally:
        if original_bare is not None:
            sys.modules[bare] = original_bare


def test_bridge_deduplicates_aliases_to_same_source_file(monkeypatch, tmp_path):
    bare = bridge._SOL_DIAGNOSTIC_MODULE
    first_name = f"custom_nodes.synthetic_sol_a.{bare}"
    second_name = f"custom_nodes.synthetic_sol_b.{bare}"
    path = tmp_path / "shared.py"
    calls: list[tuple] = []
    first = _synthetic_sol_module(first_name, path, calls)
    second = ModuleType(second_name)
    second.__file__ = str(path)
    second.record_native_local = first.record_native_local

    original_bare = sys.modules.pop(bare, None)
    monkeypatch.setitem(sys.modules, first_name, first)
    monkeypatch.setitem(sys.modules, second_name, second)
    monkeypatch.setattr(bridge, "_SOL_RECORD_NATIVE_LOCAL", None)
    try:
        resolved = bridge._resolve_sol_record_native_local()
        resolved({}, 3, "vdn_local_native_full_w", {})
    finally:
        if original_bare is not None:
            sys.modules[bare] = original_bare

    assert len(calls) == 1
    assert calls[0][1:3] == (3, "vdn_local_native_full_w")


def test_bridge_regular_package_fallback_is_fail_closed_and_cached(monkeypatch, tmp_path):
    bare = bridge._SOL_DIAGNOSTIC_MODULE
    calls: list[tuple] = []
    module = _synthetic_sol_module(bare, tmp_path / "fallback.py", calls)
    original_bare = sys.modules.pop(bare, None)
    monkeypatch.setattr(bridge, "_SOL_RECORD_NATIVE_LOCAL", None)
    imports: list[str] = []

    def fake_import(name: str):
        imports.append(name)
        return module

    monkeypatch.setattr(bridge.importlib, "import_module", fake_import)
    try:
        first = bridge._resolve_sol_record_native_local()
        second = bridge._resolve_sol_record_native_local()
    finally:
        if original_bare is not None:
            sys.modules[bare] = original_bare

    assert first is second
    assert imports == [bare]
