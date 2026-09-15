from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from vdn_h3 import first_high_sol_local_bridge as bridge
from vdn_h3 import first_high_sol_local_diagnostic as e


def _request():
    return (
        ("api", 1),
        ("capture_id", "capture-e"),
        ("mode", "all_selected_e"),
        ("stage", "high"),
        ("logical_call_limit", 1),
        ("sigma", 0.8780487775802612),
        ("target_shapes_digest", "a" * 64),
        ("source_contract_digest", "b" * 64),
    )


def test_request_parser_accepts_only_exact_immutable_e_contract():
    parsed = e.parse_request({e.REQUEST_KEY: _request()})
    assert parsed is not None
    assert parsed["mode"] == "all_selected_e"
    assert parsed["stage"] == "high"
    assert parsed["logical_call_limit"] == 1

    with pytest.raises(RuntimeError, match="immutable exact-field tuple"):
        e.parse_request({e.REQUEST_KEY: dict(_request())})

    wrong_order = list(_request())
    wrong_order[1], wrong_order[2] = wrong_order[2], wrong_order[1]
    with pytest.raises(RuntimeError, match="does not match API 1"):
        e.parse_request({e.REQUEST_KEY: tuple(wrong_order)})

    bad_mode = list(_request())
    bad_mode[2] = ("mode", "native_window")
    with pytest.raises(RuntimeError, match="unsupported"):
        e.parse_request({e.REQUEST_KEY: tuple(bad_mode)})

    bad_digest = list(_request())
    bad_digest[6] = ("target_shapes_digest", "A" * 64)
    with pytest.raises(RuntimeError, match="lowercase SHA-256"):
        e.parse_request({e.REQUEST_KEY: tuple(bad_digest)})


def _synthetic_sol_module(name: str, path: Path, calls: list[tuple]) -> ModuleType:
    path.write_text("# synthetic Sol E companion\n", encoding="utf-8")
    module = ModuleType(name)
    module.__file__ = str(path)

    def enter_local_group(*args):
        calls.append(("enter",) + args)
        return "token"

    def exit_local_group(*args):
        calls.append(("exit",) + args)

    def parse_request(*args):
        calls.append(("parse",) + args)
        return {"mode": "all_selected_e"}

    module.enter_local_group = enter_local_group
    module.exit_local_group = exit_local_group
    module.parse_request = parse_request
    return module


def _hide_loaded_sol_companions(monkeypatch) -> None:
    bare = bridge._MODULE_NAME
    suffix = f".{bare}"
    for name in tuple(sys.modules):
        if name == bare or name.endswith(suffix):
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_e_bridge_resolves_loaded_comfyui_namespaced_sol_companion(monkeypatch, tmp_path):
    bare = bridge._MODULE_NAME
    synthetic_name = f"custom_nodes.synthetic_sol.{bare}"
    calls: list[tuple] = []
    module = _synthetic_sol_module(synthetic_name, tmp_path / "first_high_sol_local_diagnostic.py", calls)

    _hide_loaded_sol_companions(monkeypatch)
    monkeypatch.setitem(sys.modules, synthetic_name, module)
    monkeypatch.setattr(bridge, "_RESOLVED", None)

    def deny_import(name: str):
        raise AssertionError(f"bare import must not be used for loaded namespaced Sol companion: {name}")

    monkeypatch.setattr(bridge.importlib, "import_module", deny_import)
    options = {"marker": object()}
    metadata = {"block_index": 2, "group_index": 0}
    token = bridge.enter_local_group(options, metadata)
    parsed = bridge.parse_sol_request(options)
    bridge.exit_local_group(token)

    assert token == "token"
    assert parsed == {"mode": "all_selected_e"}
    assert calls[0] == ("enter", options, metadata)
    assert calls[1] == ("parse", options)
    assert calls[2] == ("exit", "token")


def test_e_bridge_fails_closed_on_distinct_namespaced_sources(monkeypatch, tmp_path):
    bare = bridge._MODULE_NAME
    first = _synthetic_sol_module(
        f"custom_nodes.synthetic_sol_a.{bare}", tmp_path / "a.py", []
    )
    second = _synthetic_sol_module(
        f"custom_nodes.synthetic_sol_b.{bare}", tmp_path / "b.py", []
    )

    _hide_loaded_sol_companions(monkeypatch)
    monkeypatch.setitem(sys.modules, first.__name__, first)
    monkeypatch.setitem(sys.modules, second.__name__, second)
    monkeypatch.setattr(bridge, "_RESOLVED", None)
    with pytest.raises(RuntimeError, match="resolves ambiguously"):
        bridge._functions()


def test_e_bridge_deduplicates_same_file_aliases_and_caches(monkeypatch, tmp_path):
    bare = bridge._MODULE_NAME
    path = tmp_path / "shared.py"
    calls: list[tuple] = []
    first = _synthetic_sol_module(f"custom_nodes.synthetic_sol_a.{bare}", path, calls)
    second = ModuleType(f"custom_nodes.synthetic_sol_b.{bare}")
    second.__file__ = str(path)
    second.enter_local_group = first.enter_local_group
    second.exit_local_group = first.exit_local_group
    second.parse_request = first.parse_request

    _hide_loaded_sol_companions(monkeypatch)
    monkeypatch.setitem(sys.modules, first.__name__, first)
    monkeypatch.setitem(sys.modules, second.__name__, second)
    monkeypatch.setattr(bridge, "_RESOLVED", None)
    first_resolution = bridge._functions()
    second_resolution = bridge._functions()

    assert first_resolution is second_resolution
    assert first_resolution["enter_local_group"] is first.enter_local_group
