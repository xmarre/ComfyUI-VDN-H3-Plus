"""Resolve the loaded Sol-H3 E diagnostic companion without assuming package names."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable

_MODULE_NAME = "sol_h3.first_high_sol_local_diagnostic"
_RESOLVED: dict[str, Callable[..., Any]] | None = None


def _resolve_module():
    suffix = f".{_MODULE_NAME}"
    modules = [
        module
        for loaded_name, module in tuple(sys.modules.items())
        if module is not None and (loaded_name == _MODULE_NAME or loaded_name.endswith(suffix))
    ]
    if not modules:
        try:
            modules = [importlib.import_module(_MODULE_NAME)]
        except Exception as exc:
            raise RuntimeError("first-high Sol-local E requires the loaded Sol-H3 diagnostic companion") from exc
    by_file: dict[str, Any] = {}
    for module in modules:
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str) or not raw:
            continue
        try:
            path = Path(raw).resolve(strict=True)
        except OSError:
            continue
        by_file[str(path)] = module
    if not by_file:
        raise RuntimeError("first-high Sol-local E companion has no resolvable source file")
    if len(by_file) != 1:
        raise RuntimeError(f"first-high Sol-local E companion resolves ambiguously: {sorted(by_file)}")
    return next(iter(by_file.values()))


def _functions() -> dict[str, Callable[..., Any]]:
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED
    module = _resolve_module()
    names = ("enter_local_group", "exit_local_group", "parse_request")
    result = {name: getattr(module, name, None) for name in names}
    if any(not callable(value) for value in result.values()):
        raise RuntimeError("first-high Sol-local E companion is missing its group-context API")
    _RESOLVED = result
    return result


def enter_local_group(options: dict[str, Any], metadata: dict[str, Any]):
    return _functions()["enter_local_group"](options, metadata)


def exit_local_group(token) -> None:
    _functions()["exit_local_group"](token)


def parse_sol_request(options: dict[str, Any]):
    return _functions()["parse_request"](options)


__all__ = ["enter_local_group", "exit_local_group", "parse_sol_request"]
