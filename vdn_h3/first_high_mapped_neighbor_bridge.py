"""Resolve the loaded Sol-H3 mapped-neighbor M companion fail-closed."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable

_MODULE_NAME = "sol_h3.first_high_mapped_neighbor_diagnostic"
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
            raise RuntimeError("mapped-neighbor M requires the loaded Sol-H3 M companion") from exc
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
        raise RuntimeError("mapped-neighbor M Sol companion has no resolvable source file")
    if len(by_file) != 1:
        raise RuntimeError(f"mapped-neighbor M Sol companion resolves ambiguously: {sorted(by_file)}")
    return next(iter(by_file.values()))


def _functions() -> dict[str, Callable[..., Any]]:
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED
    module = _resolve_module()
    names = ("enter_mapped_group", "exit_mapped_group")
    result = {name: getattr(module, name, None) for name in names}
    if any(not callable(value) for value in result.values()):
        raise RuntimeError("mapped-neighbor M Sol companion is missing its group-context API")
    _RESOLVED = result
    return result


def enter_mapped_group(options: dict[str, Any], metadata: dict[str, Any]):
    return _functions()["enter_mapped_group"](options, metadata)


def exit_mapped_group(token) -> None:
    _functions()["exit_mapped_group"](token)


__all__ = ["enter_mapped_group", "exit_mapped_group"]
