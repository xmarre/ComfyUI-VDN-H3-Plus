"""Diagnostic-only Sol owner bridge for first-high operator comparison W.

The VDN W overlay computes the native local SDPA result. After that computation
succeeds, this bridge forwards the completed route to the live Sol-H3 BlockPatch
provider owner. No ordinary VDN path reaches this hook because the W request is
required before the local diagnostic path executes.

ComfyUI directory custom nodes are loaded under generated package names. The Sol
companion can therefore be present as ``<generated>.sol_h3...`` without the bare
``sol_h3`` package being importable. Resolve the already-loaded companion by
exact/suffix module identity and fail closed on distinct source files; use a bare
import only for ordinary package/test environments where no loaded candidate
exists.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable

_INSTALLED = False
_SOL_DIAGNOSTIC_MODULE = "sol_h3.first_high_operator_diagnostic"
_SOL_RECORD_NATIVE_LOCAL: Callable[..., None] | None = None


def _resolve_sol_record_native_local() -> Callable[..., None]:
    global _SOL_RECORD_NATIVE_LOCAL
    if _SOL_RECORD_NATIVE_LOCAL is not None:
        return _SOL_RECORD_NATIVE_LOCAL

    suffix = f".{_SOL_DIAGNOSTIC_MODULE}"
    modules = [
        module
        for loaded_name, module in tuple(sys.modules.items())
        if module is not None and (loaded_name == _SOL_DIAGNOSTIC_MODULE or loaded_name.endswith(suffix))
    ]
    if not modules:
        try:
            modules = [importlib.import_module(_SOL_DIAGNOSTIC_MODULE)]
        except Exception as exc:
            raise RuntimeError("first-high operator diagnostic requires the loaded Sol-H3 W companion") from exc

    by_file: dict[str, Any] = {}
    for module in modules:
        raw_file = getattr(module, "__file__", None)
        if not isinstance(raw_file, str) or not raw_file:
            continue
        try:
            path = Path(raw_file).resolve(strict=True)
        except OSError:
            continue
        by_file[str(path)] = module

    if not by_file:
        raise RuntimeError("first-high operator diagnostic Sol-H3 W companion has no resolvable source file")
    if len(by_file) != 1:
        raise RuntimeError(
            "first-high operator diagnostic Sol-H3 W companion resolves ambiguously: "
            f"{sorted(by_file)}"
        )

    module = next(iter(by_file.values()))
    record_native_local = getattr(module, "record_native_local", None)
    if not callable(record_native_local):
        raise RuntimeError("first-high operator diagnostic Sol-H3 W companion lacks record_native_local")
    _SOL_RECORD_NATIVE_LOCAL = record_native_local
    return record_native_local


def _record_backend_route(options: dict[str, Any], block_index: int, route: str, request: dict[str, Any]) -> None:
    record_native_local = _resolve_sol_record_native_local()
    record_native_local(options, block_index, route, request)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from vdn_h3 import first_high_operator_diagnostic as diagnostic

    current = diagnostic._append_backend_receipt
    if getattr(current, "__module__", None) != diagnostic.__name__:
        raise RuntimeError("first-high operator diagnostic backend receipt owner was already replaced")
    diagnostic._append_backend_receipt = _record_backend_route
    _INSTALLED = True


__all__ = ["install"]
