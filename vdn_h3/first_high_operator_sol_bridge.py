"""Diagnostic-only Sol owner bridge for first-high operator comparison W.

The VDN W overlay computes the native local SDPA result.  After that computation
succeeds, this bridge forwards the completed route to the live Sol-H3 BlockPatch
provider owner.  No ordinary VDN path reaches this hook because the W request is
required before the local diagnostic path executes.
"""
from __future__ import annotations

from typing import Any

_INSTALLED = False


def _record_backend_route(options: dict[str, Any], block_index: int, route: str, request: dict[str, Any]) -> None:
    try:
        from sol_h3.first_high_operator_diagnostic import record_native_local
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("first-high operator diagnostic requires the Sol-H3 W companion") from exc
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
