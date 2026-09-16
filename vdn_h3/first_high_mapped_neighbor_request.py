"""Retarget the reviewed VDN E request parser to the separately versioned M mode."""
from __future__ import annotations

from typing import Any, Mapping

from . import first_high_sol_local_diagnostic as diagnostic
from .first_high_sol_local_bridge import parse_sol_request

MODE = "mapped_neighbor_m"


def parse_request(options: Mapping[str, Any] | None):
    options = options or {}
    if options.get(diagnostic.REQUEST_KEY) is None:
        return None
    request = parse_sol_request(dict(options))
    if not isinstance(request, dict) or request.get("mode") != MODE:
        raise RuntimeError("mapped-neighbor M VDN request differs from the Sol companion")
    return request


diagnostic.parse_request = parse_request

__all__ = ["MODE", "parse_request"]
