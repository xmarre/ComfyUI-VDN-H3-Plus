"""Scoped compatibility guard for Comfy's AIMDO model compiler.

Upstream VDN-H3 identified a Comfy build family where the AIMDO malloc-graph
compiler cannot execute VDN-patched MiniMax-H3 forwards. Comfy currently exposes
only a process-global ``args.disable_comfy_compiler`` switch, so this module keeps
the unavoidable mutation as narrow and reversible as possible:

* detection is lazy and fail-open on older Comfy builds;
* a user-provided ``--disable-comfy-compiler`` setting is never changed;
* VDN-owned disables are reference-counted across nested/overlapping VDN calls and
  restored in ``finally``;
* no Comfy function is monkey-patched and no unload hook is installed.

The guard is registered as an ``APPLY_MODEL`` wrapper. This boundary is intentional:
MiniMax starts its AIMDO allocation graph before ``DIFFUSION_MODEL`` wrappers run, so
guarding only VDN's layout wrapper is too late on affected builds.
"""
from __future__ import annotations

from contextlib import contextmanager
import logging
import threading

import comfy.cli_args

_log = logging.getLogger("comfy.vdn")
_lock = threading.RLock()
_active_owned_guards = 0
_warned = False


def _compiler_stack_present() -> bool:
    """Return whether this Comfy build contains the affected compiler stack."""
    try:
        args = comfy.cli_args.args
        if not hasattr(args, "disable_comfy_compiler"):
            return False
        import comfy.model_prefetch as model_prefetch

        aimdo = getattr(model_prefetch, "comfy_aimdo", None)
        return aimdo is not None and hasattr(aimdo, "malloc_graph")
    except Exception:
        return False


@contextmanager
def disabled_for_vdn():
    """Temporarily disable Comfy's compiler for one complete VDN model call.

    The CLI flag is process-global, so true concurrent non-VDN execution cannot be
    isolated by any consumer-side workaround. Reference counting prevents nested or
    overlapping VDN wrappers from restoring the flag while another VDN call owns it.
    """
    global _active_owned_guards, _warned

    args = comfy.cli_args.args
    owns = False
    if _compiler_stack_present():
        with _lock:
            if _active_owned_guards > 0:
                _active_owned_guards += 1
                owns = True
            elif not bool(getattr(args, "disable_comfy_compiler", False)):
                args.disable_comfy_compiler = True
                _active_owned_guards = 1
                owns = True
                if not _warned:
                    _warned = True
                    _log.warning(
                        "[vdn] this Comfy build's AIMDO model compiler is incompatible "
                        "with VDN-H3; disabling it only for VDN model calls")

    try:
        yield owns
    finally:
        if owns:
            with _lock:
                _active_owned_guards -= 1
                if _active_owned_guards == 0:
                    args.disable_comfy_compiler = False


def apply_model_wrapper(executor, *args, **kwargs):
    """Comfy ``APPLY_MODEL`` wrapper that encloses the outer MiniMax forward."""
    with disabled_for_vdn():
        return executor(*args, **kwargs)
