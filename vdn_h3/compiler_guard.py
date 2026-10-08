"""Scoped compatibility guard for Comfy's AIMDO model compiler.

Upstream VDN-H3 v1.4.3 identified a Comfy build family where the AIMDO
malloc-graph compiler cannot execute VDN-patched MiniMax-H3 forwards.  Comfy
currently exposes only a process-global ``args.disable_comfy_compiler`` switch,
so this module keeps the unavoidable mutation as narrow and reversible as
possible:

* detection is lazy and fail-open on older Comfy builds;
* a user-provided ``--disable-comfy-compiler`` setting is never changed;
* VDN-owned disables are reference-counted across nested/overlapping VDN
  wrappers and restored in ``finally``;
* no Comfy function is monkey-patched and no unload hook is installed.

The guard is an APPLY_MODEL wrapper registered by ``vdn_h3.hybrid.apply_vdn`` on
the VDN-patched model only.  Placement matters: the native MiniMax-H3
``forward`` calls ``model_prefetch.malloc_graph_enabled`` and, when it returns
True, ``malloc_graph_begin`` *before* it runs its DIFFUSION_MODEL wrappers.  A
DIFFUSION_MODEL wrapper therefore flips the switch after the graph is already
open, and nothing inside the forward reads the switch again, so it cannot
prevent recording.  ``BaseModel.apply_model`` runs APPLY_MODEL wrappers around
``_apply_model``, which is what calls the diffusion model, so the switch is
active when ``forward`` asks and is restored after that one model evaluation.
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

        # Match upstream v1.4.3's final detector. ``import comfy_aimdo.malloc_graph``
        # binds the package on current Comfy; older builds lack this compiler path.
        aimdo = getattr(model_prefetch, "comfy_aimdo", None)
        return aimdo is not None and hasattr(aimdo, "malloc_graph")
    except Exception:
        return False


@contextmanager
def disabled_for_vdn():
    """Temporarily disable Comfy's compiler for one VDN model forward.

    The CLI flag is process-global, so true concurrent non-VDN execution cannot be
    isolated by any consumer-side workaround.  Comfy's normal prompt executor is
    serialized; reference counting prevents nested/overlapping VDN wrappers from
    restoring the flag while another VDN forward still owns it.
    """
    global _active_owned_guards, _warned

    args = comfy.cli_args.args
    owns = False
    affected = _compiler_stack_present()
    if affected:
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
                        "with VDN-H3; disabling it only for VDN model forwards")

    try:
        yield owns
    finally:
        if owns:
            with _lock:
                _active_owned_guards -= 1
                if _active_owned_guards == 0:
                    args.disable_comfy_compiler = False


def make_apply_model_wrapper():
    """Return the APPLY_MODEL wrapper that owns the switch for one evaluation."""

    def guarded(executor, *args, **kwargs):
        with disabled_for_vdn():
            return executor(*args, **kwargs)

    return guarded
