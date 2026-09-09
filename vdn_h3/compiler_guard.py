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

Hook placement is not a matter of taste.  ``comfy/ldm/minimax/model.py`` reads the
switch and opens the graph *before* it runs any DIFFUSION_MODEL wrapper::

    compile_allocations = comfy.model_prefetch.malloc_graph_enabled(x[0].device)
    if compile_allocations:
        out = [torch.empty_like(x[0]), torch.empty_like(x[1])]
        comfy.model_prefetch.malloc_graph_begin(self, x[0].device)
    graph_out = ...DIFFUSION_MODEL wrappers...

A guard installed at DIFFUSION_MODEL therefore flips the flag *after* the decision
is taken and the graph is already open.  It cannot prevent the compile; all it
achieves is silencing the per-block ``prefetch_queue_pop(..., malloc_scope="block")``
recording at ``model.py:747`` partway through a graph that stays open and is closed
normally, leaving a half-recorded pattern.  It also leaves the two full-size
``empty_like`` staging latents allocated for the whole forward.

The guard is therefore installed at OUTER_SAMPLE, which Comfy gathers in
``samplers.py`` before the sampling loop begins.  That is early enough for
``malloc_graph_enabled`` to return False at the top of every forward in the run,
so no graph is ever opened and no staging latents are allocated.  The scope is one
sampling run rather than one forward; that is the narrowest scope that is actually
correct, not a widening for convenience.

One asymmetry is inherent to mutating config instead of passing the CLI flag:
``cli_args`` applies ``disable_comfy_compiler -> disable_cuda_graphs`` once at
import time, so a runtime flip suppresses the malloc graph but not CUDA graphs.
``malloc_graph_enabled`` is the path that matters here.
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
    """Temporarily disable Comfy's compiler for one VDN sampling run.

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
                        "with VDN-H3; disabling it only for VDN sampling runs")

    try:
        yield owns
    finally:
        if owns:
            with _lock:
                _active_owned_guards -= 1
                if _active_owned_guards == 0:
                    args.disable_comfy_compiler = False


def make_outer_sample_wrapper():
    """Return VDN's OUTER_SAMPLE wrapper, which owns the compiler switch.

    ``vdn_h3.hybrid.apply_vdn`` registers this on the patched model only, so a
    sampling run against an unpatched model is untouched.  Nothing in Comfy is
    monkey-patched: this is an ordinary wrapper registration, and the switch is the
    same config value ``--disable-comfy-compiler`` sets.
    """
    def guarded(executor, *args, **kwargs):
        with disabled_for_vdn():
            return executor(*args, **kwargs)

    guarded._vdn_compiler_guard = True
    return guarded
