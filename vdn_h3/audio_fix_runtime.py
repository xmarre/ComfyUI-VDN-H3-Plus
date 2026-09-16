"""Runtime for trained ``audio_fix`` adapters.

An audio-fix checkpoint declares a LoRA adapter with
``scope=generated_audio`` and ``target_policy=portable_sequence_linear``. Unlike the
released Stage-B/Turbo adapters, this correction must never be merged globally: its
low-rank residual is evaluated only for target-audio packed rows. The portable target
policy intentionally excludes fused INT8 fc2 and all non-row-wise AdaLN/refiner/final
projections, so the Diffusers training graph and ComfyUI deployment graph are identical.
"""
from __future__ import annotations

import threading
import weakref

import torch
import torch.nn.functional as F

import comfy.patcher_extension
import comfy.utils

import vdn_h3.apply as base_apply
from vdn_h3.audio_scope import current_scope


_ACTIVE = weakref.WeakKeyDictionary()
_LOCK = threading.RLock()
_ALLOWED_PATH = (
    ".attn.qkv_proj",
    ".attn.out_proj",
    ".mlp.fc1",
)


def _validate_paths(converted):
    bad = [path for path in converted if not (
        path.startswith("blocks.") and path.endswith(_ALLOWED_PATH)
    )]
    if bad:
        raise ValueError(
            "audio_fix portable_sequence_linear contains unsupported Comfy targets: "
            + ", ".join(sorted(bad)[:8]))


class _AudioOnlyPostForwardLoRA(base_apply._PostForwardLoRA):
    def __init__(self, path, terms):
        super().__init__(terms)
        self.path = path

    def __call__(self, module, inputs, output):
        del module
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError(f"VDN audio_fix target {self.path} needs a Tensor input")
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"VDN audio_fix target {self.path} needs a Tensor output")
        scope = current_scope()
        if scope is None:
            raise RuntimeError(
                f"VDN audio_fix target {self.path!r} ran outside the packed model-call scope")
        x = inputs[0]
        if x.ndim not in (2, 3) or output.ndim != x.ndim:
            raise RuntimeError(
                f"VDN audio_fix target {self.path} expected rank-2/3 packed rows, got "
                f"input {tuple(x.shape)} output {tuple(output.shape)}")
        if not 0 <= scope.audio_start <= scope.audio_end:
            raise RuntimeError("VDN audio_fix received an invalid target-audio span")
        if scope.audio_start == scope.audio_end:
            return output
        row_dim = 1 if x.ndim == 3 else 0
        row_count = x.shape[row_dim]
        if scope.audio_end > row_count:
            raise RuntimeError(
                f"VDN audio_fix audio span [{scope.audio_start}, {scope.audio_end}) exceeds "
                f"current packed row count {row_count}")
        sl = [slice(None)] * x.ndim
        sl[row_dim] = slice(scope.audio_start, scope.audio_end)
        sl = tuple(sl)
        xa = x[sl]
        down, up, bias = self._weights_for(xa)
        if bias is not None:
            raise RuntimeError("audio_fix does not support bias residuals")
        delta = F.linear(F.linear(xa, down), up)
        # Comfy inference executes without autograd; mutating only the fresh module
        # output slice avoids cloning the entire packed qkv/fc1 tensor for a tiny audio
        # correction. No base/module forward is replaced.
        output[sl].add_(delta.to(dtype=output.dtype))
        return output


def _remove(registration):
    for handle in reversed(registration["handles"]):
        handle.remove()
    for plan in registration["plans"]:
        plan.clear()


def install_audio_fix(new_model, converted, strength=1.0):
    """Install one generated-audio-only adapter independently of VDN/Turbo hooks."""
    strength = float(strength)
    if strength < 0.0:
        raise ValueError("audio_fix_strength must be >= 0")
    if strength == 0.0:
        return {"enabled": False, "targets": 0, "terms": 0, "strength": 0.0}
    _validate_paths(converted)
    dm = new_model.get_model_object("diffusion_model")
    plans = []
    for path in sorted(converted):
        down, up, scale = converted[path]
        plan = _AudioOnlyPostForwardLoRA(
            path, [(down, up, float(scale) * strength)])
        plans.append((comfy.utils.get_attr(dm, path), plan))

    owner = new_model.model
    token = object()

    def inject(model_patcher):
        del model_patcher
        with _LOCK:
            current = _ACTIVE.get(owner)
            if current is not None and current["token"] is token:
                return
            if current is not None:
                _remove(current)
            handles = []
            hook_plans = [plan for _, plan in plans]
            try:
                for module, plan in plans:
                    plan.prepare(module)
                for module, plan in plans:
                    handles.append(module.register_forward_hook(plan))
            except Exception:
                for handle in reversed(handles):
                    handle.remove()
                for plan in hook_plans:
                    plan.clear()
                raise
            _ACTIVE[owner] = {
                "token": token,
                "handles": handles,
                "plans": hook_plans,
            }

    def eject(model_patcher):
        del model_patcher
        with _LOCK:
            current = _ACTIVE.get(owner)
            if current is None or current["token"] is not token:
                return
            _remove(current)
            try:
                del _ACTIVE[owner]
            except KeyError:
                pass

    new_model.set_injections(
        "vdn_audio_fix",
        [comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)],
    )
    return {
        "enabled": True,
        "targets": len(plans),
        "terms": len(plans),
        "strength": strength,
        "scope": "generated_audio",
        "target_policy": "portable_sequence_linear",
        "module_forward_untouched": True,
    }


__all__ = ["install_audio_fix"]
