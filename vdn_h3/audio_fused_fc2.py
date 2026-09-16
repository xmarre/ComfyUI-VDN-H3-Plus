"""Packed-scope runtime LoRA for MiniMax-H3 fused INT8 MLP fc2 targets."""
from __future__ import annotations

import contextvars
import threading
import weakref

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.patcher_extension
import comfy.utils

from vdn_h3.audio_scope import scale_adapter_delta

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_ACTIVE = weakref.WeakKeyDictionary()
_LOCK = threading.RLock()


def _device_dtype(module, fallback):
    device = torch.device(comfy.model_management.get_torch_device())
    dtype = getattr(getattr(module, "weight", None), "dtype", None)
    if dtype not in _FLOAT_DTYPES:
        dtype = fallback if fallback in _FLOAT_DTYPES else torch.bfloat16
    return device, dtype


class _FusedFc2Plan:
    def __init__(self, path, terms, audio_strength, conditioning_strength=1.0):
        self.path = path
        self.terms = tuple((a.detach(), b.detach(), float(s)) for a, b, s in terms)
        self.audio_strength = float(audio_strength)
        self.conditioning_strength = float(conditioning_strength)
        self._cache = {}
        self._fc1 = contextvars.ContextVar(f"vdn_fc1_{path}_{id(self)}", default=None)

    def _fallback_dtype(self):
        for a, b, _ in self.terms:
            if a.dtype in _FLOAT_DTYPES:
                return a.dtype
            if b.dtype in _FLOAT_DTYPES:
                return b.dtype
        return torch.bfloat16

    def _compile(self, device, dtype):
        device = torch.device(device)
        key = (device.type, device.index, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        downs, ups = [], []
        for down, up, scale in self.terms:
            d = down.to(device=device, dtype=dtype, non_blocking=False)
            u = up.to(device=device, dtype=dtype, non_blocking=False)
            if scale != 1.0:
                u = u * scale
            downs.append(d)
            ups.append(u)
        down = downs[0] if len(downs) == 1 else torch.cat(downs, dim=0)
        up = ups[0] if len(ups) == 1 else torch.cat(ups, dim=1)
        self._cache[key] = (down, up)
        return down, up

    def prepare(self, fc2):
        device, dtype = _device_dtype(fc2, self._fallback_dtype())
        self._compile(device, dtype)

    def capture_fc1(self, module, inputs, output):
        del module, inputs
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"VDN fused fc2 capture expected Tensor at {self.path}")
        self._fc1.set(output)
        return output

    def apply_mlp(self, module, inputs, output):
        del module, inputs
        fc1 = self._fc1.get()
        self._fc1.set(None)
        if fc1 is None:
            raise RuntimeError(f"VDN fused fc2 did not observe fc1 before {self.path}")
        gate, up_act = fc1.chunk(2, dim=-1)
        act = F.silu(gate).mul(up_act)
        down, up = self._compile(act.device, act.dtype)
        delta = F.linear(F.linear(act, down), up)
        delta = scale_adapter_delta(
            delta,
            act,
            self.path,
            self.audio_strength,
            self.conditioning_strength,
        )
        return output + delta

    def clear(self):
        self._fc1.set(None)
        self._cache.clear()


def _remove(registration):
    for handle in reversed(registration["handles"]):
        handle.remove()
    for plan in registration["plans"]:
        plan.clear()


def install_fused_fc2_injection(
    new_model,
    dm,
    terms_by_path,
    audio_strength,
    conditioning_strength=1.0,
):
    """Install exact post-MLP fc2 LoRA residuals without patching INT8 fc2 weights."""
    if not terms_by_path:
        return {"modules": 0, "hooks": 0, "terms": 0, "bytes": 0}

    plans = []
    source_bytes = 0
    term_count = 0
    for path, terms in sorted(terms_by_path.items()):
        parent_path = path.rsplit(".fc2", 1)[0]
        parent = comfy.utils.get_attr(dm, parent_path)
        fc1 = comfy.utils.get_attr(dm, parent_path + ".fc1")
        fc2 = comfy.utils.get_attr(dm, path)
        plan = _FusedFc2Plan(path, terms, audio_strength, conditioning_strength)
        plans.append((parent, fc1, fc2, plan))
        for down, up, _ in terms:
            source_bytes += down.numel() * down.element_size()
            source_bytes += up.numel() * up.element_size()
            term_count += 1

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
            try:
                for _parent, _fc1, fc2, plan in plans:
                    plan.prepare(fc2)
                for parent, fc1, _fc2, plan in plans:
                    handles.append(fc1.register_forward_hook(plan.capture_fc1))
                    handles.append(parent.register_forward_hook(plan.apply_mlp))
            except Exception:
                for handle in reversed(handles):
                    handle.remove()
                for *_rest, plan in plans:
                    plan.clear()
                raise
            _ACTIVE[owner] = {
                "token": token,
                "handles": handles,
                "plans": [plan for *_rest, plan in plans],
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
        "vdn_lora_audio_fc2",
        [comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)],
    )
    return {
        "modules": len(plans),
        "hooks": len(plans) * 2,
        "terms": term_count,
        "bytes": int(source_bytes),
    }