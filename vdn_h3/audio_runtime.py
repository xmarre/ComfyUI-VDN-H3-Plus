"""Bypass adapter runtime with independent packed conditioning/audio scaling."""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

import comfy.patcher_extension
import comfy.utils

import vdn_h3.apply as base_apply
from vdn_h3.audio_fused_fc2 import install_fused_fc2_injection
from vdn_h3.audio_scope import scale_adapter_delta
from vdn_h3.curve_affine import find_curve_affine, project_curve_terms


class AudioScopedPostForwardLoRA(base_apply._PostForwardLoRA):
    def __init__(
        self,
        path,
        terms,
        bias_terms=(),
        audio_strength=1.0,
        conditioning_strength=1.0,
    ):
        super().__init__(terms, bias_terms)
        self.path = path
        self.audio_strength = float(audio_strength)
        self.conditioning_strength = float(conditioning_strength)

    def __call__(self, module, inputs, output):
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                f"VDN audio-scoped bypass expected Tensor output from {type(module).__name__}")
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError(
                f"VDN audio-scoped bypass expected Tensor first input to {type(module).__name__}")
        x = inputs[0]
        down, up, bias = self._weights_for(x)
        delta = None
        if down is not None:
            delta = F.linear(F.linear(x, down), up)
        if bias is not None:
            if delta is None:
                view = (1,) * (output.ndim - 1) + (bias.shape[0],)
                delta = bias.view(view).expand_as(output).clone()
            else:
                delta = delta + bias
        if delta is None:
            return output
        return output + scale_adapter_delta(
            delta,
            x,
            self.path,
            self.audio_strength,
            self.conditioning_strength,
        )


def _remove_registration(registration):
    for handle in reversed(registration["handles"]):
        handle.remove()
    for plan in registration["plans"]:
        plan.clear()


def _install_scoped_post_forward(
    new_model,
    dm,
    terms_by_module,
    bias_terms_by_module,
    audio_strength,
    conditioning_strength,
):
    paths = sorted(set(terms_by_module) | set(bias_terms_by_module))
    if not paths:
        return 0

    plans = []
    for path in paths:
        module = comfy.utils.get_attr(dm, path)
        plans.append((
            module,
            AudioScopedPostForwardLoRA(
                path,
                terms_by_module.get(path, ()),
                bias_terms_by_module.get(path, ()),
                audio_strength=audio_strength,
                conditioning_strength=conditioning_strength,
            ),
        ))

    owner = new_model.model
    token = object()

    def inject(model_patcher):
        del model_patcher
        with base_apply._ACTIVE_POST_FORWARD_LOCK:
            current = base_apply._ACTIVE_POST_FORWARD.get(owner)
            if current is not None and current["token"] is token:
                return
            if current is not None:
                _remove_registration(current)
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
            base_apply._ACTIVE_POST_FORWARD[owner] = {
                "token": token,
                "handles": handles,
                "plans": hook_plans,
            }

    def eject(model_patcher):
        del model_patcher
        with base_apply._ACTIVE_POST_FORWARD_LOCK:
            current = base_apply._ACTIVE_POST_FORWARD.get(owner)
            if current is None or current["token"] is not token:
                return
            _remove_registration(current)
            try:
                del base_apply._ACTIVE_POST_FORWARD[owner]
            except KeyError:
                pass

    new_model.set_injections(
        "vdn_lora",
        [comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)],
    )
    return len(plans)


def apply_adapters_audio_safe(
    new_model,
    converted_by_name,
    strength,
    *,
    audio_strength,
    conditioning_strength=1.0,
    stage_path=None,
    verbose=False,
):
    """Apply VDN adapters with separate target-audio and conditioning-row scaling."""
    audio_strength = float(audio_strength)
    conditioning_strength = float(conditioning_strength)
    if not 0.0 <= audio_strength <= 1.0:
        raise ValueError("audio_adapter_strength must be in [0, 1]")
    if not 0.0 <= conditioning_strength <= 1.0:
        raise ValueError("conditioning_adapter_strength must be in [0, 1]")

    per_name = strength if isinstance(strength, dict) else None
    dm = new_model.get_model_object("diffusion_model")
    pruned = base_apply._is_pruned_base(dm)
    report = {}
    curve_terms = {}
    runtime_terms = defaultdict(list)
    runtime_bias_terms = defaultdict(list)
    fused_fc2_terms = defaultdict(list)

    for name, converted in converted_by_name.items():
        s = float(per_name.get(name, 1.0) if per_name is not None else strength)
        ordinary = {}
        curve_count = 0
        for path, (a, b, scale) in converted.items():
            effective_scale = float(scale) * s
            if pruned and base_apply._is_adaln(path):
                curve_terms.setdefault(path, []).append((a, b, effective_scale))
                curve_count += 1
            else:
                ordinary[path] = (a, b, scale)

        modules = sorted(ordinary)
        fused = set(base_apply._int8_fused_fc2(dm, modules))
        bypass_modules = [module for module in modules if module not in fused]
        for module in bypass_modules:
            down, up, term_scale = ordinary[module]
            runtime_terms[module].append((down, up, float(term_scale) * s))
        for module in sorted(fused):
            down, up, term_scale = ordinary[module]
            fused_fc2_terms[module].append((down, up, float(term_scale) * s))

        report[name] = {
            "native_weight_patches": 0,
            "runtime_bypass_targets": len(bypass_modules),
            "runtime_weight_targets": len(bypass_modules),
            "fused_native_targets": 0,
            "fused_runtime_targets": len(fused),
            "curve_adaln": curve_count,
            "strength": s,
            "generated_audio_strength": audio_strength,
            "packed_conditioning_strength": conditioning_strength,
        }

    projected_curve = {}
    curve_bias_terms = {}
    affine = None
    if curve_terms:
        if stage_path is None:
            raise RuntimeError("VDN curve AdaLN projection requires the stage path")
        table = getattr(dm, "adaln_t_table", None)
        if table is None:
            raise RuntimeError(
                "MiniMax-H3 was detected as a curve/pruned base but has no adaln_t_table")
        if getattr(table, "device", None) is not None and table.device.type == "meta":
            raise RuntimeError("MiniMax-H3 adaln_t_table is still on the meta device")
        affine = find_curve_affine(stage_path, table, base_patcher=new_model)
        projected_curve, curve_bias_terms = project_curve_terms(curve_terms, affine)
        for module, terms in projected_curve.items():
            runtime_terms[module].extend(terms)
        for module, terms in curve_bias_terms.items():
            runtime_bias_terms[module].extend(terms)

    normal_hooks = _install_scoped_post_forward(
        new_model,
        dm,
        runtime_terms,
        runtime_bias_terms,
        audio_strength,
        conditioning_strength,
    )
    fused_report = install_fused_fc2_injection(
        new_model,
        dm,
        fused_fc2_terms,
        audio_strength,
        conditioning_strength,
    )

    runtime_terms_count = sum(len(terms) for terms in runtime_terms.values())
    runtime_terms_count += int(fused_report["terms"])
    runtime_bias_count = sum(len(terms) for terms in runtime_bias_terms.values())
    managed_bytes = base_apply._runtime_source_bytes(runtime_terms, runtime_bias_terms)
    managed_bytes += int(fused_report["bytes"])
    runtime_report = {
        "mode": "post_forward_hook_bypass_packed_scoped",
        "forward_hooks": normal_hooks + int(fused_report["hooks"]),
        "pytorch_forward_post_hooks": normal_hooks + int(fused_report["hooks"]),
        "runtime_terms": runtime_terms_count,
        "runtime_bias_terms": runtime_bias_count,
        "runtime_preloaded_on_inject": True,
        "mutable_forward_wrappers": 0,
        "module_forward_untouched": True,
        "weight_wrappers": 0,
        "bias_wrappers": 0,
        "managed_adapter_bytes": managed_bytes,
        "delta_buffer_limit_bytes": 0,
        "owner_key": None,
        "stack_safe_cross_provider": True,
        "cross_provider_forward_chain_independent": True,
        "generated_audio_strength": audio_strength,
        "packed_conditioning_strength": conditioning_strength,
        "fused_fc2_runtime_modules": int(fused_report["modules"]),
        "projected_curve_runtime_targets": len(set(projected_curve) | set(curve_bias_terms)),
        "projected_curve_weight_patches": 0,
        "projected_curve_bias_patches": 0,
    }
    report["runtime_bypass"] = runtime_report
    report["runtime_lowvram"] = runtime_report

    if affine is not None:
        report["curve_adaln_projection"] = {
            "source": affine.source,
            "mode": "bypass_post_forward_projected_residual_packed_scoped",
            "weight_patches": 0,
            "bias_patches": 0,
            "runtime_targets": len(set(projected_curve) | set(curve_bias_terms)),
            "dense_width": int(affine.mean.shape[0]),
            "curve_width": int(affine.basis.shape[0]),
        }
    return report