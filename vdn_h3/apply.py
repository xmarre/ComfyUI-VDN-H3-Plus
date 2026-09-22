"""Apply released VDN adapters through ComfyUI-owned patch mechanisms.

``merge`` uses normal ``ModelPatcher.add_patches`` weight ownership.

``bypass`` is the low-residency execution mode. Ordinary VDN LoRA terms are
implemented with PyTorch forward *post-hooks*, not Comfy ``BypassForwardHook``
and not ``ModelPatcher.weight_function``. VDN therefore never replaces or
splices ``module.forward`` and cannot participate in another provider's mutable
forward-wrapper linked list.

For pruned/curve MiniMax-H3, bypass mode also keeps the projected AdaLN updates
out of the base parameter tree. The exact affine projection (basis + mean) is
computed once on CPU, then the resulting low-rank curve-coordinate residual and
constant bias are added by the same post-forward mechanism. This is deliberate:
materializing the projected curve terms as native weight/bias patches changes the
execution/storage path of the pruned base before the first attention projection.
The production RTX PRO 6000 trace that invalidated the first v1.5.2 candidate
reported the CUDA fault at the first ordinary VDN post-hook, but that hook runs
after the module forward; the preceding VDN-specific operation in the block was
the materialized curve-AdaLN update. Bypass now leaves those base parameters
untouched as well.

All runtime adapter factors are staged onto the compute device when the
``PatcherInjection`` is injected. The first H3 forward therefore performs no VDN
adapter H2D transfer and cannot use a transfer merely as the asynchronous CUDA
error-reporting boundary. Unexpected device/dtype changes still have a synchronous
fallback conversion for correctness.

Fused INT8 ``mlp.fc2`` targets whose H3 fast path bypasses ``module.forward``
remain ordinary Comfy weight patches. In ``merge`` mode, projected curve-AdaLN
weight/bias terms also remain ordinary Comfy patches.
"""
from __future__ import annotations

import logging
import threading
import weakref
from collections import defaultdict

import torch
import torch.nn.functional as F

import comfy.lora
import comfy.model_management
import comfy.patcher_extension
import comfy.utils

from vdn_h3.curve_affine import find_curve_affine, project_curve_terms

_log = logging.getLogger("comfy.vdn")
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_RUNTIME_ADAPTER_MAX_DELTA_BYTES = 512 * 1024 * 1024


def _is_adaln(module: str) -> bool:
    return module.endswith(".adaln_proj.linear")


def _is_pruned_base(dm) -> bool:
    if getattr(dm, "use_adaln_curves", False):
        return True
    try:
        w = comfy.utils.get_attr(dm, "blocks.0.adaln_proj.linear.weight")
        return w.dim() == 2 and w.shape[-1] < 64
    except Exception:
        return False


def _comfy_lora(converted):
    lora = {}
    for path, (a, b, scale) in converted.items():
        rank = a.shape[0]
        lora[path + ".lora_A.weight"] = a.contiguous()
        lora[path + ".lora_B.weight"] = b.contiguous()
        lora[path + ".alpha"] = torch.tensor(scale * rank)
    return lora


def _load_comfy_adapter(new_model, converted):
    if not converted:
        return {}, {}
    modules = sorted(converted)
    lora = _comfy_lora(converted)
    key_map = {module: f"diffusion_model.{module}.weight" for module in modules}
    state_keys = set(new_model.model.state_dict().keys())
    missing_base = sorted(key_map[m] for m in modules if key_map[m] not in state_keys)
    if missing_base:
        raise RuntimeError(
            "VDN adapter targets are absent from the loaded MiniMax-H3 base: "
            + ", ".join(missing_base[:8])
            + (" ..." if len(missing_base) > 8 else "")
        )

    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
    expected = {key_map[m] for m in modules}
    missing_loaded = sorted(expected - set(loaded))
    if missing_loaded:
        raise RuntimeError(
            "ComfyUI could not materialize VDN LoRA targets: "
            + ", ".join(missing_loaded[:8])
            + (" ..." if len(missing_loaded) > 8 else "")
        )
    return loaded, key_map


def _native_patch_adapter(new_model, converted, strength: float) -> int:
    """Register one adapter entirely through ``ModelPatcher.add_patches``."""
    if not converted:
        return 0
    loaded, key_map = _load_comfy_adapter(new_model, converted)
    expected = set(key_map.values())
    accepted = set(
        new_model.add_patches({key: loaded[key] for key in expected}, strength)
    )
    missing_patch = sorted(expected - accepted)
    if missing_patch:
        raise RuntimeError(
            "ModelPatcher rejected VDN adapter targets: "
            + ", ".join(missing_patch[:8])
            + (" ..." if len(missing_patch) > 8 else "")
        )
    return len(accepted)


def _native_patch_projected_curve(new_model, terms_by_module, bias_terms_by_module):
    """Apply projected curve LoRAs + constants through normal patches (merge only)."""
    weight_patches = 0
    bias_patches = 0
    state_keys = set(new_model.model.state_dict().keys())

    for module in sorted(terms_by_module):
        for term in terms_by_module[module]:
            weight_patches += _native_patch_adapter(new_model, {module: term}, 1.0)

    for module in sorted(bias_terms_by_module):
        key = f"diffusion_model.{module}.bias"
        if key not in state_keys:
            raise RuntimeError(
                f"VDN projected AdaLN bias target is absent from the base: {key}"
            )
        base_bias = comfy.utils.get_attr(new_model.model, key)
        for offset, scale in bias_terms_by_module[module]:
            if tuple(offset.shape) != tuple(base_bias.shape):
                raise RuntimeError(
                    f"VDN projected AdaLN bias {key} has {tuple(offset.shape)}, but "
                    f"the base bias is {tuple(base_bias.shape)}"
                )
            if scale == 0.0:
                continue
            accepted = set(new_model.add_patches({key: (offset,)}, float(scale)))
            if key not in accepted:
                raise RuntimeError(
                    f"ModelPatcher rejected VDN projected AdaLN bias: {key}"
                )
            bias_patches += 1
    return weight_patches, bias_patches


def _int8_fused_fc2(dm, modules):
    """Return fc2 targets whose fused quantized forward bypasses module.forward.

    A PyTorch forward hook cannot observe these H3 fused calls. Keep these terms
    under normal Comfy weight-patch ownership, matching the previously validated
    quantized path.
    """
    fused = []
    for module in modules:
        if not module.endswith(".mlp.fc2"):
            continue
        try:
            weight = comfy.utils.get_attr(dm, module + ".weight")
        except Exception:
            continue
        if (
            getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout"
            and not getattr(getattr(weight, "_params", None), "transposed", False)
        ):
            fused.append(module)
    return fused


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _runtime_source_bytes(terms_by_module, bias_terms_by_module) -> int:
    total = 0
    for terms in terms_by_module.values():
        for down, up, _scale in terms:
            total += _tensor_bytes(down) + _tensor_bytes(up)
    for terms in bias_terms_by_module.values():
        for offset, _scale in terms:
            total += _tensor_bytes(offset)
    return total


def _module_compute_device_dtype(module, fallback_dtype: torch.dtype):
    device = torch.device(comfy.model_management.get_torch_device())
    dtype = getattr(getattr(module, "weight", None), "dtype", None)
    if dtype not in _FLOAT_DTYPES:
        dtype = fallback_dtype if fallback_dtype in _FLOAT_DTYPES else torch.bfloat16
    return device, dtype


class _PostForwardLoRA:
    """Exact additive low-rank residual without replacing ``module.forward``.

    ``terms`` are ``(down, up, scale)`` factors. ``bias_terms`` are optional
    ``(offset, scale)`` constants used by the exact pruned-AdaLN affine projection.
    Source tensors remain CPU/checkpoint-owned. Device copies are created at
    injection time for the module's normal compute device/dtype and dropped on
    eject/replacement. A synchronous fallback handles an unexpected activation
    device/dtype without relying on an asynchronous H2D copy inside the first H3
    forward.
    """

    def __init__(self, terms, bias_terms=()):
        self.terms = tuple(
            (down.detach(), up.detach(), float(scale))
            for down, up, scale in terms
        )
        self.bias_terms = tuple(
            (offset.detach(), float(scale))
            for offset, scale in bias_terms
        )
        if not self.terms and not self.bias_terms:
            raise RuntimeError("VDN post-forward hook has no adapter terms")
        self._cache = {}

    def _fallback_dtype(self):
        for down, up, _scale in self.terms:
            if down.dtype in _FLOAT_DTYPES:
                return down.dtype
            if up.dtype in _FLOAT_DTYPES:
                return up.dtype
        for offset, _scale in self.bias_terms:
            if offset.dtype in _FLOAT_DTYPES:
                return offset.dtype
        return torch.bfloat16

    def _compile(self, device, dtype):
        device = torch.device(device)
        key = (device.type, device.index, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        downs = []
        ups = []
        for down, up, scale in self.terms:
            # Blocking copies are intentional. Injection is the ownership boundary;
            # the model forward should not become an H2D synchronization/error probe.
            down_active = down.to(device=device, dtype=dtype, non_blocking=False)
            up_active = up.to(device=device, dtype=dtype, non_blocking=False)
            if scale != 1.0:
                up_active = up_active * scale
            downs.append(down_active)
            ups.append(up_active)

        down_cat = None
        up_cat = None
        if downs:
            down_cat = downs[0] if len(downs) == 1 else torch.cat(downs, dim=0)
            up_cat = ups[0] if len(ups) == 1 else torch.cat(ups, dim=1)

        bias = None
        for offset, scale in self.bias_terms:
            active = offset.to(device=device, dtype=dtype, non_blocking=False)
            if scale != 1.0:
                active = active * scale
            bias = active if bias is None else bias + active

        cached = (down_cat, up_cat, bias)
        self._cache[key] = cached
        return cached

    def prepare(self, module):
        device, dtype = _module_compute_device_dtype(module, self._fallback_dtype())
        self._compile(device, dtype)

    def _weights_for(self, x: torch.Tensor):
        key = (x.device.type, x.device.index, x.dtype)
        cached = self._cache.get(key)
        if cached is None:
            # Unexpected mixed-dtype/device execution remains supported, but keep the
            # conversion synchronous so allocator errors are reported at this exact
            # ownership boundary rather than being deferred into later kernels.
            cached = self._compile(x.device, x.dtype)
        return cached

    def __call__(self, module, inputs, output):
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                f"VDN post-forward bypass expected Tensor output from "
                f"{type(module).__name__}, got {type(output).__name__}"
            )
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError(
                f"VDN post-forward bypass expected the first positional input to "
                f"{type(module).__name__} to be a Tensor"
            )
        x = inputs[0]
        down, up, bias = self._weights_for(x)
        inference_inplace = not torch.is_grad_enabled() and not output.requires_grad

        if inference_inplace and down is not None and x.ndim == 2 and output.ndim == 2:
            if x.shape[0] != output.shape[0]:
                raise RuntimeError(
                    "VDN post-forward bypass row count changed across a linear module"
                )
            # H3's long-sequence projections are row independent. Bound the full-width
            # LoRA residual by row chunks rather than materializing a second qkv/fc1
            # sized tensor for the entire packed sequence.
            bytes_per_row = max(
                1, int(output.shape[-1]) * int(output.element_size())
            )
            rows_per_chunk = max(
                1, _RUNTIME_ADAPTER_MAX_DELTA_BYTES // bytes_per_row
            )
            for start in range(0, int(output.shape[0]), rows_per_chunk):
                stop = min(int(output.shape[0]), start + rows_per_chunk)
                delta = F.linear(F.linear(x[start:stop], down), up)
                if bias is not None:
                    delta.add_(bias)
                output[start:stop].add_(delta)
                del delta
            return output

        delta = None
        if down is not None:
            delta = F.linear(F.linear(x, down), up)
        if bias is not None:
            if delta is None:
                if inference_inplace:
                    output.add_(bias)
                    return output
                return output + bias
            if inference_inplace:
                delta.add_(bias)
            else:
                delta = delta + bias
        if delta is None:
            return output
        if inference_inplace:
            output.add_(delta)
            return output
        return output + delta

    def clear(self):
        self._cache.clear()


# ModelPatcher clones share the same inner model. Track the currently active VDN
# registration outside the model object so a newer clone can replace an older
# registration without private attributes on the model and so an old clone's
# later eject cannot tear down the newer generation.
_ACTIVE_POST_FORWARD = weakref.WeakKeyDictionary()
_ACTIVE_POST_FORWARD_LOCK = threading.RLock()


def _remove_post_forward_registration(registration):
    for handle in reversed(registration["handles"]):
        handle.remove()
    for plan in registration["plans"]:
        plan.clear()


def _install_post_forward_injection(
    new_model,
    dm,
    terms_by_module,
    bias_terms_by_module=None,
):
    bias_terms_by_module = bias_terms_by_module or {}
    paths = sorted(set(terms_by_module) | set(bias_terms_by_module))
    if not paths:
        return 0

    plans = []
    for path in paths:
        module = comfy.utils.get_attr(dm, path)
        register = getattr(module, "register_forward_hook", None)
        if not callable(register):
            raise RuntimeError(
                f"VDN bypass target {path!r} ({type(module).__name__}) does not "
                "support PyTorch forward hooks"
            )
        plans.append((
            path,
            module,
            _PostForwardLoRA(
                terms_by_module.get(path, ()),
                bias_terms_by_module.get(path, ()),
            ),
        ))

    owner = new_model.model
    token = object()

    def inject_all(model_patcher):
        del model_patcher
        with _ACTIVE_POST_FORWARD_LOCK:
            current = _ACTIVE_POST_FORWARD.get(owner)
            if current is not None and current["token"] is token:
                return
            if current is not None:
                _remove_post_forward_registration(current)

            handles = []
            hook_plans = [plan for _, _, plan in plans]
            try:
                # Stage every adapter tensor before registering any hook. If a copy
                # fails, no partial runtime adapter topology becomes visible.
                for _, module, plan in plans:
                    plan.prepare(module)
                for _, module, plan in plans:
                    handles.append(module.register_forward_hook(plan))
            except Exception:
                for handle in reversed(handles):
                    handle.remove()
                for plan in hook_plans:
                    plan.clear()
                raise

            _ACTIVE_POST_FORWARD[owner] = {
                "token": token,
                "handles": handles,
                "plans": hook_plans,
            }

    def eject_all(model_patcher):
        del model_patcher
        with _ACTIVE_POST_FORWARD_LOCK:
            current = _ACTIVE_POST_FORWARD.get(owner)
            # A newer clone may already have replaced this registration. An old
            # eject must not remove the newer generation.
            if current is None or current["token"] is not token:
                return
            _remove_post_forward_registration(current)
            try:
                del _ACTIVE_POST_FORWARD[owner]
            except KeyError:
                pass

    injection = comfy.patcher_extension.PatcherInjection(
        inject=inject_all,
        eject=eject_all,
    )
    new_model.set_injections("vdn_lora", [injection])
    return len(plans)


def apply_adapters(
    new_model,
    converted_by_name,
    strength,
    mode="merge",
    stage_path=None,
    verbose=False,
):
    """Apply released VDN adapters through merge or non-mutating runtime bypass."""
    if mode not in ("merge", "bypass"):
        raise ValueError(
            f"VDN lora_mode must be 'merge' or 'bypass', got {mode!r}"
        )

    per_name = strength if isinstance(strength, dict) else None
    dm = new_model.get_model_object("diffusion_model")
    pruned = _is_pruned_base(dm)
    report = {}
    curve_terms = {}
    runtime_terms = defaultdict(list)
    runtime_bias_terms = defaultdict(list)

    for name, converted in converted_by_name.items():
        s = float(per_name.get(name, 1.0) if per_name is not None else strength)
        ordinary = {}
        curve_count = 0
        for path, (a, b, scale) in converted.items():
            effective_scale = float(scale) * s
            if pruned and _is_adaln(path):
                curve_terms.setdefault(path, []).append((a, b, effective_scale))
                curve_count += 1
            else:
                ordinary[path] = (a, b, scale)

        native_patches = 0
        bypass_targets = 0
        fused_native_targets = 0

        if mode == "merge":
            native_patches = (
                _native_patch_adapter(new_model, ordinary, s) if ordinary else 0
            )
        elif ordinary:
            modules = sorted(ordinary)
            fused = set(_int8_fused_fc2(dm, modules))
            bypass_modules = [module for module in modules if module not in fused]
            fused_terms = {module: ordinary[module] for module in sorted(fused)}

            for module in bypass_modules:
                down, up, term_scale = ordinary[module]
                runtime_terms[module].append(
                    (down, up, float(term_scale) * s)
                )
            bypass_targets = len(bypass_modules)
            if fused_terms:
                fused_native_targets = _native_patch_adapter(
                    new_model, fused_terms, s
                )
                native_patches += fused_native_targets

        report[name] = {
            "native_weight_patches": native_patches,
            "runtime_bypass_targets": bypass_targets,
            # Kept as a compatibility/logging alias. In v1.5.2 it counts
            # non-mutating runtime adapter terms, not ModelPatcher weight wrappers.
            "runtime_weight_targets": bypass_targets,
            "fused_native_targets": fused_native_targets,
            "curve_adaln": curve_count,
            "strength": s,
        }
        if verbose:
            _log.info(
                "[vdn] adapter %s: %d native patches, %d post-forward bypass targets, "
                "%d fused-native targets, %d curve AdaLN",
                name,
                native_patches,
                bypass_targets,
                fused_native_targets,
                curve_count,
            )

    projected_curve = {}
    curve_bias_terms = {}
    affine = None
    if curve_terms:
        if stage_path is None:
            raise RuntimeError("VDN curve AdaLN projection requires the stage path")
        table = getattr(dm, "adaln_t_table", None)
        if table is None:
            raise RuntimeError(
                "MiniMax-H3 was detected as a curve/pruned base but has no "
                "adaln_t_table"
            )
        if (
            getattr(table, "device", None) is not None
            and table.device.type == "meta"
        ):
            raise RuntimeError(
                "MiniMax-H3 adaln_t_table is still on the meta device; load the "
                "base checkpoint before applying VDN"
            )
        affine = find_curve_affine(stage_path, table, base_patcher=new_model)
        projected_curve, curve_bias_terms = project_curve_terms(
            curve_terms, affine
        )
        _log.info(
            "[vdn] curve AdaLN affine: %s (%d weight targets + %d float32 bias targets)",
            affine.source,
            len(projected_curve),
            len(curve_bias_terms),
        )

    curve_weight_patches = 0
    curve_bias_patches = 0
    curve_runtime_targets = 0
    if projected_curve or curve_bias_terms:
        if mode == "merge":
            curve_weight_patches, curve_bias_patches = _native_patch_projected_curve(
                new_model,
                projected_curve,
                curve_bias_terms,
            )
        else:
            for module, terms in projected_curve.items():
                runtime_terms[module].extend(terms)
            for module, terms in curve_bias_terms.items():
                runtime_bias_terms[module].extend(terms)
            curve_runtime_targets = len(
                set(projected_curve) | set(curve_bias_terms)
            )

    if mode == "bypass":
        forward_hook_modules = _install_post_forward_injection(
            new_model,
            dm,
            runtime_terms,
            runtime_bias_terms,
        )
        runtime_term_count = sum(len(terms) for terms in runtime_terms.values())
        runtime_bias_count = sum(len(terms) for terms in runtime_bias_terms.values())
        runtime_report = {
            "mode": "post_forward_hook_bypass",
            "forward_hooks": forward_hook_modules,
            "pytorch_forward_post_hooks": forward_hook_modules,
            "runtime_terms": runtime_term_count,
            "runtime_bias_terms": runtime_bias_count,
            "runtime_preloaded_on_inject": True,
            "output_residual_mode": "bounded_chunked_inplace_inference_out_of_place_grad",
            "max_runtime_delta_bytes": _RUNTIME_ADAPTER_MAX_DELTA_BYTES,
            "mutable_forward_wrappers": 0,
            "module_forward_untouched": True,
            "weight_wrappers": 0,
            "bias_wrappers": 0,
            "managed_adapter_bytes": _runtime_source_bytes(
                runtime_terms,
                runtime_bias_terms,
            ),
            "delta_buffer_limit_bytes": 0,
            "owner_key": None,
            "stack_safe_cross_provider": True,
            "cross_provider_forward_chain_independent": True,
            "projected_curve_runtime_targets": curve_runtime_targets,
            "projected_curve_weight_patches": 0,
            "projected_curve_bias_patches": 0,
        }
        report["runtime_bypass"] = runtime_report
        # Preserve the v1.5.0 report key for callers/log parsers while making the
        # changed ownership explicit in its values.
        report["runtime_lowvram"] = runtime_report

    if affine is not None:
        report["curve_adaln_projection"] = {
            "source": affine.source,
            "mode": (
                "merge" if mode == "merge"
                else "bypass_post_forward_projected_residual"
            ),
            "weight_patches": curve_weight_patches,
            "bias_patches": curve_bias_patches,
            "runtime_targets": curve_runtime_targets,
            "dense_width": int(affine.mean.shape[0]),
            "curve_width": int(affine.basis.shape[0]),
        }

    return report
