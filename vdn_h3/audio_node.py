"""Workflow-compatible Advanced VDN node with packed audio/conditioning isolation."""
from __future__ import annotations

import copy
import logging
import os

from comfy.patcher_extension import WrappersMP

import vdn_h3.nodes as base_nodes
from vdn_h3.adapters import convert_adapter
from vdn_h3.audio_fix_runtime import install_audio_fix
from vdn_h3.audio_runtime import apply_adapters_audio_safe
from vdn_h3.audio_scope import enter_scope, exit_scope, scope_from_model_call
from vdn_h3.hybrid import VDNState, apply_vdn
from vdn_h3.managed import make_managed_branch_patcher
from vdn_h3.retained import RuntimeLinearBranch
import vdn_h3.policy as policy
import vdn_h3.spec as spec

_log = logging.getLogger("comfy.vdn")


def _make_audio_scope_wrapper(dm):
    def wrapper(executor, *args, **kwargs):
        scope = scope_from_model_call(dm, args, kwargs)
        token = enter_scope(scope)
        try:
            return executor(*args, **kwargs)
        finally:
            exit_scope(token)
    return wrapper


def _adapter_payload(adapter_cfg):
    return adapter_cfg.get("config", adapter_cfg) if isinstance(adapter_cfg, dict) else {}


def _has_audio_fix(vdn_checkpoint):
    path = spec.resolve_vdn_checkpoint(vdn_checkpoint)
    directory = os.path.join(path, "adapters", "audio_fix")
    return (
        os.path.isfile(os.path.join(directory, "adapter_config.json"))
        and os.path.isfile(os.path.join(directory, "adapter_model.safetensors"))
    )


def _validate_audio_fix_config(adapter_cfg):
    cfg = _adapter_payload(adapter_cfg)
    if cfg.get("scope") != "generated_audio":
        raise ValueError(
            "audio_fix adapter must declare scope='generated_audio'; refusing a globally "
            "applicable correction adapter")
    if cfg.get("target_policy") != "portable_sequence_linear":
        raise ValueError(
            "audio_fix adapter must declare target_policy='portable_sequence_linear'")
    if not cfg.get("exact_targets"):
        raise ValueError("audio_fix adapter must carry exact_targets=true")
    return cfg


def _apply_vdn_audio_safe(
    model,
    vdn_checkpoint,
    strength,
    branch_weights,
    attention_backend,
    verbose,
    *,
    audio_adapter_strength,
    conditioning_adapter_strength,
    audio_video_context_strength,
    conditioning_video_context_strength,
    audio_fix_strength,
    apply_turbo_adapter=True,
    cfg_overrides=None,
    fast_kernels=False,
    retain_buffers="auto",
    global_gate_mode="checkpoint",
    adapter_ablation="none",
):
    if branch_weights == "cache_gpu":
        _log.warning("[vdn] branch_weights=cache_gpu is deprecated; using resident")
        branch_weights = "resident"
    if branch_weights not in ("auto", "stream", "resident"):
        raise ValueError(
            f"branch_weights must be auto, stream or resident, got {branch_weights!r}")
    if retain_buffers not in ("auto", "on", "off"):
        raise ValueError(f"retain_buffers must be auto, on or off, got {retain_buffers!r}")
    if global_gate_mode not in ("checkpoint", "video_only"):
        raise ValueError(f"invalid global_gate_mode {global_gate_mode!r}")
    if adapter_ablation not in base_nodes._ADAPTER_ABLATION_TARGETS:
        raise ValueError(f"invalid adapter_ablation {adapter_ablation!r}")
    audio_adapter_strength = float(audio_adapter_strength)
    conditioning_adapter_strength = float(conditioning_adapter_strength)
    audio_video_context_strength = float(audio_video_context_strength)
    conditioning_video_context_strength = float(conditioning_video_context_strength)
    audio_fix_strength = float(audio_fix_strength)
    if not 0.0 <= audio_adapter_strength <= 1.0:
        raise ValueError("audio_adapter_strength must be in [0, 1]")
    if not 0.0 <= conditioning_adapter_strength <= 1.0:
        raise ValueError("conditioning_adapter_strength must be in [0, 1]")
    if not 0.0 <= audio_video_context_strength <= 1.0:
        raise ValueError("audio_video_context_strength must be in [0, 1]")
    if not 0.0 <= conditioning_video_context_strength <= 1.0:
        raise ValueError("conditioning_video_context_strength must be in [0, 1]")
    if not 0.0 <= audio_fix_strength <= 2.0:
        raise ValueError("audio_fix_strength must be in [0, 2]")

    path = spec.resolve_vdn_checkpoint(vdn_checkpoint)
    free = (
        base_nodes._effective_free_vram(model)
        if branch_weights == "auto" or retain_buffers == "auto"
        else None
    )
    prefer_int8 = False
    if branch_weights == "auto":
        branch_weights, prefer_int8 = policy.auto_branch_policy(path, free)

    cfg, branch_weights_by_block, adapters, branch_path = policy.load_vdn_checkpoint(
        path, prefer_int8=prefer_int8)
    cfg = dict(cfg)
    cfg.setdefault("linear_enabled", True)
    cfg["global_gate_mode"] = global_gate_mode
    cfg["audio_video_context_strength"] = audio_video_context_strength
    cfg["conditioning_video_context_strength"] = conditioning_video_context_strength

    if retain_buffers == "auto":
        retain = policy.auto_retain_policy(path, prefer_int8, free)
    else:
        retain = retain_buffers == "on"

    if cfg_overrides:
        changed = {
            key: (cfg.get(key), value)
            for key, value in cfg_overrides.items()
            if cfg.get(key) != value
        }
        if changed:
            _log.warning(
                "[vdn] architecture override active; execution deviates from checkpoint "
                "ModelSpec: %s", changed)
        cfg.update(cfg_overrides)

    dm = model.get_model_object("diffusion_model")
    blocks = getattr(dm, "blocks", None)
    if blocks is None or not blocks or not hasattr(getattr(blocks[0], "attn", None), "qkv_proj"):
        raise RuntimeError(
            "ApplyVDNH3 needs a current ComfyUI MiniMax-H3 MODEL "
            "(diffusion_model.blocks[].attn.qkv_proj).")
    if len(blocks) != len(branch_weights_by_block):
        raise RuntimeError(
            f"VDN checkpoint has {len(branch_weights_by_block)} blocks but the loaded "
            f"MiniMax-H3 base has {len(blocks)}")

    for key, patched in model.object_patches.items():
        if key.endswith(".attn.forward") and getattr(patched, "_vdn_forward", False):
            raise RuntimeError(
                "This MODEL already has VDN-H3 applied. Apply the node exactly once; "
                "changing options should re-execute from the upstream base MODEL.")

    attn0 = blocks[0].attn
    heads, head_dim = attn0.heads, attn0.head_dim
    hidden = dm.hidden_size
    base_nodes._validate_branch_shapes(
        path, branch_weights_by_block, cfg, hidden, heads, head_dim)

    branches = [
        RuntimeLinearBranch(
            weights,
            heads,
            head_dim,
            delta_rule=cfg["delta_rule"],
            bridge=cfg["bridge"],
            a_fp32=cfg["a_fp32"],
            short_conv=cfg["short_conv"],
            enable_text_state=cfg["enable_text_state"],
        )
        for weights in branch_weights_by_block
    ]
    for branch in branches:
        branch.fuse_epilogue = fast_kernels

    managed_weights = None
    managed_patcher = None
    if branch_weights == "resident":
        managed_weights, managed_patcher = make_managed_branch_patcher(
            branch_weights_by_block, model)

    state = VDNState(
        vdn_checkpoint,
        cfg,
        branches,
        heads,
        head_dim,
        managed_weights=managed_weights,
        retain_buffers=retain,
    )
    state.softmax_backend = attention_backend

    new_model = model.clone()
    if managed_patcher is not None:
        new_model.set_additional_models("vdn_branch", [managed_patcher])
    apply_vdn(new_model, state)

    wanted = {"default"}
    if apply_turbo_adapter:
        wanted.add("turbo")
    if "default" not in adapters:
        raise RuntimeError(f"{vdn_checkpoint}: required Stage-B adapter 'default' is missing")
    if apply_turbo_adapter and "turbo" not in adapters:
        raise RuntimeError(
            f"{vdn_checkpoint}: apply_turbo_adapter is enabled but this stage has no 'turbo' adapter")

    converted = {}
    for name in sorted(wanted):
        state_dict, adapter_cfg = adapters[name]
        converted[name] = convert_adapter(state_dict, adapter_cfg)
        if verbose:
            _log.info("[vdn] adapter %s converted: %d modules", name, len(converted[name]))

    converted, removed = base_nodes._apply_adapter_ablation(converted, adapter_ablation)
    if removed:
        _log.warning(
            "[vdn] diagnostic adapter ablation %s removed targets: %s",
            adapter_ablation,
            ", ".join(f"{name}={count}" for name, count in sorted(removed.items())),
        )

    # Register audio_fix before the released adapter injection. This makes its normal
    # fc1 hook live before the released fused-fc2 capture hook, so that hook observes
    # the final fc1 output exactly. The portable audio_fix policy has no fc2 target.
    audio_fix_report = {"enabled": False, "targets": 0, "strength": audio_fix_strength}
    if "audio_fix" in adapters and audio_fix_strength != 0.0:
        if not apply_turbo_adapter:
            raise ValueError(
                "audio_fix was trained on the finished Turbo stack; enable Turbo or set "
                "audio_fix_strength=0")
        state_dict, adapter_cfg = adapters["audio_fix"]
        _validate_audio_fix_config(adapter_cfg)
        audio_fix_converted = convert_adapter(state_dict, adapter_cfg)
        audio_fix_report = install_audio_fix(
            new_model, audio_fix_converted, audio_fix_strength)
        if verbose:
            _log.info(
                "[vdn] audio_fix converted: %d modules, strength %.3f",
                len(audio_fix_converted), audio_fix_strength)

    report = apply_adapters_audio_safe(
        new_model,
        converted,
        strength,
        audio_strength=audio_adapter_strength,
        conditioning_strength=conditioning_adapter_strength,
        stage_path=path,
        verbose=verbose,
    )
    report["audio_fix"] = audio_fix_report
    new_model.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL,
        "vdn_h3_audio_adapter_scope",
        _make_audio_scope_wrapper(dm),
    )
    _log.info(
        "[vdn] %s applied: blocks=%d radius=%d chunk=%d anchors=%s rule=%s "
        "branch=%s/%s buffers=%s backend=%s lora_mode=bypass global_gate=%s "
        "adapter_ablation=%s audio_adapter_strength=%.3f "
        "conditioning_adapter_strength=%.3f audio_video_context_strength=%.3f "
        "conditioning_video_context_strength=%.3f audio_fix_strength=%.3f adapters=%s",
        vdn_checkpoint,
        len(branches),
        cfg["radius"],
        cfg["chunk"],
        cfg["anchor_frames"],
        cfg["delta_rule"],
        os.path.basename(branch_path),
        branch_weights,
        "retained" if retain else "transient",
        attention_backend,
        global_gate_mode,
        adapter_ablation,
        audio_adapter_strength,
        conditioning_adapter_strength,
        audio_video_context_strength,
        conditioning_video_context_strength,
        audio_fix_strength,
        report,
    )
    return (new_model,)


class ApplyVDNH3AdvancedAudioSafe(base_nodes.ApplyVDNH3Advanced):
    @classmethod
    def INPUT_TYPES(cls):
        schema = copy.deepcopy(super().INPUT_TYPES())
        schema["optional"]["audio_adapter_strength"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
            "tooltip": (
                "Diagnostic for released Stage-B/Turbo residuals on generated target "
                "audio. The trained audio_fix path should normally be evaluated with "
                "this restored to 1.0."
            ),
        })
        schema["optional"]["conditioning_adapter_strength"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
            "tooltip": (
                "Diagnostic packed-conditioning scale. Production tests did not remove "
                "the yapping with this control; keep at 1.0 for trained audio_fix evaluation."
            ),
        })
        schema["optional"]["audio_video_context_strength"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
            "tooltip": (
                "Diagnostic audio->video context control. 0.0 damaged legitimate audio "
                "without removing chatter; keep at 1.0 for production/audio_fix use."
            ),
        })
        schema["optional"]["conditioning_video_context_strength"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
            "tooltip": (
                "Diagnostic video->conditioning control. 0.0 radically changed shot, "
                "environment and action while yapping persisted; keep at 1.0."
            ),
        })
        schema["optional"]["audio_fix_strength"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
            "tooltip": (
                "Strength of a trained checkpoint's generated-audio-only `audio_fix` "
                "adapter. 1.0 is the trained value. It never applies directly to video, "
                "text, references, token-refiner, AdaLN, final-layer or fused fc2 targets."
            ),
        })
        return schema

    def apply(self, model, vdn_checkpoint, apply_turbo_adapter, stage_b_strength,
              turbo_strength, lora_mode, branch_weights, retain_buffers,
              attention_backend, verbose, architecture_mode="checkpoint",
              window_radius=1, window_chunk=5, anchor_frames="both", text_state=True,
              linear_branch=True, fast_kernels=False, global_gate_mode="checkpoint",
              adapter_ablation="none", audio_adapter_strength=1.0,
              conditioning_adapter_strength=1.0, audio_video_context_strength=1.0,
              conditioning_video_context_strength=1.0, audio_fix_strength=1.0):
        audio_adapter_strength = float(audio_adapter_strength)
        conditioning_adapter_strength = float(conditioning_adapter_strength)
        audio_video_context_strength = float(audio_video_context_strength)
        conditioning_video_context_strength = float(conditioning_video_context_strength)
        audio_fix_strength = float(audio_fix_strength)

        # Validate scalar controls before any checkpoint/filesystem lookup. Aside from
        # being clearer to callers, this preserves the node's fail-fast contract for an
        # explicitly requested bypass-only diagnostic even if the checkpoint name is
        # stale or synthetic (as in workflow-migration/unit tests).
        if not 0.0 <= audio_adapter_strength <= 1.0:
            raise ValueError("audio_adapter_strength must be in [0, 1]")
        if not 0.0 <= conditioning_adapter_strength <= 1.0:
            raise ValueError("conditioning_adapter_strength must be in [0, 1]")
        if not 0.0 <= audio_video_context_strength <= 1.0:
            raise ValueError("audio_video_context_strength must be in [0, 1]")
        if not 0.0 <= conditioning_video_context_strength <= 1.0:
            raise ValueError("conditioning_video_context_strength must be in [0, 1]")
        if not 0.0 <= audio_fix_strength <= 2.0:
            raise ValueError("audio_fix_strength must be in [0, 2]")
        if architecture_mode not in ("checkpoint", "override"):
            raise ValueError(f"invalid architecture_mode {architecture_mode!r}")

        explicit_scoped_controls = (
            audio_adapter_strength != 1.0
            or conditioning_adapter_strength != 1.0
            or audio_video_context_strength != 1.0
            or conditioning_video_context_strength != 1.0
        )
        if explicit_scoped_controls and lora_mode != "bypass":
            raise ValueError(
                "packed audio/conditioning controls require lora_mode='bypass'")

        # Only probe the selected checkpoint after explicit controls have been
        # validated. A native-default workflow still needs this lookup so a checkpoint
        # carrying a trained audio_fix automatically selects the scoped runtime.
        has_audio_fix = _has_audio_fix(vdn_checkpoint)
        audio_fix_active = has_audio_fix and audio_fix_strength != 0.0
        if audio_fix_active and lora_mode != "bypass":
            raise ValueError(
                "generated-audio audio_fix requires lora_mode='bypass'")

        needs_scoped_runtime = explicit_scoped_controls or audio_fix_active
        if not needs_scoped_runtime:
            return super().apply(
                model, vdn_checkpoint, apply_turbo_adapter, stage_b_strength,
                turbo_strength, lora_mode, branch_weights, retain_buffers,
                attention_backend, verbose, architecture_mode=architecture_mode,
                window_radius=window_radius, window_chunk=window_chunk,
                anchor_frames=anchor_frames, text_state=text_state,
                linear_branch=linear_branch, fast_kernels=fast_kernels,
                global_gate_mode=global_gate_mode,
                adapter_ablation=adapter_ablation)

        overrides = None
        if architecture_mode == "override":
            overrides = {
                "radius": window_radius,
                "chunk": window_chunk,
                "anchor_frames": anchor_frames,
                "enable_text_state": text_state,
                "linear_enabled": linear_branch,
            }
        strength = {"default": stage_b_strength, "turbo": turbo_strength}
        return _apply_vdn_audio_safe(
            model,
            vdn_checkpoint,
            strength,
            branch_weights,
            attention_backend,
            verbose,
            audio_adapter_strength=audio_adapter_strength,
            conditioning_adapter_strength=conditioning_adapter_strength,
            audio_video_context_strength=audio_video_context_strength,
            conditioning_video_context_strength=conditioning_video_context_strength,
            audio_fix_strength=audio_fix_strength,
            apply_turbo_adapter=apply_turbo_adapter,
            cfg_overrides=overrides,
            fast_kernels=fast_kernels,
            retain_buffers=retain_buffers,
            global_gate_mode=global_gate_mode,
            adapter_ablation=adapter_ablation,
        )
