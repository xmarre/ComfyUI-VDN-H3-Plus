"""ComfyUI nodes for applying official VDN-H3 checkpoints to MiniMax-H3."""
from __future__ import annotations

import logging
import os

import comfy.model_management

from vdn_h3.adapters import convert_adapter
from vdn_h3.apply import apply_adapters
from vdn_h3.hybrid import VDNState, apply_vdn
from vdn_h3.managed import make_managed_branch_patcher
from vdn_h3.retained import RuntimeLinearBranch
import vdn_h3.policy as policy
import vdn_h3.spec as spec

_log = logging.getLogger("comfy.vdn")

_ADAPTER_ABLATION_TARGETS = {
    "none": None,
    "stage_b_dit_off": ("default", "dit"),
    "stage_b_refiner_off": ("default", "refiner"),
    "turbo_attention_off": ("turbo", "attention"),
    "turbo_mlp_off": ("turbo", "mlp"),
    "turbo_adaln_off": ("turbo", "adaln"),
    "turbo_refiner_off": ("turbo", "refiner"),
}


def _adapter_target_group(path):
    """Classify converted Comfy targets for targeted diagnostic ablations."""
    if path.startswith("token_refiner."):
        return "refiner"
    if path.endswith(".adaln_proj.linear"):
        return "adaln"
    if ".mlp." in path:
        return "mlp"
    if ".attn." in path:
        return "attention"
    if path.startswith("blocks."):
        return "dit"
    return "other"


def _apply_adapter_ablation(converted_by_name, ablation):
    """Remove one adapter target class without changing any surviving tensor/scale."""
    if ablation not in _ADAPTER_ABLATION_TARGETS:
        raise ValueError(f"invalid adapter_ablation {ablation!r}")
    target = _ADAPTER_ABLATION_TARGETS[ablation]
    if target is None:
        return converted_by_name, {}

    adapter_name, group = target
    if adapter_name not in converted_by_name:
        raise ValueError(
            f"adapter_ablation={ablation!r} requires adapter {adapter_name!r} to be enabled")

    filtered = {}
    removed = {}
    for name, modules in converted_by_name.items():
        if name != adapter_name:
            filtered[name] = modules
            continue
        keep = {}
        count = 0
        for path, term in modules.items():
            path_group = _adapter_target_group(path)
            matches = (path_group != "refiner") if group == "dit" else (path_group == group)
            if matches:
                count += 1
            else:
                keep[path] = term
        if not count:
            raise RuntimeError(
                f"adapter_ablation={ablation!r} matched no converted {adapter_name!r} targets")
        filtered[name] = keep
        removed[name] = count
    return filtered, removed


def _validate_branch_shapes(path, branches, cfg, hidden, heads, head_dim):
    """Validate every enabled trained tensor on every block against the loaded base."""
    linear_dim = cfg["linear_head_dim"]
    if linear_dim != head_dim:
        raise RuntimeError(
            f"{path}: checkpoint linear_head_dim={linear_dim}, but this Comfy port shares "
            f"the base Q/K/V whose head_dim={head_dim}. The official architecture has "
            "no projection between them; refusing an incompatible base/checkpoint pair.")

    expected = {
        "to_out_linear.weight": (hidden, heads * linear_dim),
        "beta_proj.weight": (heads, hidden),
        "norm.weight": (linear_dim,),
        "alpha.A_log": (heads,),
        "alpha.dt_bias": (heads * linear_dim,),
        "alpha.down.weight": (linear_dim, hidden),
        "alpha.up.weight": (heads * linear_dim, linear_dim),
        "output_gate.down.weight": (linear_dim, hidden),
        "output_gate.up.weight": (heads * linear_dim, linear_dim),
        "output_gate.up.bias": (heads * linear_dim,),
    }
    if cfg["enable_softmax_gate"]:
        expected.update({
            "softmax_gate.up.weight": (heads, hidden),
            "softmax_gate.up.bias": (heads,),
        })
    channels = heads * linear_dim
    for target in cfg["short_conv"]:
        expected[f"short_conv.{target}_sp.weight"] = (channels, 1, 5, 5)
        expected[f"short_conv.{target}_tm.weight"] = (channels, 1, 5)

    errors = []
    for index, weights in enumerate(branches):
        for key, shape in expected.items():
            tensor = weights.get(key)
            if tensor is None:
                errors.append(f"block {index}: missing {key}")
            elif tuple(tensor.shape) != shape:
                errors.append(
                    f"block {index}: {key} has {tuple(tensor.shape)}, expected {shape}")
    if errors:
        preview = "; ".join(errors[:12])
        if len(errors) > 12:
            preview += f"; ... and {len(errors) - 12} more"
        raise RuntimeError(f"VDN checkpoint/base shape mismatch in {path}: {preview}")


def _free_vram():
    device = comfy.model_management.get_torch_device()
    return comfy.model_management.get_free_memory(device)


def _effective_free_vram(model):
    """Conservative free VRAM after reserving still-unloaded base-model bytes.

    Apply nodes may execute before Comfy has made the supplied H3 MODEL resident. Raw
    allocator free memory would then make `auto` optimistically choose a resident VDN
    branch or retained scratch even though the base itself still has to load. Reserve
    the ModelPatcher's remaining model bytes before applying the placement heuristics.
    """
    raw = int(_free_vram())
    try:
        model_size = int(model.model_size())
        loaded_size = int(model.loaded_size())
        remaining = max(0, model_size - loaded_size)
    except Exception as exc:
        _log.warning(
            "[vdn] could not reserve unloaded MODEL bytes for auto VRAM policy (%s); "
            "using raw free-memory reading", exc)
        remaining = 0
    effective = max(0, raw - remaining)
    _log.info(
        "[vdn] auto VRAM budget: %.2f GiB raw - %.2f GiB unloaded base = %.2f GiB",
        raw / (1 << 30), remaining / (1 << 30), effective / (1 << 30))
    return effective


def _apply_vdn(model, vdn_checkpoint, strength, lora_mode, branch_weights,
               attention_backend, verbose, apply_turbo_adapter=True,
               cfg_overrides=None, fast_kernels=False, retain_buffers="auto",
               global_gate_mode="checkpoint", adapter_ablation="none"):
    if lora_mode not in ("merge", "bypass"):
        raise ValueError(f"lora_mode must be merge or bypass, got {lora_mode!r}")
    if global_gate_mode not in ("checkpoint", "video_only"):
        raise ValueError(f"invalid global_gate_mode {global_gate_mode!r}")
    if adapter_ablation not in _ADAPTER_ABLATION_TARGETS:
        raise ValueError(f"invalid adapter_ablation {adapter_ablation!r}")
    if branch_weights == "cache_gpu":
        _log.warning("[vdn] branch_weights=cache_gpu is deprecated; using resident")
        branch_weights = "resident"
    if branch_weights not in ("auto", "stream", "resident"):
        raise ValueError(
            f"branch_weights must be auto, stream or resident, got {branch_weights!r}")
    if retain_buffers not in ("auto", "on", "off"):
        raise ValueError(
            f"retain_buffers must be auto, on or off, got {retain_buffers!r}")

    path = spec.resolve_vdn_checkpoint(vdn_checkpoint)
    free = (
        _effective_free_vram(model)
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
    _validate_branch_shapes(
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
        raise RuntimeError(
            f"{vdn_checkpoint}: required Stage-B adapter 'default' is missing")
    if apply_turbo_adapter and "turbo" not in adapters:
        raise RuntimeError(
            f"{vdn_checkpoint}: apply_turbo_adapter is enabled but this stage has no "
            "'turbo' adapter")

    converted = {}
    for name in sorted(wanted):
        state_dict, adapter_cfg = adapters[name]
        converted[name] = convert_adapter(state_dict, adapter_cfg)
        if verbose:
            _log.info("[vdn] adapter %s converted: %d modules", name, len(converted[name]))

    converted, removed = _apply_adapter_ablation(converted, adapter_ablation)
    if removed:
        _log.warning(
            "[vdn] diagnostic adapter ablation %s removed targets: %s",
            adapter_ablation,
            ", ".join(f"{name}={count}" for name, count in sorted(removed.items())),
        )

    report = apply_adapters(
        new_model,
        converted,
        strength,
        mode=lora_mode,
        stage_path=path,
        verbose=verbose,
    )
    _log.info(
        "[vdn] %s applied: blocks=%d radius=%d chunk=%d anchors=%s rule=%s "
        "branch=%s/%s buffers=%s backend=%s lora_mode=%s global_gate=%s "
        "adapter_ablation=%s adapters=%s",
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
        lora_mode,
        global_gate_mode,
        adapter_ablation,
        report,
    )
    return (new_model,)


class ApplyVDNH3:
    @classmethod
    def INPUT_TYPES(cls):
        names = spec.list_vdn_checkpoints()
        return {"required": {
            "model": ("MODEL",),
            "vdn_checkpoint": (
                names or ["<place a VDN stage directory under models/vdn>"],),
            "apply_turbo_adapter": ("BOOLEAN", {
                "default": True,
                "tooltip": "Apply the stage's released Turbo/DMD adapter when present. "
                           "Stage-DMD was trained for 8 sampling steps with Turbo on."}),
            "strength": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "Adapter strength. 1.0 is the released checkpoint setting."}),
            "lora_mode": (["merge", "bypass"], {
                "default": "merge",
                "tooltip": "merge uses normal Comfy weight patches. bypass uses "
                           "VDN-owned PyTorch forward post-hooks, pre-stages runtime "
                           "factors before H3 execution, and leaves pruned curve AdaLN "
                           "base weights untouched; fused INT8 fc2 remains native."}),
            "branch_weights": (["auto", "stream", "resident"], {
                "default": "auto",
                "tooltip": "auto: resident BF16 when the base-reserved VRAM budget "
                           "exceeds 1.5x branch size + 4 GiB; otherwise stream and "
                           "prefer native INT8 ConvRot when available. stream resolves "
                           "one block at a time with safe one-block prefetch when buffers "
                           "are retained. resident is a Comfy-managed additional model."}),
            "retain_buffers": (["auto", "on", "off"], {
                "default": "auto",
                "tooltip": "auto retains VDN-owned scratch when the base-reserved VRAM "
                           "budget is at least selected branch size + 10 GiB. Scratch is "
                           "leased per execution; concurrent/nested runs use isolated "
                           "transient storage."}),
            "verbose": ("BOOLEAN", {"default": False}),
            "attention_backend": (["grouped", "flex"], {
                "default": "grouped",
                "tooltip": "grouped is the portable exact-window path; flex uses "
                           "PyTorch FlexAttention and falls back to grouped on failure."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model_patch/video"
    DESCRIPTION = (
        "Apply the released VDN-H3 hybrid-attention transform and adapters to ComfyUI's "
        "native MiniMax-H3 MODEL using checkpoint-bound architecture settings.")

    def apply(self, model, vdn_checkpoint, apply_turbo_adapter, strength, lora_mode,
              branch_weights, retain_buffers, attention_backend, verbose):
        return _apply_vdn(
            model, vdn_checkpoint, strength, lora_mode, branch_weights,
            attention_backend, verbose,
            apply_turbo_adapter=apply_turbo_adapter,
            retain_buffers=retain_buffers)


class ApplyVDNH3Advanced:
    @classmethod
    def INPUT_TYPES(cls):
        names = spec.list_vdn_checkpoints()
        return {"required": {
            "model": ("MODEL",),
            "vdn_checkpoint": (
                names or ["<place a VDN stage directory under models/vdn>"],),
            "apply_turbo_adapter": ("BOOLEAN", {"default": True}),
            "stage_b_strength": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "turbo_strength": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "lora_mode": (["merge", "bypass"], {
                "default": "merge",
                "tooltip": "bypass uses VDN-owned PyTorch forward post-hooks and "
                           "never replaces module.forward or materializes projected "
                           "curve AdaLN base weights; fused INT8 fc2 stays native."}),
            "branch_weights": (["auto", "stream", "resident"], {"default": "auto"}),
            "retain_buffers": (["auto", "on", "off"], {"default": "auto"}),
            "verbose": ("BOOLEAN", {"default": False}),
            "attention_backend": (["grouped", "flex"], {"default": "grouped"}),
            "architecture_mode": (["checkpoint", "override"], {
                "default": "checkpoint",
                "tooltip": "checkpoint uses model_spec.json exactly. override enables "
                           "the ablation fields below and is not the trained spec."}),
        }, "optional": {
            "window_radius": ("INT", {
                "default": 1, "min": 0, "max": 8,
                "tooltip": "Used only when architecture_mode=override."}),
            "window_chunk": ("INT", {
                "default": 5, "min": 0, "max": 64,
                "tooltip": "Used only when architecture_mode=override; 0 = frame mode."}),
            "anchor_frames": (["both", "columns", "rows", "none"], {
                "default": "both",
                "tooltip": "Used only when architecture_mode=override."}),
            "text_state": ("BOOLEAN", {
                "default": True,
                "tooltip": "Used only when architecture_mode=override."}),
            "linear_branch": ("BOOLEAN", {
                "default": True,
                "tooltip": "Used only when architecture_mode=override; off is a "
                           "window-only ablation."}),
            "fast_kernels": ("BOOLEAN", {
                "default": False,
                "tooltip": "Compile selected mathematically equivalent branch helpers; "
                           "first run includes compile cost and eager remains fallback."}),
            "global_gate_mode": (["checkpoint", "video_only"], {
                "default": "checkpoint",
                "tooltip": "Audio-fidelity diagnostic. checkpoint reproduces the released "
                           "VDN gate on every packed row. video_only forces the already-dense "
                           "text/reference/audio rows to unit gate while retaining learned "
                           "gating on generated video rows."}),
            "adapter_ablation": (list(_ADAPTER_ABLATION_TARGETS), {
                "default": "none",
                "tooltip": "Diagnostic target-class ablation. Removes exactly one Stage-B "
                           "or Turbo target class after adapter conversion; surviving targets "
                           "and strengths are unchanged."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model_patch/video"
    DESCRIPTION = (
        "VDN-H3 advanced controls. Checkpoint architecture is the default; ablations "
        "are opt-in and leave released-checkpoint behavior unchanged by default.")

    def apply(self, model, vdn_checkpoint, apply_turbo_adapter, stage_b_strength,
              turbo_strength, lora_mode, branch_weights, retain_buffers,
              attention_backend, verbose, architecture_mode="checkpoint",
              window_radius=1, window_chunk=5, anchor_frames="both", text_state=True,
              linear_branch=True, fast_kernels=False, global_gate_mode="checkpoint",
              adapter_ablation="none"):
        if architecture_mode not in ("checkpoint", "override"):
            raise ValueError(f"invalid architecture_mode {architecture_mode!r}")
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
        return _apply_vdn(
            model, vdn_checkpoint, strength, lora_mode, branch_weights,
            attention_backend, verbose,
            apply_turbo_adapter=apply_turbo_adapter,
            cfg_overrides=overrides,
            fast_kernels=fast_kernels,
            retain_buffers=retain_buffers,
            global_gate_mode=global_gate_mode,
            adapter_ablation=adapter_ablation)


NODE_CLASS_MAPPINGS = {
    "ApplyVDNH3": ApplyVDNH3,
    "ApplyVDNH3Advanced": ApplyVDNH3Advanced,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyVDNH3": "Apply VDN-H3 (MiniMax-H3 Hybrid Attention)",
    "ApplyVDNH3Advanced": "Apply VDN-H3 Advanced (Checkpoint / Ablations)",
}
