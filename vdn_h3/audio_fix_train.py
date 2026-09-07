"""Training primitives for a standalone generated-audio correction LoRA.

This module deliberately trains against ComfyUI's already-installed production
MiniMax-H3 model instead of requiring a second BF16 Diffusers copy. The base H3,
VDN branch, Stage-B and Turbo stay frozen; only a small FP32 sidecar is trainable.

The sidecar targets the same portable row-wise projections used by inference:
``blocks.*.attn.qkv_proj``, ``blocks.*.attn.out_proj`` and ``blocks.*.mlp.fc1``.
It is active only on generated-audio rows. Export converts the fused Comfy layout
back to the ordinary Diffusers/PEFT names consumed by :mod:`vdn_h3.adapters`.
"""
from __future__ import annotations

import contextvars
import json
import math
import os
import shutil
import weakref
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class TrainAudioScope:
    audio_start: int
    audio_end: int
    enabled: bool = True
    checkpoint_blocks: bool = False


_SCOPE = contextvars.ContextVar("vdn_h3_audio_fix_train_scope", default=None)
_TARGET_SUFFIXES = (".attn.qkv_proj", ".attn.out_proj", ".mlp.fc1")


def current_train_scope():
    return _SCOPE.get()


def expected_comfy_targets(num_blocks: int = 50) -> tuple[str, ...]:
    """Return the exact portable target set for a current 50-block Comfy H3."""
    out = []
    for index in range(int(num_blocks)):
        root = f"blocks.{index}"
        out.extend((
            root + ".attn.qkv_proj",
            root + ".attn.out_proj",
            root + ".mlp.fc1",
        ))
    return tuple(out)


def _get_submodule(model, path):
    if hasattr(model, "get_submodule"):
        return model.get_submodule(path)
    cur = model
    for part in path.split("."):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


def _logical_features(module, path):
    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    if not isinstance(in_features, int) or not isinstance(out_features, int):
        raise TypeError(
            f"audio-fix target {path} is not Linear-like: {type(module).__name__}")
    if in_features <= 0 or out_features <= 0:
        raise ValueError(f"audio-fix target {path} has invalid feature dimensions")
    return in_features, out_features


def _quant_parts(weight):
    candidates = (weight, getattr(weight, "data", None))
    for candidate in candidates:
        if candidate is None:
            continue
        params = getattr(candidate, "_params", None)
        qdata = getattr(candidate, "_qdata", None)
        if params is not None:
            return params, qdata
    return None, None


def validate_production_quant_targets(model, num_blocks: int = 50) -> tuple[str, ...]:
    """Fail closed unless every training target is the installed INT8 ConvRot path."""
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) != int(num_blocks):
        got = None if blocks is None else len(blocks)
        raise RuntimeError(
            f"production audio-fix expects {num_blocks} MiniMax-H3 blocks, got {got}")
    targets = expected_comfy_targets(num_blocks)
    failures = []
    for path in targets:
        module = _get_submodule(model, path)
        _logical_features(module, path)
        params, qdata = _quant_parts(getattr(module, "weight", None))
        if params is None:
            failures.append(f"{path}: weight is not a Comfy QuantizedTensor")
            continue
        if not bool(getattr(params, "convrot", False)):
            failures.append(f"{path}: quantized weight is not ConvRot")
        if qdata is not None and getattr(qdata, "dtype", None) != torch.int8:
            failures.append(f"{path}: expected INT8 storage, got {qdata.dtype}")
    if failures:
        preview = "; ".join(failures[:8])
        if len(failures) > 8:
            preview += f"; ... and {len(failures) - 8} more"
        raise RuntimeError(
            "audio-fix INT8 production training requires the existing INT8/ConvRot H3; "
            + preview)
    return targets


def generated_audio_span(text_len: int, video_t: int, latent_h: int,
                         latent_w: int, audio_t: int) -> tuple[int, int]:
    """Resolve target-audio rows through Comfy's authoritative PackedLayout."""
    from comfy.ldm.minimax.model import PackedLayout

    layout = PackedLayout(
        int(text_len), int(video_t), int(latent_h), int(latent_w), int(audio_t))
    aa, ab, _kind = next(seg for seg in layout.segments if seg[2] == "audio")
    va, _vb, _ = next(seg for seg in layout.segments if seg[2] == "video")
    if ab != va:
        raise RuntimeError("MiniMax-H3 packed target audio is not immediately before video")
    if ab - aa != 2 * int(audio_t):
        raise RuntimeError(
            f"MiniMax-H3 audio row count {ab-aa} != stereo*latent_t {2*int(audio_t)}")
    return int(aa), int(ab)


class AudioFixPair(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32))
        self.scale = float(alpha) / float(rank)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        x = x.to(self.lora_A.dtype)
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


class TrainableAudioFixBank(nn.Module):
    """FP32 sidecar LoRAs attached to, but never owning, the frozen Comfy H3 tree."""

    def __init__(self, model, rank=32, alpha=32, targets=None):
        super().__init__()
        if int(rank) < 1 or float(alpha) <= 0:
            raise ValueError("audio-fix rank/alpha must be positive")
        # Do not assign the frozen H3 as a normal nn.Module attribute. Doing so would
        # register the 33B model as a child of the tiny sidecar, causing bank.to(),
        # bank.parameters() and bank.state_dict() to traverse/serialize the base model.
        object.__setattr__(self, "_model_ref", weakref.ref(model))
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.targets = tuple(targets or expected_comfy_targets(len(model.blocks)))
        self.pairs = nn.ModuleList()
        for path in self.targets:
            if not (path.startswith("blocks.") and path.endswith(_TARGET_SUFFIXES)):
                raise ValueError(f"unsupported audio-fix training target {path}")
            module = _get_submodule(model, path)
            in_features, out_features = _logical_features(module, path)
            self.pairs.append(AudioFixPair(in_features, out_features, self.rank, self.alpha))
        self._handles = []

    @property
    def model(self):
        model = self._model_ref()
        if model is None:
            raise RuntimeError("audio-fix frozen H3 model no longer exists")
        return model

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    @contextmanager
    def scope(self, audio_start, audio_end, *, enabled=True, checkpoint_blocks=False):
        audio_start, audio_end = int(audio_start), int(audio_end)
        if not 0 <= audio_start <= audio_end:
            raise ValueError("invalid generated-audio span")
        token = _SCOPE.set(TrainAudioScope(
            audio_start, audio_end, bool(enabled), bool(checkpoint_blocks)))
        try:
            yield
        finally:
            _SCOPE.reset(token)

    @contextmanager
    def disabled(self):
        token = _SCOPE.set(TrainAudioScope(0, 0, False, False))
        try:
            yield
        finally:
            _SCOPE.reset(token)

    def _hook(self, pair, path):
        def hook(_module, inputs, output):
            scope = _SCOPE.get()
            if scope is None:
                raise RuntimeError(
                    f"audio-fix training target {path} executed without an explicit scope")
            if not scope.enabled:
                return output
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError(f"audio-fix training target {path} needs Tensor input")
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(f"audio-fix training target {path} needs Tensor output")
            x = inputs[0]
            if x.ndim not in (2, 3) or output.ndim != x.ndim:
                raise RuntimeError(
                    f"audio-fix target {path} expected rank-2/3 packed rows, got "
                    f"input {tuple(x.shape)} output {tuple(output.shape)}")
            row_dim = 1 if x.ndim == 3 else 0
            if scope.audio_end > x.shape[row_dim]:
                raise RuntimeError(
                    f"audio-fix span [{scope.audio_start},{scope.audio_end}) exceeds "
                    f"{x.shape[row_dim]} packed rows at {path}")
            if scope.audio_start == scope.audio_end:
                return output
            if row_dim == 0:
                xa = x[scope.audio_start:scope.audio_end]
            else:
                xa = x[:, scope.audio_start:scope.audio_end]
            delta = pair(xa).to(dtype=output.dtype)
            # Training cannot use the inference runtime's in-place output-slice update:
            # downstream autograd/checkpoint recomputation needs a functional graph.
            if row_dim == 0:
                return torch.cat((
                    output[:scope.audio_start],
                    output[scope.audio_start:scope.audio_end] + delta,
                    output[scope.audio_end:],
                ), dim=0)
            return torch.cat((
                output[:, :scope.audio_start],
                output[:, scope.audio_start:scope.audio_end] + delta,
                output[:, scope.audio_end:],
            ), dim=1)
        return hook

    def install(self, model=None):
        if self._handles:
            raise RuntimeError("audio-fix training hooks already installed")
        model = self.model if model is None else model
        for path, pair in zip(self.targets, self.pairs):
            self._handles.append(
                _get_submodule(model, path).register_forward_hook(self._hook(pair, path)))
        return len(self._handles)

    def uninstall(self):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def export_peft(self, dtype=torch.float32):
        """Export ordinary source-layout LoRA tensors understood by convert_adapter()."""
        state = {}
        targets = []
        for path, pair in zip(self.targets, self.pairs):
            a = pair.lora_A.detach().to(device="cpu", dtype=dtype).contiguous()
            b = pair.lora_B.detach().to(device="cpu", dtype=dtype).contiguous()
            block = int(path.split(".", 2)[1])
            source_root = f"transformer_blocks.{block}"
            if path.endswith(".attn.qkv_proj"):
                if b.shape[0] % 3:
                    raise RuntimeError(f"fused QKV output is not divisible by three: {path}")
                q, k, v = b.chunk(3, dim=0)
                for suffix, part in (("to_q", q), ("to_k", k), ("to_v", v)):
                    source = source_root + f".attn.{suffix}"
                    targets.append(source)
                    state[f"{source}.lora_A.audio_fix.weight"] = a.clone()
                    state[f"{source}.lora_B.audio_fix.weight"] = part.contiguous()
            elif path.endswith(".attn.out_proj"):
                source = source_root + ".attn.to_out.0"
                targets.append(source)
                state[f"{source}.lora_A.audio_fix.weight"] = a
                state[f"{source}.lora_B.audio_fix.weight"] = b
            elif path.endswith(".mlp.fc1"):
                if b.shape[0] % 2:
                    raise RuntimeError(f"SwiGLU fc1 output is not divisible by two: {path}")
                gate, value = b.chunk(2, dim=0)
                # Comfy stores [gate; value], Diffusers/OpenVDN stores [value; gate].
                source = source_root + ".ff.net.0.proj"
                targets.append(source)
                state[f"{source}.lora_A.audio_fix.weight"] = a
                state[f"{source}.lora_B.audio_fix.weight"] = torch.cat(
                    (value, gate), dim=0).contiguous()
            else:  # pragma: no cover - constructor already rejects this
                raise RuntimeError(f"unhandled audio-fix target {path}")
        config = {
            "name": "audio_fix",
            "rank": self.rank,
            "alpha": self.alpha,
            "targets": targets,
            "rank_pattern": {},
            "alpha_pattern": {},
            "exact_targets": True,
            "scope": "generated_audio",
            "target_policy": "portable_sequence_linear",
            "source_stack": "comfy_int8_convrot_vdn_stage_b_turbo",
        }
        return state, config

    def save_adapter(self, out_dir, *, step, metadata=None, dtype=torch.float32):
        return save_standalone_adapter(
            self, out_dir, step=step, metadata=metadata, dtype=dtype)


def save_standalone_adapter(bank, out_dir, *, step, metadata=None, dtype=torch.float32):
    """Write only the tiny correction adapter; never duplicate the frozen H3/VDN stage."""
    if os.path.exists(out_dir):
        raise FileExistsError(out_dir)
    tmp = out_dir.rstrip("/") + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    try:
        state, config = bank.export_peft(dtype=dtype)
        with open(os.path.join(tmp, "adapter_config.json"), "w", encoding="utf-8") as fh:
            json.dump({"type": "lora", "version": 1, "config": config},
                      fh, indent=2, sort_keys=True)
            fh.write("\n")
        save_file(state, os.path.join(tmp, "adapter_model.safetensors"))
        with open(os.path.join(tmp, "training_metadata.json"), "w", encoding="utf-8") as fh:
            json.dump({"step": int(step), **dict(metadata or {})},
                      fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, out_dir)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out_dir


def _phi1(z):
    return torch.expm1(z) / z


def _phi2(z):
    return (_phi1(z) - 1.0) / z


def res_multistep_update(sample, denoised, sigma, sigma_next, *,
                         previous_denoised=None, previous_sigma_down=None,
                         previous_sigma=None):
    """Deterministic eta=0 Comfy RES-multistep interval used by production H3."""
    sigma = torch.as_tensor(sigma, dtype=sample.dtype, device=sample.device)
    sigma_next = torch.as_tensor(sigma_next, dtype=sample.dtype, device=sample.device)
    if previous_denoised is None or float(sigma_next) == 0.0:
        return sample + ((sample - denoised) / sigma) * (sigma_next - sigma)
    if previous_sigma_down is None or previous_sigma is None:
        raise ValueError("second-order RES update needs previous sigma history")
    old_down = torch.as_tensor(
        previous_sigma_down, dtype=sample.dtype, device=sample.device)
    old_sigma = torch.as_tensor(
        previous_sigma, dtype=sample.dtype, device=sample.device)
    t = -torch.log(sigma)
    t_old = -torch.log(old_down)
    t_next = -torch.log(sigma_next)
    t_prev = -torch.log(old_sigma)
    h = t_next - t
    c2 = (t_prev - t_old) / h
    z = -h
    b2 = torch.nan_to_num(_phi2(z) / c2, nan=0.0)
    b1 = torch.nan_to_num(_phi1(z) - b2, nan=0.0)
    return torch.exp(-h) * sample + h * (b1 * denoised + b2 * previous_denoised)


def differentiable_run_scans(backend, alpha, a_raw, b_raw, text_state=None):
    """Autograd-safe VDN recurrence with the same baddbmm arithmetic as inference."""
    with torch.autocast(device_type=a_raw.device.type, enabled=False):
        transitions, injections = backend.factor_apply(alpha, a_raw, b_raw)
        start = (
            torch.zeros_like(injections[0])
            if text_state is None else text_state.to(injections.dtype)
        )
        prefix = []
        state = start
        for frame in range(transitions.shape[0]):
            state = torch.baddbmm(injections[frame], state, transitions[frame])
            prefix.append(state)
        suffix = [None] * transitions.shape[0]
        state = start
        for frame in range(transitions.shape[0] - 1, -1, -1):
            state = torch.baddbmm(injections[frame], state, transitions[frame])
            suffix[frame] = state
        return torch.stack(prefix, dim=0), torch.stack(suffix, dim=0)


@contextmanager
def differentiable_vdn_runtime():
    """Replace only the inference scan-bank writer while a training process is active.

    RuntimeLinearBranch normally writes recurrence results into retained/preallocated
    buffers with ``out=``. That is correct and cheaper for inference but PyTorch rejects
    ``out=`` operations when their inputs participate in autograd. Training uses this
    functional equivalent and restores the production implementation afterward.
    """
    from vdn_h3 import retained

    previous = retained.run_scans_runtime
    retained.run_scans_runtime = differentiable_run_scans
    try:
        yield
    finally:
        retained.run_scans_runtime = previous


@contextmanager
def comfy_quant_training_mode():
    """Enable Comfy's quantized autograd plus the differentiable VDN recurrence."""
    import comfy.model_management as mm

    old_training = mm.in_training
    old_fp8_bwd = mm.training_fp8_bwd
    mm.in_training = True
    mm.training_fp8_bwd = False
    try:
        with differentiable_vdn_runtime():
            yield
    finally:
        mm.in_training = old_training
        mm.training_fp8_bwd = old_fp8_bwd


def install_block_checkpointing(model, _bank=None):
    """Checkpoint H3 blocks only during the graph-building audio-fix student forward."""
    originals = []
    for block in model.blocks:
        original = block.forward

        def wrapped(*args, _original=original, **kwargs):
            scope = _SCOPE.get()
            if (scope is not None and scope.enabled and scope.checkpoint_blocks
                    and torch.is_grad_enabled()):
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)
            return _original(*args, **kwargs)

        block.forward = wrapped
        originals.append((block, original))
    return originals


def restore_block_checkpointing(originals):
    for block, original in originals:
        block.forward = original


__all__ = [
    "TrainableAudioFixBank",
    "comfy_quant_training_mode",
    "current_train_scope",
    "differentiable_run_scans",
    "differentiable_vdn_runtime",
    "expected_comfy_targets",
    "generated_audio_span",
    "install_block_checkpointing",
    "res_multistep_update",
    "restore_block_checkpointing",
    "save_standalone_adapter",
    "validate_production_quant_targets",
]
