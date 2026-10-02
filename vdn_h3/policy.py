"""VRAM-aware branch-file and scratch policy layered over strict VDN loading.

Upstream v1.4 introduced useful automatic decisions for branch residency and scratch
retention. This adaptation deliberately keeps those heuristics separate from
``spec.py``'s validation and file-lifetime contract. In particular, selecting the
INT8 ConvRot branch never reintroduces persistent ``safe_open`` handles or a private
GPU weight cache.
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict

from vdn_h3 import spec

_log = logging.getLogger("comfy.vdn")
_GIB = 1 << 30
RETAINED_SAMPLING_HEADROOM_BYTES = 10 * _GIB


def branch_candidates(path):
    return (
        os.path.join(path, "linear_branch", spec.BRANCH_FILE),
        os.path.join(path, "linear_branch", spec.BRANCH_FILE_INT8),
    )


def select_branch_file(path, prefer_int8=False):
    plain, quant = branch_candidates(path)
    if prefer_int8 and os.path.isfile(quant):
        return quant
    if os.path.isfile(plain):
        return plain
    if os.path.isfile(quant):
        return quant
    return plain


def auto_branch_policy(path, free_bytes):
    """Return ``(stream|resident, prefer_int8)`` without unsafe quant residency.

    Match upstream v1.4's 1.5x-stage + 4 GiB headroom rule for the ordinary BF16
    branch. When that does not fit and an INT8 ConvRot branch exists, prefer it but
    keep it streamed: this repository's resident mode is a real Comfy additional
    ModelPatcher and intentionally fails closed for QuantizedTensor parameters.
    """
    plain, quant = branch_candidates(path)
    have_plain = os.path.isfile(plain)
    have_quant = os.path.isfile(quant)
    size_plain = os.path.getsize(plain) if have_plain else 0
    size_quant = os.path.getsize(quant) if have_quant else 0

    def fits(size):
        return free_bytes > 1.5 * size + 4 * _GIB

    if have_plain and fits(size_plain):
        mode, prefer_int8 = "resident", False
        selected_size = size_plain
    elif have_quant:
        # Quantized resident ownership is not silently emulated with an unmanaged
        # cache. Stream the smaller native representation instead.
        mode, prefer_int8 = "stream", True
        selected_size = size_quant
    else:
        mode, prefer_int8 = "stream", False
        selected_size = size_plain

    _log.info(
        "[vdn] branch_weights=auto: %s / %s (%.1f GiB free; selected %.2f GiB%s)",
        "int8_convrot" if prefer_int8 else "bf16",
        mode,
        free_bytes / _GIB,
        selected_size / _GIB if selected_size else 0.0,
        "; quantized branch stays streamed under managed-lifecycle policy"
        if prefer_int8 else "",
    )
    return mode, prefer_int8


def auto_retain_policy(path, prefer_int8, free_bytes):
    """Adopt upstream's stage-size + 10 GiB scratch-retention headroom rule."""
    branch_path = select_branch_file(path, prefer_int8=prefer_int8)
    size = os.path.getsize(branch_path) if os.path.isfile(branch_path) else 0
    retain = free_bytes >= size + RETAINED_SAMPLING_HEADROOM_BYTES
    _log.info(
        "[vdn] retain_buffers=auto: %s (%.1f GiB free; branch %.2f GiB + 10 GiB)",
        "retained" if retain else "transient",
        free_bytes / _GIB,
        size / _GIB,
    )
    return retain


def _stage_identity(path, branch_path):
    files = [branch_path, os.path.join(path, "model_spec.json")]
    adapters_root = os.path.join(path, "adapters")
    if os.path.isdir(adapters_root):
        for name in sorted(os.listdir(adapters_root)):
            directory = os.path.join(adapters_root, name)
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                candidate = os.path.join(directory, filename)
                if os.path.isfile(candidate):
                    files.append(candidate)
    return tuple(spec.file_identity(filename) for filename in files)


_CACHE = OrderedDict()
_MAX_CACHE = 4


def load_vdn_checkpoint(path, prefer_int8=False):
    """Strict stage load using the policy-selected branch file.

    Adapter tensors remain owned CPU copies and branch tensors remain file-identity
    descriptors resolved through ``spec.resolve_branch_weights``. No mmap survives a
    helper call.
    """
    branch_path = select_branch_file(path, prefer_int8=prefer_int8)
    if not os.path.isfile(branch_path):
        raise FileNotFoundError(
            f"{path}: missing linear_branch/{spec.BRANCH_FILE} and "
            f"{spec.BRANCH_FILE_INT8}")
    spec_path = os.path.join(path, "model_spec.json")
    if not os.path.isfile(spec_path):
        raise FileNotFoundError(f"{path}: missing model_spec.json")

    identity = _stage_identity(path, branch_path)
    hit = _CACHE.get(identity)
    if hit is not None:
        _CACHE.move_to_end(identity)
        return hit

    model_spec = spec._read_json(spec_path)
    cfg = spec.transform_config(model_spec)
    branch_sd = spec._lazy_branch_sd(branch_path)
    branches = spec._split_branches(path, branch_sd, cfg)

    adapters = {}
    adapters_root = os.path.join(path, "adapters")
    if os.path.isdir(adapters_root):
        for name in sorted(os.listdir(adapters_root)):
            directory = os.path.join(adapters_root, name)
            config_path = os.path.join(directory, "adapter_config.json")
            weights_path = os.path.join(directory, "adapter_model.safetensors")
            if not (os.path.isfile(config_path) and os.path.isfile(weights_path)):
                continue
            config = spec._read_json(config_path)
            state = spec._load_owned_safetensors(weights_path)
            spec._validate_adapter_weights(name, state, config, path)
            adapters[name] = (state, config)

    result = (cfg, branches, adapters, branch_path)
    _CACHE[identity] = result
    _CACHE.move_to_end(identity)
    while len(_CACHE) > _MAX_CACHE:
        _CACHE.popitem(last=False)
    return result
