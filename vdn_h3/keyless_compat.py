"""Architecture guard for released VDN-H3 checkpoints.

Released VDN stages are trained against the native MiniMax-H3 Q/K/V core.  Their
hybrid branch consumes raw projected Q, K and V, including raw K in the learned
linear complement.  A Keyless H3 base intentionally removes persistent K from the
50 core blocks.  Reinterpreting V as K, creating a fake ``qkv_proj`` alias, or
silently dropping the learned complement would therefore change the checkpoint's
operator rather than make it compatible.

This module is intentionally independent of the MiniMax-H3-Keyless package.  It
recognizes the public contract key only to produce a precise fail-closed boundary.
Future Keyless-native VDN checkpoints need a separate checkpoint/schema contract
and execution path; they must not weaken this guard for existing released stages.
"""
from __future__ import annotations

from typing import Any

KEYLESS_CONTRACT_KEY = "minimax_h3_keyless_contract_v1"
KEYLESS_ARCHITECTURE = "h3_keyless_core50_v1"
KEYLESS_ROUTING_SOURCE = "value"
KEYLESS_RETRIEVAL_SOURCE = "raw_projected_value"
KEYLESS_QV_ORDER = "q_effective;v"


class VDNBaseCompatibilityError(RuntimeError):
    """The loaded base cannot preserve the released VDN checkpoint semantics."""


def advertised_keyless_identity(model: Any) -> tuple[Any, ...] | None:
    """Return the advertised Keyless semantic identity or fail on a malformed ad.

    The purpose here is not to authorize Keyless execution.  The exact public
    fields are checked so a typo/collision on the contract key is not reported as
    a supported Keyless architecture.  Every successfully recognized identity is
    still rejected by :func:`require_released_qkv_base`.
    """
    if model is None or not hasattr(model, KEYLESS_CONTRACT_KEY):
        return None
    contract = getattr(model, KEYLESS_CONTRACT_KEY)
    expected = {
        "api": 1,
        "architecture": KEYLESS_ARCHITECTURE,
        "core_blocks": 50,
        "token_refiner": "native_qkv",
        "token_refiner_blocks": 2,
        "heads": 56,
        "head_dim": 128,
        "inner_dim": 7168,
        "hidden_size": 5376,
        "routing_source": KEYLESS_ROUTING_SOURCE,
        "retrieval_source": KEYLESS_RETRIEVAL_SOURCE,
        "routing_norm": "rmsnorm",
        "routing_norm_epsilon": 1e-5,
        "rope_policy": "h3_split_half_96_v1",
        "qv_order": KEYLESS_QV_ORDER,
        "projection_attr": "qv_proj",
        "checkpoint_format_version": 1,
    }
    mismatches = []
    for name, wanted in expected.items():
        if not hasattr(contract, name):
            mismatches.append(f"missing {name}")
            continue
        actual = getattr(contract, name)
        if actual != wanted:
            mismatches.append(f"{name}={actual!r} (expected {wanted!r})")
    if mismatches:
        raise VDNBaseCompatibilityError(
            f"malformed/unsupported {KEYLESS_CONTRACT_KEY}: " + ", ".join(mismatches)
        )

    identity_fn = getattr(contract, "identity", None)
    if not callable(identity_fn):
        raise VDNBaseCompatibilityError(
            f"{KEYLESS_CONTRACT_KEY} must expose callable identity()"
        )
    try:
        identity = tuple(identity_fn())
        hash(identity)
    except (TypeError, ValueError) as exc:
        raise VDNBaseCompatibilityError(
            f"{KEYLESS_CONTRACT_KEY}.identity() must return a hashable tuple-like value"
        ) from exc
    if not identity:
        raise VDNBaseCompatibilityError(
            f"{KEYLESS_CONTRACT_KEY}.identity() may not be empty"
        )
    return (KEYLESS_CONTRACT_KEY, *identity)


def require_released_qkv_base(model: Any):
    """Return native-QKV blocks or reject before checkpoint loading/patching.

    This is a structural gate only.  Existing downstream shape/checkpoint checks
    remain authoritative for the actual released VDN stage.
    """
    keyless = advertised_keyless_identity(model)
    if keyless is not None:
        raise VDNBaseCompatibilityError(
            "Released VDN-H3 checkpoints are QKV-trained and cannot be applied to "
            f"{KEYLESS_ARCHITECTURE}. The learned VDN linear complement consumes raw "
            "Q/K/V, while Keyless core attention intentionally has no persistent K. "
            "A Keyless-native VDN stage requires a separately trained checkpoint and "
            "explicit checkpoint/schema contract; refusing to substitute V for K or "
            "synthesize qkv_proj."
        )

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise VDNBaseCompatibilityError(
            "ApplyVDNH3 needs a current ComfyUI MiniMax-H3 MODEL "
            "(diffusion_model.blocks[].attn.qkv_proj)."
        )
    try:
        block_count = len(blocks)
    except TypeError as exc:
        raise VDNBaseCompatibilityError("MiniMax-H3 blocks are not sized") from exc
    if block_count <= 0:
        raise VDNBaseCompatibilityError(
            "ApplyVDNH3 needs a current ComfyUI MiniMax-H3 MODEL "
            "(diffusion_model.blocks[].attn.qkv_proj)."
        )

    for index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        if attention is None or not hasattr(attention, "qkv_proj"):
            raise VDNBaseCompatibilityError(
                f"ApplyVDNH3 requires native QKV attention on every core block; "
                f"block {index} has no attn.qkv_proj"
            )
        if not hasattr(attention, "q_norm") or not hasattr(attention, "k_norm"):
            raise VDNBaseCompatibilityError(
                f"ApplyVDNH3 requires native Q/K normalization on every core block; "
                f"block {index} is missing q_norm/k_norm"
            )
    return blocks


__all__ = [
    "KEYLESS_ARCHITECTURE",
    "KEYLESS_CONTRACT_KEY",
    "VDNBaseCompatibilityError",
    "advertised_keyless_identity",
    "require_released_qkv_base",
]
