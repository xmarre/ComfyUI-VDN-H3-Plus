"""Reference Keyless-H3 softmax/window path for VDN.

This module implements only VDN's grouped softmax/window ownership for
`h3_keyless_core50_v1`. It does not load or reinterpret released VDN QKV-trained
linear-branch weights/adapters.

The important invariant is that each local restricted domain selects V exactly
once and derives the routing tensor from that selected V with the correspondingly
selected routing positions. No fake K/qkv projection is introduced.
"""
from __future__ import annotations

import contextvars
import uuid
from typing import Any

import torch

from comfy.patcher_extension import WrappersMP

from vdn_h3.hybrid import VDNLayout, make_layout_wrapper
from vdn_h3.keyless_compat import require_keyless_softmax_base
from vdn_h3.query_positions import bind_query_map, describe_window_geometry
from vdn_h3.runtime import RuntimeBufferOwner
from vdn_h3.softmax_provider import KEY, KEY_V2, KEY_V3, KEY_V4, PREPROCESS_KEY, dispatch
from vdn_h3.window import _sdpa, full_coverage, window_bounds

KEYLESS_PROVIDER_KEY = "minimax_h3_keyless_provider_v1"
KEYLESS_PROVIDER_IDENTITY = "vdn_h3_keyless_softmax_v1"
KEYLESS_VALUE_DOMAIN_KEY = "minimax_h3_keyless_value_domain_v1"
KEYLESS_QUERY_DOMAIN_KEY = "minimax_h3_keyless_query_domain_v1"
KEYLESS_ROUTING_POSITION_DOMAIN_KEY = "minimax_h3_keyless_routing_position_domain_v1"


class KeylessVDNSoftmaxCompatibilityError(RuntimeError):
    """The requested composition cannot preserve Keyless VDN softmax semantics."""


class KeylessVDNSoftmaxState:
    """Execution state shared by one applied Keyless softmax reference patch."""

    def __init__(self, *, cfg: dict[str, Any], retain_buffers: bool, semantic: tuple[Any, ...]):
        self.cfg = dict(cfg)
        self.branches = (None,) * 50
        self.softmax_backend = "grouped"
        self.query_position_owner_generation = "vdn-keyless-" + uuid.uuid4().hex
        self.runtime = RuntimeBufferOwner(bool(retain_buffers))
        self._layout = contextvars.ContextVar(
            f"vdn_keyless_layout_{id(self)}",
            default=None,
        )
        self.semantic = tuple(semantic)

    @property
    def layout(self):
        return self._layout.get()

    @property
    def retain_buffers(self):
        return self.runtime.retain


def _strict_exact_request(exact_blocks: Any, block_index: int) -> bool:
    if exact_blocks is None:
        return False
    if isinstance(exact_blocks, dict):
        return bool(exact_blocks.get(block_index, False))
    if isinstance(exact_blocks, (set, frozenset, tuple, list)):
        return block_index in exact_blocks
    raise KeylessVDNSoftmaxCompatibilityError(
        "unsupported minimax_h3_keyless_exact_blocks_v1 descriptor"
    )


def _tensor_indices(rows: tuple[int, ...], device: torch.device) -> torch.Tensor:
    if not rows:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(rows, dtype=torch.long, device=device)


def _frame_rows(
    frames: tuple[int, ...],
    *,
    video_start: int,
    tokens_per_frame: int,
    device: torch.device,
) -> torch.Tensor:
    parts = []
    for frame in frames:
        start = video_start + frame * tokens_per_frame
        parts.append(
            torch.arange(
                start,
                start + tokens_per_frame,
                dtype=torch.long,
                device=device,
            )
        )
    if not parts:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(parts)


class KeylessVDNSoftmaxProviderV1:
    """API-1 Keyless provider implementing only VDN's grouped softmax branch."""

    api = 1

    def __init__(self, state: KeylessVDNSoftmaxState):
        self.state = state
        self.identity = (
            KEYLESS_PROVIDER_IDENTITY,
            1,
            state.semantic,
            state.query_position_owner_generation,
            tuple(sorted(state.cfg.items())),
        )

    def _attend(
        self,
        transformer_options,
        q,
        route,
        value,
        *,
        kind,
        scale,
        square_aligned=False,
        sink_rows=0,
        query_position_map=None,
    ):
        return dispatch(
            transformer_options,
            lambda: _sdpa(q, route, value, scale, None),
            q,
            route,
            value,
            kind=kind,
            scale=scale,
            square_aligned=square_aligned,
            sink_rows=sink_rows,
            query_position_map=query_position_map,
        )

    def __call__(
        self,
        *,
        q,
        v,
        heads,
        scale,
        routing,
        mask,
        log_measure,
        exact_blocks,
        query_domain,
        value_domain,
        dense_fallback,
        transformer_options,
    ):
        layout = self.state.layout
        if layout is None:
            raise KeylessVDNSoftmaxCompatibilityError(
                "VDN Keyless softmax provider executed outside its layout wrapper"
            )
        if not isinstance(layout, VDNLayout):
            raise KeylessVDNSoftmaxCompatibilityError("VDN Keyless layout state is malformed")
        if not torch.is_tensor(q) or not torch.is_tensor(v):
            raise KeylessVDNSoftmaxCompatibilityError("Keyless VDN requires tensor Q and V")
        if q.ndim != 3 or v.ndim != 3 or q.shape != v.shape:
            raise KeylessVDNSoftmaxCompatibilityError(
                f"Keyless VDN requires aligned [rows,heads,dim] Q/V, got {tuple(q.shape)} and {tuple(v.shape)}"
            )
        if int(q.shape[1]) != int(heads):
            raise KeylessVDNSoftmaxCompatibilityError("Keyless VDN head count changed inside provider")
        if int(q.shape[0]) != int(layout.seq_len):
            raise KeylessVDNSoftmaxCompatibilityError(
                "Keyless VDN softmax currently requires the native packed sequence; "
                "reduced/mixed-grid row domains need their own value-domain transport"
            )
        if query_domain is not None or value_domain is not None:
            raise KeylessVDNSoftmaxCompatibilityError(
                "Keyless VDN softmax does not yet authorize external query/value row domains"
            )
        if transformer_options.get(KEYLESS_ROUTING_POSITION_DOMAIN_KEY) is not None:
            raise KeylessVDNSoftmaxCompatibilityError(
                "Keyless VDN softmax does not yet authorize an external routing-position domain"
            )
        if mask is not None or log_measure is not None:
            raise KeylessVDNSoftmaxCompatibilityError(
                "Keyless VDN softmax does not yet support Keyless mask/log-measure composition"
            )
        if transformer_options.get(PREPROCESS_KEY) is not None:
            raise KeylessVDNSoftmaxCompatibilityError(
                "vdn_attention_preprocess_v1 must not run on Keyless Q/V; "
                "routing transforms belong in minimax_h3_keyless_routing_preprocessors_v1"
            )
        if KEY_V4 not in transformer_options and any(
            key in transformer_options for key in (KEY, KEY_V2, KEY_V3)
        ):
            raise KeylessVDNSoftmaxCompatibilityError(
                "Keyless VDN softmax authorizes only provider-v4 subcalls; "
                "legacy VDN softmax provider APIs do not carry the reviewed query-position map"
            )

        block_index = getattr(routing, "block_index", None)
        if type(block_index) is not int or not 0 <= block_index < 50:
            raise KeylessVDNSoftmaxCompatibilityError(
                "Keyless routing spec has an invalid core block index"
            )
        if _strict_exact_request(exact_blocks, block_index):
            return dense_fallback()

        bounds = tuple(tuple(pair) for pair in layout.bounds)
        if full_coverage(bounds, layout.num_frames):
            return dense_fallback()

        select_rows = getattr(routing, "select_value_rows", None)
        materialize = getattr(routing, "materialize", None)
        if not callable(select_rows) or not callable(materialize):
            raise KeylessVDNSoftmaxCompatibilityError(
                "Keyless routing spec must expose select_value_rows() and materialize()"
            )
        legacy_preprocessors = [
            getattr(preprocessor, "identity", "<unknown>")
            for preprocessor in tuple(getattr(routing, "preprocessors", ()) or ())
            if not callable(getattr(preprocessor, "domain_fn", None))
        ]
        if legacy_preprocessors:
            raise KeylessVDNSoftmaxCompatibilityError(
                "Keyless VDN selected-domain routing requires domain-aware preprocessors; "
                "legacy full-domain preprocessors are not safe after row selection: "
                + ", ".join(str(identity) for identity in legacy_preprocessors)
            )

        geometry = describe_window_geometry(
            int(layout.video_start),
            int(layout.video_end),
            int(layout.num_frames),
            int(layout.tokens_per_frame),
            bounds,
            str(self.state.cfg["anchor_frames"]),
            int(layout.seq_len),
        )

        out = torch.empty_like(q)
        device = q.device
        global_idx = _tensor_indices(geometry.global_rows, device)
        full_route = None

        def ensure_full_route():
            nonlocal full_route
            if full_route is None:
                full_route = materialize(v)
                if full_route.shape != v.shape:
                    raise KeylessVDNSoftmaxCompatibilityError(
                        "Keyless routing materialization changed full-domain topology"
                    )
            return full_route

        if global_idx.numel():
            out[global_idx] = self._attend(
                transformer_options,
                q.index_select(0, global_idx),
                ensure_full_route(),
                v,
                kind="global",
                scale=scale,
            )

        for group in geometry.groups:
            q_idx = _frame_rows(
                group.query_frames,
                video_start=geometry.video_start,
                tokens_per_frame=geometry.tokens_per_frame,
                device=device,
            )
            win_idx = _frame_rows(
                group.key_frames,
                video_start=geometry.video_start,
                tokens_per_frame=geometry.tokens_per_frame,
                device=device,
            )
            if global_idx.numel():
                domain_idx = torch.cat((global_idx, win_idx))
            else:
                domain_idx = win_idx

            selected_v, selected_routing, selected_measure = select_rows(
                v,
                domain_idx,
                log_measure=None,
                identity=(
                    f"vdn-keyless:{self.state.query_position_owner_generation}:"
                    f"{geometry.plan_digest}:group-{group.group_index}"
                ),
            )
            if selected_measure is not None:
                raise KeylessVDNSoftmaxCompatibilityError(
                    "Keyless VDN row selection unexpectedly produced a log measure"
                )
            route = selected_routing.materialize(selected_v)
            if route.shape != selected_v.shape:
                raise KeylessVDNSoftmaxCompatibilityError(
                    "Keyless VDN selected-domain routing changed V topology"
                )
            q_rows = q.index_select(0, q_idx)
            out[q_idx] = self._attend(
                transformer_options,
                q_rows,
                route,
                selected_v,
                kind="local",
                scale=scale,
                square_aligned=bool(group.square_aligned),
                sink_rows=int(group.sink_rows),
                query_position_map=bind_query_map(
                    geometry,
                    int(group.group_index),
                    self.state.query_position_owner_generation,
                ),
            )

        for start, stop in geometry.anchor_slices:
            out[start:stop] = self._attend(
                transformer_options,
                q[start:stop],
                ensure_full_route(),
                v,
                kind="anchor",
                scale=scale,
            )
        return out


def apply_keyless_softmax_reference(
    model,
    *,
    radius: int = 1,
    chunk: int = 5,
    anchor_frames: str = "both",
    retain_buffers: bool = False,
):
    """Install VDN grouped softmax ownership without any released VDN branch weights."""
    if type(radius) is not int or radius < 0:
        raise ValueError("radius must be a non-negative integer")
    if type(chunk) is not int or chunk < 0:
        raise ValueError("chunk must be a non-negative integer")
    if anchor_frames not in {"none", "columns", "rows", "both"}:
        raise ValueError(f"unsupported VDN anchor mode {anchor_frames!r}")

    diffusion_model = model.get_model_object("diffusion_model")
    semantic, blocks = require_keyless_softmax_base(diffusion_model)

    cfg = {
        "radius": radius,
        "chunk": chunk,
        "anchor_frames": anchor_frames,
    }
    state = KeylessVDNSoftmaxState(
        cfg=cfg,
        retain_buffers=retain_buffers,
        semantic=semantic,
    )
    provider = KeylessVDNSoftmaxProviderV1(state)

    cloned = model.clone()
    options = dict(getattr(cloned, "model_options", {}) or {})
    transformer_options = dict(options.get("transformer_options", {}) or {})
    existing = transformer_options.get(KEYLESS_PROVIDER_KEY)
    if existing is not None:
        raise KeylessVDNSoftmaxCompatibilityError(
            "A Keyless attention provider is already installed. "
            "VDN Keyless softmax must be the authoritative provider for this reference path."
        )
    for index in range(len(blocks)):
        patched = getattr(cloned, "object_patches", {}).get(
            f"diffusion_model.blocks.{index}.attn.forward"
        )
        if getattr(patched, "_vdn_forward", False):
            raise KeylessVDNSoftmaxCompatibilityError(
                "Released QKV VDN attention patches are already installed; "
                "Keyless softmax reference cannot reinterpret them."
            )

    transformer_options[KEYLESS_PROVIDER_KEY] = provider
    options["transformer_options"] = transformer_options
    cloned.model_options = options
    cloned.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL,
        "vdn_h3_keyless_softmax_v1",
        make_layout_wrapper(state),
    )
    return cloned


__all__ = [
    "KEYLESS_PROVIDER_IDENTITY",
    "KEYLESS_PROVIDER_KEY",
    "KeylessVDNSoftmaxCompatibilityError",
    "KeylessVDNSoftmaxProviderV1",
    "KeylessVDNSoftmaxState",
    "apply_keyless_softmax_reference",
]
