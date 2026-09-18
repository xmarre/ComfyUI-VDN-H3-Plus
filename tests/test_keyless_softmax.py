from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vdn_h3.hybrid import VDNLayout
from vdn_h3.keyless_softmax import (
    KEYLESS_PROVIDER_KEY,
    KeylessVDNSoftmaxCompatibilityError,
    KeylessVDNSoftmaxProviderV1,
    KeylessVDNSoftmaxState,
    apply_keyless_softmax_reference,
)
from vdn_h3.softmax_provider import KEY, KEY_V4
from vdn_h3 import window


class _Contract:
    api = 1
    architecture = "h3_keyless_core50_v1"
    core_blocks = 50
    token_refiner = "native_qkv"
    token_refiner_blocks = 2
    heads = 56
    head_dim = 128
    inner_dim = 7168
    hidden_size = 5376
    routing_source = "value"
    retrieval_source = "raw_projected_value"
    routing_norm = "rmsnorm"
    routing_norm_epsilon = 1e-5
    rope_policy = "h3_split_half_96_v1"
    qv_order = "q_effective;v"
    projection_attr = "qv_proj"
    checkpoint_format_version = 1

    def identity(self):
        return (
            self.api,
            self.architecture,
            self.checkpoint_format_version,
            self.qv_order,
            self.heads,
            self.head_dim,
            self.inner_dim,
            self.routing_source,
            self.retrieval_source,
            self.rope_policy,
            "test-keyless",
        )


class _Routing:
    def __init__(self, recorder, *, block_index=0, rows=None, preprocessors=()):
        self.recorder = recorder
        self.block_index = block_index
        self.rows = rows
        self.preprocessors = tuple(preprocessors)

    def materialize(self, value):
        self.recorder["materialize"].append(
            None if self.rows is None else tuple(self.rows)
        )
        route = value * 0.5
        for preprocessor in self.preprocessors:
            if self.rows is None:
                route = preprocessor.fn(route)
            else:
                domain = SimpleNamespace(
                    indices=tuple(self.rows),
                    start=None,
                    stop=None,
                    identity="selected",
                )
                route = preprocessor.domain_fn(route, domain, domain)
        return route

    def select_value_rows(self, value, selector, *, log_measure=None, identity=None):
        assert log_measure is None
        assert torch.is_tensor(selector)
        rows = tuple(int(x) for x in selector.cpu().tolist())
        self.recorder["select"].append((rows, identity))
        selected = value.index_select(0, selector.to(value.device))
        return selected, _Routing(
            self.recorder,
            block_index=self.block_index,
            rows=rows,
            preprocessors=self.preprocessors,
        ), None


def _layout(
    *,
    video_start=0,
    video_end=8,
    num_frames=4,
    tokens_per_frame=2,
    seq_len=8,
    radius=0,
    chunk=2,
    anchor_frames="none",
):
    return VDNLayout(
        video_start,
        video_end,
        num_frames,
        tokens_per_frame,
        (1, tokens_per_frame),
        video_end,
        max(0, seq_len - video_end),
        seq_len,
        radius,
        chunk,
        anchor_frames,
    )


def _provider(layout, *, anchor_frames="none"):
    state = KeylessVDNSoftmaxState(
        cfg={"radius": 0, "chunk": 2, "anchor_frames": anchor_frames},
        retain_buffers=False,
        semantic=("minimax_h3_keyless_contract_v1", "test"),
    )
    state._layout.set(layout)
    return KeylessVDNSoftmaxProviderV1(state), state


def _run(provider, q, v, routing, **overrides):
    kwargs = {
        "q": q,
        "v": v,
        "heads": q.shape[1],
        "scale": q.shape[-1] ** -0.5,
        "routing": routing,
        "mask": None,
        "log_measure": None,
        "exact_blocks": None,
        "query_domain": None,
        "value_domain": None,
        "dense_fallback": lambda: torch.full_like(q, 77),
        "transformer_options": {},
    }
    kwargs.update(overrides)
    return provider(**kwargs)


def test_keyless_window_matches_materialized_oracle_and_selects_v_before_route():
    torch.manual_seed(101)
    q = torch.randn(8, 2, 4)
    v = torch.randn(8, 2, 4)
    layout = _layout()
    provider, _state = _provider(layout)
    recorder = {"select": [], "materialize": []}

    got = _run(provider, q, v, _Routing(recorder))
    want = window.window_softmax_grouped(
        q,
        v * 0.5,
        v,
        layout.video_start,
        layout.video_end,
        layout.num_frames,
        layout.tokens_per_frame,
        layout.bounds,
        q.shape[-1] ** -0.5,
        anchor_frames="none",
    )

    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert [rows for rows, _identity in recorder["select"]] == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    ]
    assert recorder["materialize"] == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    ]


def test_keyless_window_preserves_v4_query_position_maps_for_restricted_domains():
    torch.manual_seed(103)
    q = torch.randn(13, 2, 4)
    v = torch.randn(13, 2, 4)
    layout = _layout(video_start=2, video_end=10, seq_len=13)
    provider, state = _provider(layout)
    recorder = {"select": [], "materialize": []}
    seen = []

    def v4(native, q, k, v, **contract):
        if contract["kind"] == "local":
            query_map = contract["query_position_map"]
            assert query_map[0] == "vdn_query_positions"
            assert query_map[2] == state.query_position_owner_generation
            assert contract["sink_rows"] == 5
            assert q.shape[0] < k.shape[0]
            assert k.shape == v.shape
            seen.append(query_map)
        return native()

    got = _run(
        provider,
        q,
        v,
        _Routing(recorder),
        transformer_options={KEY_V4: v4},
    )
    want = window.window_softmax_grouped(
        q,
        v * 0.5,
        v,
        layout.video_start,
        layout.video_end,
        layout.num_frames,
        layout.tokens_per_frame,
        layout.bounds,
        q.shape[-1] ** -0.5,
        anchor_frames="none",
    )

    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert len(seen) == 2
    assert all(rows[0][:5] == (0, 1, 10, 11, 12) for rows in recorder["select"])


def test_exact_block_uses_dense_keyless_fallback_without_window_selection():
    q = torch.randn(8, 2, 4)
    v = torch.randn(8, 2, 4)
    provider, _state = _provider(_layout())
    recorder = {"select": [], "materialize": []}
    sentinel = torch.randn_like(q)

    got = _run(
        provider,
        q,
        v,
        _Routing(recorder),
        exact_blocks={0},
        dense_fallback=lambda: sentinel,
    )

    assert got is sentinel
    assert recorder == {"select": [], "materialize": []}


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("mask", torch.ones(1), "mask/log-measure"),
        ("log_measure", torch.ones(8), "mask/log-measure"),
        ("query_domain", object(), "external query/value"),
        ("value_domain", object(), "external query/value"),
    ],
)
def test_unreviewed_keyless_compositions_fail_closed(field, value, match):
    q = torch.randn(8, 2, 4)
    v = torch.randn(8, 2, 4)
    provider, _state = _provider(_layout())
    recorder = {"select": [], "materialize": []}

    with pytest.raises(KeylessVDNSoftmaxCompatibilityError, match=match):
        _run(provider, q, v, _Routing(recorder), **{field: value})

    assert recorder == {"select": [], "materialize": []}


def test_selected_window_rejects_legacy_full_domain_keyless_preprocessor():
    q = torch.randn(8, 2, 4)
    v = torch.randn(8, 2, 4)
    provider, _state = _provider(_layout())
    recorder = {"select": [], "materialize": []}
    legacy = SimpleNamespace(
        identity="legacy-full-domain",
        fn=lambda route: route,
        domain_fn=None,
    )

    with pytest.raises(
        KeylessVDNSoftmaxCompatibilityError,
        match="domain-aware preprocessors",
    ):
        _run(
            provider,
            q,
            v,
            _Routing(recorder, preprocessors=(legacy,)),
        )

    assert recorder == {"select": [], "materialize": []}


def test_selected_window_passes_composed_domain_to_domain_aware_preprocessor():
    torch.manual_seed(104)
    q = torch.randn(8, 2, 4)
    v = torch.randn(8, 2, 4)
    layout = _layout()
    provider, _state = _provider(layout)
    recorder = {"select": [], "materialize": []}
    calls = []

    def domain_fn(route, value_domain, routing_position_domain):
        calls.append(
            (
                tuple(value_domain.indices),
                tuple(routing_position_domain.indices),
            )
        )
        return route * 2.0

    domain_aware = SimpleNamespace(
        identity="domain-aware",
        fn=lambda route: route,
        domain_fn=domain_fn,
    )
    got = _run(
        provider,
        q,
        v,
        _Routing(recorder, preprocessors=(domain_aware,)),
    )
    want = window.window_softmax_grouped(
        q,
        v,
        v,
        layout.video_start,
        layout.video_end,
        layout.num_frames,
        layout.tokens_per_frame,
        layout.bounds,
        q.shape[-1] ** -0.5,
        anchor_frames="none",
    )

    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert calls == [
        ((0, 1, 2, 3), (0, 1, 2, 3)),
        ((4, 5, 6, 7), (4, 5, 6, 7)),
    ]


def test_legacy_vdn_softmax_provider_is_not_silently_reused_for_keyless():
    q = torch.randn(8, 2, 4)
    v = torch.randn(8, 2, 4)
    provider, _state = _provider(_layout())
    recorder = {"select": [], "materialize": []}

    with pytest.raises(
        KeylessVDNSoftmaxCompatibilityError,
        match="provider-v4",
    ):
        _run(
            provider,
            q,
            v,
            _Routing(recorder),
            transformer_options={KEY: lambda native, *args, **kwargs: native()},
        )


def test_native_vdn_preprocess_is_not_reapplied_to_keyless_qv():
    q = torch.randn(8, 2, 4)
    v = torch.randn(8, 2, 4)
    provider, _state = _provider(_layout())
    recorder = {"select": [], "materialize": []}

    with pytest.raises(
        KeylessVDNSoftmaxCompatibilityError,
        match="routing transforms belong",
    ):
        _run(
            provider,
            q,
            v,
            _Routing(recorder),
            transformer_options={"vdn_attention_preprocess_v1": lambda *args: args},
        )


def _keyless_inner():
    attn = SimpleNamespace(
        qv_proj=object(),
        q_norm=object(),
        route_norm=object(),
        heads=56,
        head_dim=128,
    )
    model = SimpleNamespace(
        blocks=[SimpleNamespace(attn=attn) for _ in range(50)]
    )
    setattr(model, "minimax_h3_keyless_contract_v1", _Contract())
    return model


class _Patcher:
    def __init__(self, inner=None):
        self.inner = inner or _keyless_inner()
        self.model_options = {}
        self.object_patches = {}
        self.wrappers = []

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.inner

    def clone(self):
        clone = _Patcher(self.inner)
        clone.model_options = {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in self.model_options.items()
        }
        clone.object_patches = dict(self.object_patches)
        return clone

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.append((wrapper_type, key, wrapper))


def test_apply_reference_installs_only_keyless_provider_and_layout_wrapper():
    source = _Patcher()

    patched = apply_keyless_softmax_reference(
        source,
        radius=1,
        chunk=5,
        anchor_frames="both",
        retain_buffers=False,
    )

    assert source.model_options == {}
    assert patched is not source
    provider = patched.model_options["transformer_options"][KEYLESS_PROVIDER_KEY]
    assert isinstance(provider, KeylessVDNSoftmaxProviderV1)
    assert patched.object_patches == {}
    assert len(patched.wrappers) == 1
    assert patched.wrappers[0][1] == "vdn_h3_keyless_softmax_v1"


def test_apply_reference_refuses_foreign_keyless_provider():
    source = _Patcher()
    source.model_options = {
        "transformer_options": {KEYLESS_PROVIDER_KEY: object()}
    }

    with pytest.raises(
        KeylessVDNSoftmaxCompatibilityError,
        match="already installed",
    ):
        apply_keyless_softmax_reference(source)
