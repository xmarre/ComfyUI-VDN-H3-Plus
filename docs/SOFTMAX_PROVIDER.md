# Optional attention subcall providers

Production companion: [ComfyUI-Sol-H3 v0.1.5](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.5).

VDN keeps ownership of its trained window/global/anchor geometry, learned softmax gate, output projection and learned linear complement. External providers can only operate on domains VDN has already selected.

The provider contract is implemented in grouped retained attention. Full-domain preprocessing still occurs before VDN gathers local domains, and VDN remains responsible for the gather/scatter topology.

## v1

`transformer_options["vdn_softmax_provider_v1"]` remains supported:

```python
provider(native, q, k, v, *, kind, scale, square_aligned=False)
```

It returns `[query_rows, heads, head_dim]` with unchanged dtype/device. `native()` executes the original operation. `square_aligned=True` means Q and KV already describe the same ordered row domain, not merely equal tensor dimensions.

Normal MiniMax-H3 VDN windows include packed non-video/global rows in KV. Their requested video Q rows are therefore usually rectangular against the restricted KV domain.

## v2 square-domain compatibility

`transformer_options["vdn_softmax_provider_v2"]` remains available for square-QKV-only providers:

```python
provider(
    native, q, k, v,
    *, kind, scale, square_aligned=False,
    square_q=None, query_positions=None, sink_rows=0,
)
```

For a grouped local call, VDN constructs K/V in its existing order `[global_rows, permitted_window_rows]`. `square_q`, when supplied, contains the real Q rows from those exact same packed indices and in the same order. `query_positions` maps the original requested Q rows into that square domain.

This path does not broaden VDN attention, but it evaluates extra disposable Q rows and is retained only for compatibility with providers that genuinely require square Q/K/V.

## v3 direct rectangular contract

`transformer_options["vdn_softmax_provider_v3"]` remains the direct rectangular contract for providers that do not need physical query-position metadata:

```python
provider(
    native, q, k, v,
    *, kind, scale, square_aligned=False, sink_rows=0,
)
```

For local calls, `q` contains only the requested VDN query rows while `k/v` contain the exact already-restricted VDN K/V domain. `sink_rows` is the leading global/prefix K/V row count. No `square_q` or `query_positions` payload is constructed.

## v4 mapped query positions

`transformer_options["vdn_softmax_provider_v4"]` extends the v3 rectangular tensors with VDN-owned query positions in the already-gathered restricted K/V domain:

```python
provider(
    native, q, k, v,
    *, kind, scale, square_aligned=False, sink_rows=0,
    query_position_map=None,
)
```

For a grouped local call, `query_position_map` is the immutable tagged tuple:

```text
("vdn_query_positions", 1,
 owner_generation, plan_digest, group_index,
 q_rows, kv_rows, sink_rows,
 query_position_runs)
```

Each run is `(q_begin, q_end, kv_begin)` and means that requested query rows `[q_begin:q_end]` correspond, in order, to restricted-domain K/V positions starting at `kv_begin`. Runs cover every requested Q row exactly once. The map describes positions only; it does not broaden or reorder the K/V domain.

VDN builds these maps from the same CPU window geometry that drives the actual grouped gather. `owner_generation` is specific to one Apply-VDN state, while `plan_digest` identifies the immutable grouped geometry. Consumers must validate ownership, group/count/sink agreement and the complete mapping before using it.

The v4 key has fail-closed precedence. If it is present but non-callable, VDN executes the supplied `native()` callback for that subcall rather than silently downgrading to an older sparse provider. A callable v4 provider can likewise reject a missing, malformed or unsupported map by invoking `native()` on the already-restricted Q/K/V domain.

When v4 is present, VDN does not construct v2's square-Q compatibility payload. Global and anchor calls carry no mapped local-domain contract and remain VDN-native. Training/reference grouped execution is unchanged.

Current dispatch order is v4 when present, then v3, v2, v1, and finally native. Older consumers that do not publish v4 keep their existing signatures and dispatch behavior.

### Preflight geometry

Each VDN attention forward exposes `vdn_query_position_plan_v1(options, layout)`. It derives an immutable CPU summary from the supplied native MiniMax-H3 `PackedLayout` and the captured VDN configuration; it does not read a previous runtime `state.layout`. The summary contains the owner generation, plan digest and ordered v4 maps used by grouped native execution.

Unsupported/opaque layouts, external or reduced sequence ownership, Flex routing, or invalid native layout geometry return no mapped preflight. This allows forecast/history consumers to treat unknown mapped routing as actual-only instead of predicting from stale state.

## Full-domain QKV preprocessing

`transformer_options["vdn_attention_preprocess_v1"]` is an optional shape-preserving preprocessing callable:

```python
preprocess(q, k, v, *, heads, transformer_options)
```

For grouped execution it runs once on the complete post-RoPE packed tensors inside `window_softmax_grouped_runtime`, before VDN gathers local row domains. This is required for transforms such as Untwist whose metadata uses original packed-row coordinates. Shape, dtype and device must remain unchanged.

Generic model-level dense attention overrides still do not automatically leak into VDN's trained local operator. The explicit provider/preprocess contracts define composition instead.

## Compatibility

v1/v2/v3 remain callable without additional keyword arguments. A v4-capable provider is a paired capability: VDN transports exact positions, while the provider decides whether that mapping is representable by its backend. Missing or unsupported mapped capability must use the same restricted-domain native callback; equal Q/K tensor lengths alone are not proof that rows are aligned.

The provider contracts do not transfer ownership of VDN's learned gate, output projection, linear complement, global/anchor topology, external/reduced sequence handling, or other VDN runtime ownership.

The separate PR #8 audio-fidelity experiment is not part of this provider-v4 release and remains unreleased.

## Validation and evidence boundary

The v4 implementation has CPU tests for exact restricted-domain mapping, owner-bound preflight, provider precedence, malformed-v4 native fallback, and suppression of the v2 square-Q payload. Paired Sol-H3 v0.1.5 production validation additionally passed real SM120 same-input validation, controlled first-high replay, the historical-M timing gate, and the representative 2x7-second Target Input trajectory.

The released v3 stack remains historical evidence for direct rectangular routing. v4 changes the numerical routing policy only when a paired provider validates and consumes the explicit map; it does not retroactively change the behavior of older providers.
