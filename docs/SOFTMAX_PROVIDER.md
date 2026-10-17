# Optional attention subcall providers

Production companion: [ComfyUI-Sol-H3 v0.1.5](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.5).

VDN keeps ownership of its trained window/global/anchor geometry, learned softmax gate, output projection and learned linear complement. External providers can only operate on domains VDN has already selected.

The provider contract is implemented in grouped retained attention. Full-domain preprocessing still occurs before VDN gathers local domains, and VDN remains responsible for the gather/scatter topology.

## Optional bounded raw-QKV projection

The native VDN attention forward advertises
`_vdn_bounded_projection_api = 1`. The Flow partitioned VDN wrapper advertises
API 2 when it wraps an API-1 forward, supporting the identical raw projection
contract in both native and physical mixed/equal-grid domain execution.
Both forwards accept an optional
`transformer_options["sol_h3_vdn_qkv_projection_v1"]` callable:

```python
q, k, v = projection(qkv_proj, x, heads, head_dim)
```

This is a producer interface, not a softmax provider. The producer must invoke
the original projection module and return three separately allocated contiguous
tensors of shape `[packed_rows, heads, head_dim]`, with the same dtype and device
as `x`. Invalid carriers fail immediately. VDN then computes its learned linear
branch from the original raw Q/K/V, applies the existing Q/K RMSNorm and RoPE,
and executes its original grouped-window or dense softmax, learned gate,
projection and linear addition.

Native packed QKV shares a backing allocation, so VDN clones V to avoid
carrying that packed buffer beyond its useful lifetime. Independently allocated
V already owns its storage and does not require this clone. With a BF16
56-head H3 model at 100,000 rows, the standalone QKV+V-clone source allocations
are about 5.34 GiB. Three independent carriers and a 4096-row packed
projection temporary use about 4.17 GiB. These figures are storage estimates,
not GPU peak-memory measurements.

The interface is opt-in. Older Sol/VDN pairings, absent producer keys and
unsupported projection owners continue through the existing packed-QKV path.
Sol-H3 PR #41 offers a checked implementation and request telemetry for the
compatible VDN forward, including Flow's VDN-owned physical partitioned
producer when API 2 is present. Flow retains its physical query/key domains,
prefix key weighting, grouped softmax and learned linear behavior. VDN does
not cache producer activations or modify attention dispatch. CPU equivalence
does not demonstrate CUDA bitwise identity, VRAM savings, sampler speed or
decoded quality. Release acceptance
requires measured full-workflow A/B comparisons.

## Physical partitioned attention provider

`PARTITIONED_PROVIDER_API_VERSION = 1` and the bridged forward's
`_vdn_partitioned_attention_provider_api = 1` advertise backend-neutral dispatch
for Flow's physical partitioned sequences. These calls use the optional
`transformer_options["vdn_partitioned_attention_provider_v1"]` hook:

```python
provider(q, k, v, *, transformer_options, block_index, kind, scale,
         sink_rows, prefix_k_range, prefix_log_key_measure, semantic_digest,
         query_position_map=None, force_dense=False)
```

Q is `[requested_rows, heads, head_dim]`; K/V contain only the already-gathered
union. Preprocessing has already run once over the complete post-RoPE domain.
The provider must preserve that union, apply the additive log key measure to
`prefix_k_range`, respect exact-query requests and return Q's shape, dtype and
device. Mapped queries carry the same VDN-owned position descriptor as v4.
There is no square-Q compatibility allocation. VDN retains its learned linear
branch, gates, projection and scatter ownership. Invalid provider presence or
results raise an error without switching backend.

Clients without this hook retain the legacy request-owned Sol dispatch. Ordinary
non-partitioned VDN uses the existing v1-v4 dispatch below. Flow's native provider
uses the selected ComfyUI backend with additive key masks; sparse backends that
cannot consume weighted or mapped rectangular domains use their native dense
fallback. This can increase time and memory. CPU equivalence does not establish
GPU performance or rendered quality.

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

## Partitioned target-query conditioning measure

Flow's opt-in `target_query_sink_measure` softmax diagnostic extends the
partitioned target-key measure over non-video rows for target-grid local
and anchor queries. A local group's K/V order is non-video rows, target-grid
head keys, then reduced-grid tail keys; `(0, target_end)` is therefore the
contiguous affected range. Reduced-grid and global queries retain the baseline
measure. Query maps, temporal windows, dense/sparse routing and learned linear
statistics are preserved. Anchor rows on different grids use separate dense
calls. Sol partitioned sink-measure API 1 is required. Normal remains the default.

The head domain includes both exact protected frames and mutable target-band
frames. This selector changes their attention weighting, not their timestep or
protection masks. It must not be combined with `raw_token_measure`. It is an
experimental weighting policy; output quality requires matched rendered runs.

## Zero-copy grouped query intervals (draft GPU qualification)

Both native retained grouped VDN attention and Flow physical partitioned VDN
now use a Q tensor slice instead of `index_select` **only** when CPU-owned
query geometry describes a single contiguous interval and the input Q tensor
is contiguous, and the attention provider is native or explicitly guarantees read-only Q. Unknown providers retain isolated copies. This avoids an extra Q activation allocation, device gather,
and indexed output scatter per eligible attention group. Noncontiguous groups,
noncontiguous Q layouts, all key/value gather orders, query-position maps,
feature recurrence, and the grouped softmax implementation are unchanged.
The fast path is checked against the unchanged reference on CPU, and requires
RTX PRO 6000 measurements (peak allocated and stage latency) before any
full-stack speed/VRAM claim or release.

## Synchronous grouped K/V producer capability

A cooperating v4 or partitioned provider may expose an additive callable attribute
`grouped_kv_v1`:

```python
provider.grouped_kv_v1(
    key, value, k_scratch, v_scratch, window_index, sink_rows, consumer,
    transformer_options=options, force_dense=False,
)
```

`key` and `value` are the full post-RoPE, post-preprocessing THD arrays.
`window_index` contains VDN's exact ordered int64 window row indices on the same
device. The scratch views have exactly `sink_rows + len(window_index)` rows, with
the conditioning prefix already initialized. The provider fills only their window
suffix and calls `consumer(k_scratch, v_scratch)` synchronously. It returns that
consumer's attention result. It must not retain scratch, prepared summaries or the
consumer beyond the invocation. VDN reuses the scratch for the next group.

This permits Sol to fuse the existing two K/V gathers with its M64 summary
reduction, preserving the restricted-domain order and selector. The capability
belongs to the selected provider. Other providers and older Sol versions keep the
original gathers. No new softmax arguments or mandatory Sol import are introduced
for standalone VDN. Exceptions propagate without a second attention execution.
The learned linear branch and raw QKV projection contract are independent of this
capability and execute as before.

Flow's learned-linear output assembly also avoids a redundant full output copy:
one equal-grid run returns its epilogue result directly; multiple runs copy into
one destination and release each partial result before the next run. The same
matmuls and epilogue execute, including the existing optional fused epilogue.
This reduces simultaneously live output storage without claiming a measured
whole-generation peak or speed improvement.

Deploy Sol PR #41 with this PR through Patcher; apply the separate VDN PR #41
admission overlay first. See Sol's `docs/GROUPED_ATTENTION.md` and
`tools/grouped_attention_acceptance.py` for exhaustive GPU parity, component
latency and repeated-allocation checks. Full VDN participation and matched native
and Flow generation quality/performance remain acceptance requirements.
