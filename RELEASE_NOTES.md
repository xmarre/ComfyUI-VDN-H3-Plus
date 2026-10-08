## Unreleased: native routing on equal-grid partitions

Equal-grid partitions (Flow's target-grid high stage, domain-uniform streams and
same-grid control) are ordinary uniform clips with unit key measure. Their local
query groups now use the selected backend's native routing for every frame,
including protected-prefix and target-band frames, and keep only the native
conditioning sink. Sol therefore applies its sparse selection there exactly as
in a non-partitioned call, instead of dense attention for the prefix/band frames
and pinned protected keys. Mixed-grid partitions keep the dense prefix and
boundary-group policy. Global and anchor queries stay dense, as in native VDN.
The bridge advertises `_vdn_partitioned_uniform_query_policy`. GPU time and
rendered effect are unvalidated.

## Unreleased: backend-neutral partitioned attention

Physical partitioned sequences can dispatch through the optional
`vdn_partitioned_attention_provider_v1` hook. It transports the complete
gathered Q/K/V domain, key measure, mapped-query descriptor and exact-query
requirement independently of Sol ownership. Bridged forwards advertise provider
API 1. Existing clients without the hook retain Sol dispatch; ordinary VDN v1-v4
providers and learned branch arithmetic are unchanged. Results must retain Q's
shape, dtype and device. Native dense fallback may increase attention time and
memory; GPU validation remains necessary.

## Unreleased: Flow domain-uniform streams

Accept Flow's opt-in target-band `target_band_context=domain_uniform_v1`, which
evaluates two uniform-grid hidden streams per model call.

- A stream publishes an ordinary equal-grid partition contract plus the
  `h3_flow_partitioned_domain_stream_v1` leaf. VDN then derives the stream's
  window layout (frames, rows per frame, window bounds, coverage) from that
  contract instead of from the call's native carrier layout. Text rows and the
  native carrier geometry are checked against the active native layout.
- Grouped softmax, global and anchor queries, the dense-query head and the
  learned linear complement operate on the stream's own rows only.
- Malformed leaves, non-uniform contracts, digest or row mismatches fail before
  attention. Calls without the leaf are unchanged.
- The bridged forward advertises `_vdn_partitioned_domain_stream_api = 1`.
  `VDNLayout` moves to `vdn_h3.window` and remains importable from
  `vdn_h3.hybrid`.

CPU tests cover layout derivation, fail-closed validation and grouped routing on
stream rows. GPU arithmetic, performance and rendered quality are unvalidated.

# ComfyUI-VDN-H3-Plus v1.5.8

Accept a target-grid native carrier for Flow's opt-in target-band continuation.

## Partition contract

- The partitioned exact-prefix contract may declare `native_carrier_grid="target"`
  with an integer `native_carrier_rows_per_frame`. The native (pre-partition)
  sequence is then the uniform target grid; the partition presented to VDN keeps
  its `[target-grid head | source-grid tail]` layout.
- External execution validates Core's native layout against the declared
  carrier. Undeclared, mismatched or non-integer row counts are rejected before
  attention.
- Contracts without the field keep the reduced-grid carrier with unchanged
  fields and semantic digest. The bridged forward advertises the accepted
  carriers in `_vdn_partitioned_native_carrier_grids`.
- The dense K/V workspace oracle covers the target carrier in FP32, FP16 and BF16.

Attention arithmetic, routing and the reduced-grid carrier path are unchanged.

## Coordinated release set

Update the coordinated components together. Every release links this same
version set and identifies its implementation PRs.

| Component | Release | Included PRs |
| --- | --- | --- |
| Flow-Aligned Regenerate | [v0.3.10](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/releases/tag/v0.3.10) | [#96](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/96) |
| Sol-H3 | [v0.1.9](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.9) | [#39](https://github.com/xmarre/ComfyUI-Sol-H3/pull/39) |
| VDN-H3-Plus | [v1.5.8](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.8) | [#38](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/38) |
| H3 Continuum-Plus | [v3.4.6](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/releases/tag/v3.4.6) | [#39](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/39), [#40](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/40) |
| Latent Upscaler-Plus | [v0.2.2](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus/releases/tag/v0.2.2) | unchanged |

[Spectrum MiniMax H3 v0.2.28](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.28)
is the unchanged companion. Separate Keyless, audio-training and rejected
decoded-geometry experiments are outside this release set.

The tested Core adapter repair is
[ComfyUI #16783](https://github.com/Comfy-Org/ComfyUI/pull/16783).
For INT8 fused MLP runtime adapters, retain that ComfyUI Patcher PR overlay until
the repair is available upstream. The independent Core #16720 optimization is
not included in this release set.

---

# ComfyUI-VDN-H3-Plus v1.5.7

Preserve resident H3 clones and bounded inference ownership while completing the
partitioned transport used by the coordinated target-grid continuation profile.

## Transport, memory and factorization

- Batch equal-grid spatial short-conv, same-grid temporal taps, frame statistics
  and learned-linear readout while preserving accumulation order and physical
  cross-grid mapping.
- Keep required resident clones at sampler admission, reuse block-local
  inference workspaces and reduce unnecessary raw-QKV/output lifetimes.
- Cache weight-dtype epsilon rounding and defer CUDA Cholesky status checks only
  inside a bounded execution owner. Flush at the required attention/prediction,
  stream and storage boundaries; clear statuses on cancellation and nesting.
  CPU, training and unsupported execution modes retain immediate checks.
- Keep the first generated local query group dense under the existing bounded
  boundary policy. Later groups retain their configured sparse route.
- Retain optional feature observations and the opt-in destination-grid temporal
  stencil. The default remains `native_grid_then_map_v1`; the experimental
  stencil is not promoted as a continuity repair.

The dense-boundary policy alone does not fix heterogeneous seams. The coordinated
same-grid profile has reported video/audio acceptance, without changing adapter
strength defaults or promoting the separate audio-training experiments.

## Coordinated release set

Update the coordinated components together. Every release links this same
version set and identifies its implementation PRs.

| Component | Release | Included PRs |
| --- | --- | --- |
| Flow-Aligned Regenerate | [v0.3.9](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/releases/tag/v0.3.9) | [#89](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/89), [#93](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/93) |
| Sol-H3 | [v0.1.8](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.8) | [#37](https://github.com/xmarre/ComfyUI-Sol-H3/pull/37) |
| VDN-H3-Plus | [v1.5.7](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.7) | [#33](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/33), [#34](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/34), [#35](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/35), [#36](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/36), [#37](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/37) |
| H3 Continuum-Plus | [v3.4.5](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/releases/tag/v3.4.5) | [#37](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/37), [#38](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/38) |
| Latent Upscaler-Plus | [v0.2.2](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus/releases/tag/v0.2.2) | [#16](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus/pull/16) |

[Spectrum MiniMax H3 v0.2.28](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.28)
is the unchanged companion. Separate Keyless, audio-training and rejected
decoded-geometry experiments are outside this release set.

The tested Core adapter repair is
[ComfyUI #16783](https://github.com/Comfy-Org/ComfyUI/pull/16783).
It remains an upstream review item, with upstream workflow approval and merge
controlled by Comfy-Org maintainers. For INT8 fused MLP runtime adapters,
retain that ComfyUI Patcher PR overlay until the repair is available upstream.
The independent Core #16720 optimization is not included in this release set.

---

# ComfyUI-VDN-H3-Plus v1.5.6

Coordinated production release with [ComfyUI-Sol-H3 v0.1.6](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.6), [MiniMax H3 Flow-Aligned Regenerate v0.3.6](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/releases/tag/v0.3.6), and [H3 Continuum v3.4.4](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/releases/tag/v3.4.4). [Spectrum MiniMax H3 v0.2.28](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.28) remains unchanged.

Production consolidation PRs: [VDN-H3-Plus #32](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/32), [Sol-H3 #32](https://github.com/xmarre/ComfyUI-Sol-H3/pull/32), [Flow #73](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/73), and [Continuum #34](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/34). VDN #32 consolidates the validated VDN #30 source line without merging the intermediate diagnostic PRs.

## Exact-prefix VDN transport and lifetime control

v1.5.6 completes the VDN side of the heterogeneous exact-prefix progressive path. The released normal route preserves the grouped VDN softmax domain, provider-v4 mapped query-position ownership, variable-grid learned linear complement, output projection and existing adapter arithmetic while allowing Flow to execute the protected target-grid prefix and generated source-grid suffix under one explicit partitioned contract.

The runtime lifetime work keeps retained VDN scratch under bounded ownership across progressive low/probe/high stages and releases it at the established quiescent boundaries. The final production fix adds sampler-admission eviction of unrelated resident models when retained buffers are active. This frees stale text-encoder/VAE residency before the H3 sampler needs the space while explicitly preserving the current H3 ModelPatcher.

That admission fix changes memory residency, not VDN mathematics: Q/K/V values, grouped support, learned branch arithmetic, adapter weights, scheduler/NFE ownership and Flow's partitioned geometry remain unchanged.

## Why the final VRAM fix is narrow

An earlier bounded-workset experiment improved allocator pressure but changed output behavior and was rejected. The shipped fix therefore does not replace VDN scratch/branch arithmetic. It keeps the accepted normal route and performs only the pre-sampling residency eviction before Core prepares the active sample.

On the production RTX PRO 6000 stack, this restored the later high stage to the healthy allocator/timing class instead of the previous spill/cliff behavior.

## Coordinated stack

- **Sol-H3 v0.1.6** consumes the mapped physical query-position contract through the real packaged SM120 CuTe backend.
- **Flow v0.3.6** uses the validated fast source-uniform exact-prefix continuation topology with 16 sampler-owned overlap ticks and no duplicate shadow low/probe lifetimes.
- **H3 Continuum v3.4.4** fixes physical prompt/audio ownership at continuation and terminal boundaries.

The release does not require a new Spectrum build; Spectrum v0.2.28 remains the companion version used in the current stack.

## Validation

The #30 exact head passed the full VDN CI matrix. Hardware validation showed the sampler-admission policy removed the third-high VRAM/performance cliff while preserving the accepted arithmetic path. The final Flow #70 acceptance run retained three continuation sampler lifetimes / two history boundaries and healthy high-stage execution.

## Release scope

Historical VDN linear-bypass, raw-token-measure, cross-grid short-conv and allocator diagnostics remain diagnostic evidence and are **not merged as independent production PRs**. Keyless research PRs are also excluded. This release consolidates the validated normal production tree and the narrow #30 admission fix into one mainline release commit.


# ComfyUI-VDN-H3-Plus v1.5.5

Coordinated production release with [ComfyUI-Sol-H3 v0.1.5](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.5) and [MiniMax H3 Flow-Aligned Regenerate v0.3.5](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/releases/tag/v0.3.5).

Implementation PRs: [VDN-H3-Plus #18](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/18), [Sol-H3 #14](https://github.com/xmarre/ComfyUI-Sol-H3/pull/14), and Flow [#33](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/33) + [#48](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/48).

## Motivation: the first-high artifact

The production MiniMax-H3 stack exposed a reproducible first-high visual artifact after the progressive low-to-high handoff when VDN retained grouped local attention was executed through Sol-H3 sparse attention.

Controlled same-input investigation isolated the boundary:

- replay R reproduced the broken first-high output;
- native W-window was clean while preserving VDN's restricted local K/V support and learned linear complement;
- W-full-support was also clean;
- global and anchor operations remained native.

The problem was therefore not that VDN's local K/V window was too small and not that its learned complement had to be removed. It was the coordinate contract between VDN's grouped local domain and Sol's exact-neighbor protection.

VDN gathers requested local Q rows separately from its restricted K/V domain, whose order is `[global_rows, permitted_window_rows]`. A requested query's **physical row position inside that gathered K/V domain is not its local Q ordinal**. The old provider contract supplied rectangular Q/K/V but did not transport that physical mapping, so Sol's normal ordinal rule `abs(q_block - kv_block) <= 1` could preserve the wrong neighboring K blocks exactly. In the captured failing group-10 geometry, Q physically corresponded to K positions `9245..10268` while the local Q block ordinals were only `0..15`.

Diagnostic M established the required correction on the preserved input: keep VDN's exact restricted K/V support and Sol's sparse approximation, but protect exact neighbors around each query's mapped physical K/V position.

## Resolution: provider v4 owns the mapping

v1.5.5 adds `vdn_softmax_provider_v4`. VDN now transports the missing information from the component that actually owns it.

For each grouped local call, VDN derives the query-position map from the **same pure CPU geometry that drives the real gather**, binds it to one Apply-VDN owner generation, and sends an immutable tagged affine-run description alongside the existing direct rectangular Q/K/V tensors:

```text
("vdn_query_positions", 1,
 owner_generation, plan_digest, group_index,
 q_rows, kv_rows, sink_rows,
 query_position_runs)
```

The map describes positions only. It does not broaden or reorder the restricted K/V domain, change Q/K/V values, or transfer ownership of VDN's learned softmax gate, output projection, linear complement, global/anchor operations or window topology.

Provider-v4 dispatch is fail-closed: a present malformed v4 entry runs the supplied native restricted-domain callback rather than silently downgrading to an older sparse provider. v1/v2/v3 remain compatible. When v4 is present, the old v2 square-Q compatibility payload is not constructed.

VDN also exposes `vdn_query_position_plan_v1(options, layout)` for preflight/history consumers. It derives the upcoming owner-bound native/grouped map directly from the supplied MiniMax-H3 `PackedLayout`, not from stale execution state. External/reduced and Flex-owned paths remain actual-only when the mapping cannot be proven.

The paired Sol-H3 v0.1.5 validates this map, compiles bounded K64 mapped-neighbor intervals and adds them to its real SM120 route as:

```text
new_exact = old_exact OR mapped_neighbor
```

That paired change is the production fix for the captured artifact.

## Flow / Continuum production path

The coordinated Flow v0.3.5 release standardizes **Progressive Handoff (Target Input)** and retires Mixed-Grid from the production acceptance path. Exact protected continuation stays on the target grid and uses the validated four-audio-tick guided overlap while restoring caller-owned exact video/audio values at output.

The final production stack validated for this release is:

```text
MiniMax H3 Flow-Aligned Regenerate v0.3.5
ComfyUI-Sol-H3                    v0.1.5
ComfyUI-VDN-H3-Plus              v1.5.5
```

The separate VDN PR #8 audio-fidelity/training experiment is **not included** in v1.5.5 and remains unreleased.

## Production validation

Real RTX PRO 6000 / SM120 validation of the coordinated stack included the same-input mapped route, controlled first-high replay, historical-M performance comparison, and the representative two-chunk production trajectory.

Run `00494` completed:

```text
17 logical calls
13 actual H3 NFE
4 Spectrum forecasts

low:    4 actual / 1 forecast
probe:  1 actual / 0 forecast
high:   2 actual / 1 forecast
later:  6 actual / 2 forecast
```

Mapped local routing remained direct with requested Q rows equal to kernel Q rows and zero square expansion. There was no mapped-local native fallback or kernel-unavailable failure. The decoded 14-second output showed no first-high corruption, frame shift, zoom-out, top-edge reveal, flash, grid artifact or physical AV seam; the chunk boundary was perceptually seamless.

The matched Sol production mapped-kernel timing gate measured `1.109472036 ms` versus `1.105535984 ms` for historical diagnostic M, a `+0.356031%` median delta inside the `+5%` acceptance budget.

00494 evidence hashes:

- runtime log: `699c17b6d85d186039b9179c4b99edb596e88fdda37d3f8814917c50b6905ca8`;
- metrics JSON: `c25a90af61d83c1430c3f29b9e99de7b0adb22f60d714e2d1392a269e6f0802e`;
- final MP4: `1ffb5ebff47417f2f9354d3ae7cdfa32b6b6e292cd661efb9ddc2fd818d66ab2`.

---

# ComfyUI-VDN-H3-Plus v1.5.4

v1.5.4 is a packaging/release follow-up to v1.5.3. Runtime VDN math and provider behavior are unchanged.

## Dedicated Comfy Registry identity

The Plus fork now publishes under the distinct registry package name `comfyui-vdn-h3-plus` with `PublisherId = "xmarre"` and display name `ComfyUI-VDN-H3-Plus`.

The previous package metadata still used upstream's `comfyui-vdn-h3` project name while changing only the publisher. That name is already associated with Saganaki22's upstream package, so the Comfy Registry correctly rejected the Plus fork publication with HTTP 403 even when a valid xmarre registry token was present.

This release therefore avoids claiming or overwriting upstream's registry identity. Existing Git installs are unaffected.

## Release pipeline hardening

Registry publication is now tied to a successfully completed GitHub release workflow instead of publishing immediately from an arbitrary `pyproject.toml` push.

Before publishing, the workflow verifies that:

- the requested/current release exists and is not a draft;
- the release tag targets current `main` rather than a stale commit;
- the release tag exactly matches `[project].version`;
- the registry package name is `comfyui-vdn-h3-plus`;
- the publisher is `xmarre`.

Release ZIP names, archive prefixes and GitHub release titles now consistently use the `ComfyUI-VDN-H3-Plus` fork name.

## Relationship to v1.5.3

All v1.5.3 functionality remains intact: rectangular softmax-provider API v3, lazy v2 square-Q compatibility, exactly-once full-domain preprocessing, INT8 ConvRot stage support/documentation, pruned AdaLN affine handling, and the production-validated Sol-H3 composition contract.

The separate PR #8 audio-fidelity experiment remains unreleased and is still not a dependency of this release.

# ComfyUI-VDN-H3 v1.5.3

v1.5.3 adds the composable VDN softmax-provider API used by ComfyUI-Sol-H3 v0.1.0, removes the historical square-Q compatibility cost for rectangular providers, and includes the post-v1.5.2 setup/documentation fixes for INT8 ConvRot stages and pruned AdaLN affine sidecars.

## VDN softmax-provider API v3

The preferred provider contract is:

```python
provider(
    native, q, k, v,
    *, kind, scale, square_aligned=False, sink_rows=0,
)
```

For grouped local attention:

- `q` contains only the VDN query rows actually requested by the trained local operator;
- `k/v` remain VDN's exact already-restricted K/V domain and ordering;
- `sink_rows` identifies the leading global/prefix K/V rows;
- VDN retains ownership of window membership, global/anchor operations, learned softmax gate, learned linear complement and output projection.

Dispatch remains `v3 -> v2 -> v1 -> native`. The v2-only `square_q` / `query_positions` payload is now lazy and is built only when an actual v2-only provider is selected. Masked Flex remains VDN-native.

`vdn_attention_preprocess_v1` also runs once on the complete post-RoPE packed Q/K/V tensors before grouped row gathering, allowing composable transforms such as Untwisting RoPE to retain original packed-row semantics without being applied twice.

## Production result with Sol-H3 v0.1.0

The released [ComfyUI-Sol-H3 v0.1.0](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.0) production stack on RTX PRO 6000 Blackwell confirms API v3 is active and direct rectangular VDN execution is 1:1 in requested/kernel Q rows:

| Stage | Rectangular SOL calls | Requested Q rows | Kernel Q rows | Square expansion |
|---|---:|---:|---:|---:|
| Native low | 1,584 | 3,744,000 | 3,744,000 | 0 |
| Native high | 1,056 | 4,972,800 | 4,972,800 | 0 |
| Later native high | 1,248 | 5,967,360 | 5,967,360 | 0 |

The old v2 compatibility bridge expanded Q kernel work by roughly **4.4–5.4x** in affected stages. v1.5.3 removes that provider-side expansion without broadening VDN's attention domain or changing its trained gating/output semantics.

Flow's separate external mixed-grid API-2 route remained direct at `144` SOL calls and `6,270,480` requested/kernel Q rows 1:1. The final Spectrum schedule was:

```text
sampler_logical_calls       18
transformer_actual_nfe      14
spectrum_forecast_calls      4

low:    8 actual / 2 forecast
high:   4 actual / 2 forecast
probe:  2 actual / 0 forecast
```

Those counters validate composition across VDN, Sol-H3, Spectrum, Untwist, Diff-Aid and Flow. They do not transfer ownership of Flow's external route or Spectrum's forecast policy to VDN.

## INT8 ConvRot stage setup

The documentation now leads users to the existing pre-quantized INT8 ConvRot VDN stage while retaining the in-repository converter as the reproducible path for building a supported stage directly from the official OpenVDN checkpoint. The recommended Ref-Delta INT8 ConvRot base and auto-discovery behavior are clarified.

This is documentation/package guidance around functionality already present in the repository; it does not introduce a second incompatible VDN model format.

## Pruned / curve AdaLN affine setup

Standard Comfy-Org `*_pruned_*` MiniMax-H3 checkpoints do not contain `adaln_basis` / `adaln_mean`, so the extractor no longer instructs users to derive those auxiliaries from files that cannot provide them. The docs point to the matching published FL2VA/Ref2VA affine sidecars, and the extraction tool now fails with actionable guidance when those tensors are absent.

## Relationship to PR #8

The separate PR #8 audio-fidelity experiment remains unreleased and semantically unvalidated. v1.5.3 does **not** include or depend on that experiment. The softmax-provider work lives outside `vdn_h3/hybrid.py`, so #8 may still be stacked independently for research without being a prerequisite for this release.

## Validation

- Provider tests cover v1/v2/v3 dispatch, restricted-domain equivalence, lazy v2 construction, direct rectangular v3 routing and full-domain preprocessing order.
- The normal VDN CI lanes retain pinned Comfy/OpenVDN oracle coverage, current-Comfy smoke and legacy workflow migration.
- Sol-H3 v0.1.0's pinned native-interop suite exercised this exact VDN provider contract together with Untwist, Spectrum, Diff-Aid and Flow.
- Real SM120 production execution established the row-accounting and 18/14/4 schedule above with zero VDN square-Q expansion.

v1.5.2's non-mutating bypass ownership, pruned-AdaLN runtime residual math, `cudaMallocAsync` prefetch lifetime fix, exact retained-window SDPA semantics and model-aware metadata remain unchanged.

---

Previous release notes through v1.5.2 are preserved verbatim in `docs/RELEASE_NOTES_v1.5.2_AND_EARLIER.md`.

## Target-query conditioning measure

Adds Flow's opt-in `target_query_sink_measure` diagnostic to partitioned
softmax. Target-grid local and row-anchor queries bias conditioning keys
alongside target-grid video keys. Source-grid and global queries and learned
linear measure policy are preserved. Requires paired Sol sink-measure API 1.
Normal is unchanged. Mixed-grid row anchors require separate dense subcalls;
there are no additional model evaluations. Rendered quality and GPU timing
are not established by CPU arithmetic and lifetime tests.
