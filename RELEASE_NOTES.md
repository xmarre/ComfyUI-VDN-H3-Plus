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
