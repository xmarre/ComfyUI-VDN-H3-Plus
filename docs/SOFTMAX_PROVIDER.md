# Optional attention subcall providers

Production companion: [ComfyUI-Sol-H3 v0.1.0](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.0).

VDN keeps ownership of its trained window/global/anchor geometry, learned softmax gate, output projection and learned linear complement. External providers can only operate on domains VDN has already selected.

The provider contract is implemented in grouped retained attention rather than `vdn_h3/hybrid.py`. It therefore remains independent of the separate experimental audio-fidelity work in PR #8; #8 is not required for this provider contract or for v1.5.3.

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

`transformer_options["vdn_softmax_provider_v3"]` is the preferred contract for providers that accept rectangular attention:

```python
provider(
    native, q, k, v,
    *, kind, scale, square_aligned=False, sink_rows=0,
)
```

For local calls, `q` contains only the requested VDN query rows while `k/v` contain the exact already-restricted VDN K/V domain. `sink_rows` is the leading global/prefix K/V row count. No `square_q` or `query_positions` payload is constructed.

Dispatch order is v3, then v2, then v1. Sol-H3 v0.1.0 publishes v3, so production rectangular SOL no longer pays the v2 square-Q gather/allocation. v2 construction is lazy and occurs only when an actual v2-only provider is installed.

Global and anchor operations remain separate. Masked Flex remains VDN-native; if Flex falls back to grouped, that grouped execution can consume the explicit provider contract.

## Full-domain QKV preprocessing

`transformer_options["vdn_attention_preprocess_v1"]` is an optional shape-preserving preprocessing callable:

```python
preprocess(q, k, v, *, heads, transformer_options)
```

For grouped execution it runs once on the complete post-RoPE packed tensors inside `window_softmax_grouped_runtime`, before VDN gathers local row domains. This is required for transforms such as Untwist whose metadata uses original packed-row coordinates. Shape, dtype and device must remain unchanged.

Generic model-level dense attention overrides still do not automatically leak into VDN's trained local operator. The explicit provider/preprocess contracts define composition instead.

## Optional stacking with PR #8

The unreleased PR #8 audio-fidelity experiment owns separate hybrid/audio/training behavior. This provider work does not edit `vdn_h3/hybrid.py` and can be stacked with #8 for experimentation, but v1.5.3 does not depend on #8 and does not include its unvalidated semantic experiment.

## Validation and production evidence

The provider suite checks v1/v2/v3 dispatch, restricted-domain equivalence, lazy v2 square mapping, direct rectangular v3 routing, and full-domain preprocessing before grouped row gathering. The normal VDN CI lanes cover the pinned Comfy/OpenVDN oracle, current-Comfy smoke and legacy workflow migration.

The released Sol-H3 v0.1.0 production stack on RTX PRO 6000 Blackwell confirms direct v3 routing with no VDN square-Q expansion:

| Stage | Rectangular SOL calls | Requested Q rows | Kernel Q rows | Square expansion |
|---|---:|---:|---:|---:|
| Native low | 1,584 | 3,744,000 | 3,744,000 | 0 |
| Native high | 1,056 | 4,972,800 | 4,972,800 | 0 |
| Later native high | 1,248 | 5,967,360 | 5,967,360 | 0 |

The historical v2 compatibility bridge expanded Q kernel work by roughly 4.4–5.4x in affected stages. v3 removes that expansion without changing VDN's K/V membership, learned gate, linear complement, output projection or global/anchor ownership.

Flow's separate external mixed-grid API-2 route also remained direct at 144 SOL calls / 6,270,480 requested and kernel Q rows 1:1. Spectrum retained 18 logical calls / 14 actual transformer NFEs / 4 forecasts in the final stack. These results validate provider composition; they are not a claim that VDN itself owns Flow's external route or Spectrum's forecast policy.
