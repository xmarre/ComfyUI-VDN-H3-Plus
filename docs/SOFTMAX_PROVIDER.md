# Optional attention subcall providers

Companion: [Sol-H3 PR #1](https://github.com/xmarre/ComfyUI-Sol-H3/pull/1).

VDN keeps ownership of its trained window/global/anchor geometry, learned softmax gate, output projection and learned linear complement. External providers can only operate on domains VDN has already selected.

This overlay is intentionally implemented in the grouped retained-attention layer rather than by replacing `vdn_h3/hybrid.py`. That keeps the provider contract composable with the earlier audio-fidelity overlay (#8), which owns substantial hybrid-forward behavior.

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

Dispatch order is v3, then v2, then v1. The current Sol-H3 companion publishes v3, so production rectangular SOL no longer pays the v2 square-Q gather/allocation. v2 construction is now lazy and occurs only when an actual v2-only provider is installed.

Global and anchor operations remain separate. Masked Flex remains VDN-native; if Flex falls back to grouped, that grouped execution can consume the explicit provider contract.

## Full-domain QKV preprocessing

`transformer_options["vdn_attention_preprocess_v1"]` is an optional shape-preserving preprocessing callable:

```python
preprocess(q, k, v, *, heads, transformer_options)
```

For grouped execution it runs once on the complete post-RoPE packed tensors inside `window_softmax_grouped_runtime`, before VDN gathers local row domains. This is required for transforms such as Untwist whose metadata uses original packed-row coordinates. Shape, dtype and device must remain unchanged.

Generic model-level dense attention overrides still do not automatically leak into VDN's trained local operator. The explicit provider/preprocess contracts define composition instead.

## Overlay compatibility

PR #11 is designed to be applied after PR #8. Their runtime changes are separated deliberately: #8 keeps its hybrid-forward/audio/training behavior, while #11 adds the provider contract in `retained.py` plus the provider module. The two overlays do not both edit `vdn_h3/hybrid.py`.

## Validation and production evidence

The provider suite checks v1/v2/v3 dispatch, restricted-domain equivalence, lazy v2 square mapping, direct rectangular v3 routing, and full-domain preprocessing before grouped row gathering. The normal VDN CI lanes cover the pinned Comfy/OpenVDN oracle, current-Comfy smoke and legacy workflow migration.

The companion Sol-H3 production run on RTX PRO 6000 established rectangular v2-era kernel execution with requested Q rows matching actual kernel Q rows and zero Sol-H3 square-expanded calls. v3 removes the remaining upstream compatibility `square_q` construction for that same rectangular provider path; it does not alter VDN's K/V membership, learned gate, linear complement, or output projection.
