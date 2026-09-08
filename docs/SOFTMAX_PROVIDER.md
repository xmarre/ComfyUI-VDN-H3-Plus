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

Normal MiniMax-H3 VDN windows include packed non-video/global rows in KV. Their requested video Q rows are therefore usually rectangular against the restricted KV domain, so v1 cannot feed a square-QKV-only kernel.

## v2 square-domain contract

`transformer_options["vdn_softmax_provider_v2"]` receives the same restricted operation plus an optional real square query domain:

```python
provider(
    native, q, k, v,
    *, kind, scale, square_aligned=False,
    square_q=None, query_positions=None, sink_rows=0,
)
```

For a grouped local call, VDN constructs K/V in its existing order `[global_rows, permitted_window_rows]`. `square_q`, when supplied, contains the real Q rows from those exact same packed indices and in the same order. `query_positions` maps the original requested Q rows into that square domain. A provider may evaluate the extra query rows only to satisfy its kernel shape contract and then return `square_output.index_select(0, query_positions)`.

This does **not** broaden VDN attention. K/V remain the exact VDN-restricted local domain. Global and anchor operations remain separate. The extra square-domain query results are disposable implementation work and do not enter VDN state or the learned linear complement.

The current Sol-H3 companion uses v2 for representable grouped-local calls. Global and anchor operations retain native execution. Masked Flex remains VDN-native; if Flex falls back to grouped, that grouped execution can consume v2.

## Full-domain QKV preprocessing

`transformer_options["vdn_attention_preprocess_v1"]` is an optional shape-preserving preprocessing callable:

```python
preprocess(q, k, v, *, heads, transformer_options)
```

For grouped execution it runs once on the complete post-RoPE packed tensors inside `window_softmax_grouped_runtime`, before VDN gathers local row domains. This is required for transforms such as Untwist whose metadata uses original packed-row coordinates. Shape, dtype and device must remain unchanged.

Generic model-level dense attention overrides still do not automatically leak into VDN's trained local operator. The explicit provider/preprocess contracts define composition instead.

## Overlay compatibility

PR #11 is designed to be applied after PR #8. Their runtime changes are separated deliberately: #8 keeps its hybrid-forward/audio/training behavior, while #11 adds the provider contract in `retained.py` plus the new provider module. The two overlays no longer both edit `vdn_h3/hybrid.py`.

## Validation

The provider suite checks restricted-domain equivalence, v1/v2 dispatch, square-domain mapping, and full-domain preprocessing before grouped row gathering. The normal VDN CI lanes cover the pinned Comfy/OpenVDN oracle, current-Comfy smoke and legacy workflow migration.

This establishes structural equivalence of the provider mapping, not GPU performance or decoded-media quality. The correctly ordered production RTX PRO 6000 run that motivated v2 used the older contract and executed zero sparse SOL calls; it is failure-reproduction evidence, not a SOL timing result. Square expansion can increase query work, so Sol-H3 reports requested versus kernel rows for the fresh post-v2 GPU validation.
