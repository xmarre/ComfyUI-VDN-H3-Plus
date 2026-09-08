# Optional attention subcall providers

Companion: [Sol-H3 PR #1](https://github.com/xmarre/ComfyUI-Sol-H3/pull/1).

VDN keeps ownership of its trained window/global/anchor geometry, learned softmax gate, output projection and learned linear complement. External providers can only operate on domains VDN has already selected.

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

`kind` distinguishes `local`, `global`, `anchor` and `flex_masked`. Masked Flex retains its native mask semantics unless a provider explicitly supports them. The current Sol-H3 companion uses v2 only for representable local grouped calls; global/anchor/Flex retain native execution.

## Full-domain QKV preprocessing

`transformer_options["vdn_attention_preprocess_v1"]` is an optional shape-preserving preprocessing callable:

```python
preprocess(q, k, v, *, heads, transformer_options)
```

It runs once on the full post-RoPE VDN tensors before local row gathering. This is required for transforms such as Untwist whose metadata uses original packed-row coordinates; applying those transforms after window gathering would make the coordinates wrong. Shape, dtype and device must remain unchanged.

Generic model-level dense attention overrides still do not automatically leak into VDN's trained local operator. The explicit provider/preprocess contracts define composition instead.

## Validation

The v2 PR suite passes **148 tests** against pinned ComfyUI and the official OpenVDN oracle, plus current-Comfy import smoke tests and legacy workflow migration. Sol-H3's native interoperability suite uses the real Comfy `ModelPatcher` object-patch lifecycle and confirms that a VDN v2 object patch reaches SOL's square-domain provider with CPU SDPA substituted for the unavailable CUDA kernel.

This establishes structural equivalence of the square-domain mapping, not GPU performance or decoded-media quality. The correctly ordered production RTX PRO 6000 run that motivated v2 used the older contract and executed zero sparse SOL calls; it is failure-reproduction evidence, not a SOL timing result. Square expansion can increase query work, so Sol-H3 reports requested versus kernel rows for the fresh post-v2 GPU validation.