# Optional softmax subcall providers

Companion: [Sol-H3 PR #1](https://github.com/xmarre/ComfyUI-Sol-H3/pull/1).

`transformer_options["vdn_softmax_provider_v1"]` is an optional callable:

```python
provider(native, q, k, v, *, kind, scale, square_aligned=False)
```

It returns `[query_rows, heads, head_dim]` with unchanged dtype/device. `native()` executes the original operation. Model-level optimized-attention overrides are never passed to VDN's trained local SDPA helper.

VDN selects all row domains before dispatch. `kind` distinguishes local, global, anchor and flex_masked operations. `square_aligned` means query and KV refer to the same ordered row domain, not merely equal tensor dimensions. A square-QKV kernel may accelerate eligible local calls; rectangular windows, global/anchor queries and masked Flex must retain their native semantics when unrepresentable. Never replace local KV with unrestricted full-sequence KV.

Softmax gates, output projection, learned linear complement, API-1/API-2 validation and Flex-to-grouped fallback remain VDN-owned. `attention_history_v1(options, packed_layout)` on the patched forward describes the current geometry for an optional forecasting consumer. A missing/stale descriptor returns None and must not authorize forecasting.

CPU suite: 133 passed, 12 skipped. Additional Sol-H3 tests exercise real VDN/H3/Spectrum dispatch with a CPU SOL oracle and the linear branch explicitly disabled. Existing VDN math/oracle tests are separate. No GPU SOL, Flex or media evidence is claimed. Typical local windows include prefix KV and remain rectangular/native; no local speedup is promised.
