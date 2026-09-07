# ComfyUI-VDN-H3-Plus

> **Plus fork:** This is the `xmarre` maintained Plus fork of [Saganaki22/ComfyUI-VDN-H3](https://github.com/Saganaki22/ComfyUI-VDN-H3). It preserves the upstream project's foundation while carrying additional features, integrations, fixes, and behavior that may intentionally diverge from upstream.

A ComfyUI port of the released [OpenVDN VDN-H3](https://github.com/OpenVDN/vdn-minimax-h3) hybrid-attention architecture for ComfyUI's native MiniMax-H3 model.

This xmarre fork keeps the released VDN checkpoint/math contract while adding current pruned/INT8 H3 support, stricter Comfy lifecycle handling, and the external mixed-grid sequence contract used by [MiniMax-H3 Flow-Aligned Regenerate](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate).

> The VDN model weights are separate from this repository and retain their upstream license. See [NOTICE](NOTICE) for implementation provenance and attribution.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/ComfyUI-VDN-H3-Plus.git ComfyUI-VDN-H3
```

Restart ComfyUI.

Download an official VDN stage under `ComfyUI/models/vdn/` while preserving its directory structure, for example:

```bash
hf download OpenVDN/vdn-minimax-h3 \
  --include "stage-dmd-step-250/*" \
  --local-dir <ComfyUI>/models/vdn
```

Official stages include:

- `stage-dmd-step-250` — released 8-step DMD/Turbo stage;
- `stage-b-step-2000` — released Stage-B/default stage.

## Nodes

### Apply VDN-H3

| Input | Meaning |
|---|---|
| `vdn_checkpoint` | Stage directory under `models/vdn/` |
| `apply_turbo_adapter` | Apply the released Turbo/DMD adapter when the stage provides it |
| `strength` | Adapter strength; `1.0` is the released setting |
| `lora_mode` | `merge` or isolated runtime `bypass` |
| `branch_weights` | `auto`, `stream`, or `resident` |
| `retain_buffers` | `auto`, `on`, or `off` |
| `attention_backend` | `grouped` or opt-in `flex` |
| `verbose` | Additional runtime logging |

### Apply VDN-H3 Advanced

Adds independent Stage-B/Turbo strengths, optional compiled helpers, and explicit architecture ablations.

`architecture_mode=checkpoint` is the default for newly created nodes and uses `model_spec.json` exactly. Select `architecture_mode=override` only when intentionally changing `window_radius`, `window_chunk`, `anchor_frames`, `text_state`, or `linear_branch`.

## Adapter modes

### `lora_mode=merge`

Uses ordinary Comfy `ModelPatcher.add_patches()` weight ownership. This is the conservative reference path for matched output validation.

### `lora_mode=bypass`

v1.5.2 keeps ordinary VDN LoRA terms off both Comfy weight wrappers and mutable `module.forward` bypass chains. Each affected module receives one VDN-owned PyTorch **forward post-hook**. The normal module forward executes first, including any independently managed provider that chooses to wrap it, and VDN then adds its exact low-rank residual to the returned tensor.

The runtime contract is:

- VDN never replaces, splices, saves, or restores `module.forward`;
- VDN does not install `ModelPatcher.add_weight_wrapper()` / `weight_function` callbacks;
- all active VDN terms for one module are combined into one exact low-rank residual;
- adapter factors are staged onto the intended compute device when the VDN `PatcherInjection` is injected, before the first H3 forward;
- post-hook handles are generation-owned on the clone-shared inner model;
- applying a newer VDN clone replaces the previous VDN registration instead of accumulating deltas;
- ejecting an older/stale clone cannot remove the newer registration;
- another provider using Comfy `BypassForwardHook` remains outside VDN's ownership and its forward chain is never rewritten by VDN;
- fused INT8 `mlp.fc2` targets whose H3 fast path bypasses `module.forward` remain ordinary Comfy weight patches.

## Pruned / curve MiniMax-H3 bases

Supported pruned H3 checkpoints represent the original dense AdaLN timestep field approximately as:

```text
dense(t) ≈ mean + curve(t) @ basis
```

Released VDN Turbo adapters contain full-width AdaLN LoRAs. For an update `B @ A`, this fork projects it once into native curve coordinates:

```text
A_pruned   = A @ basis.T
bias_delta = B @ (A @ mean)
```

Both terms are required. VDN must resolve the matching `adaln_basis` + `adaln_mean` pair and fails closed rather than guessing or dropping incompatible AdaLN updates.

Ownership depends on adapter mode:

- `merge`: projected curve weight and constant bias use ordinary Comfy weight/bias patches;
- `bypass`: the exact projected low-rank residual plus constant bias are added by a post-forward hook on `adaln_proj.linear`; the pruned base weight and bias remain untouched.

This distinction matters on quantized/pruned H3. Earlier v1.5.x candidates that materialized the projected AdaLN terms in bypass mode were part of the remaining VDN-specific execution preceding the production CUDA failure boundary. The bypass path now avoids that base-weight mutation entirely.

If a matching BF16 source checkpoint remains beside an INT8 derivative, VDN can read only the small affine tensors from it. Otherwise use:

```bash
python tools/extract_h3_adaln_affine.py \
  <matching-pruned-bf16.safetensors> \
  <ComfyUI>/models/vdn/<stage>/adaln_affine.safetensors
```

## Branch weights and retained buffers

`branch_weights` controls the VDN linear branch and is independent from `lora_mode`.

- `auto` reserves still-unloaded H3 base-model bytes before deciding what VDN may keep resident. It uses resident BF16 branch weights when headroom is sufficient; otherwise it streams and prefers the native INT8 ConvRot branch when available.
- `stream` resolves one block at a time and can use one-block lookahead when retained buffers are enabled.
- `resident` registers ordinary BF16 branch weights as a Comfy-managed additional `ModelPatcher`.

`retain_buffers=on` reuses execution-owned scan/window/activation scratch. The retained pool belongs to one VDN state and is leased for one diffusion-model execution; nested/concurrent runs that cannot acquire it use isolated transient scratch.

### CUDA stream ownership

One-block branch prefetch copies weights on a dedicated CUDA producer stream and consumes them on the model stream. VDN records that consumer stream for every prefetched tensor and for the backing tensors of quantized branch weights under **both** the native allocator and `cudaMallocAsync`.

This intentionally differs from a blanket “skip `record_stream` under `cudaMallocAsync`” rule. PyTorch's async allocator still tracks non-creation usage streams before `cudaFreeAsync`; only redundant same-stream recording is unnecessary.

## Flow-Aligned Regenerate interoperability

VDN supports the external-sequence contracts used by MiniMax-H3 Flow-Aligned Regenerate:

- API 1 target-sparse compatibility;
- API 2 `mixed_grid_low_suffix` for a target-grid protected prefix plus a genuine low-grid generated suffix.

During an API-2 mixed sequence, VDN keeps the learned dense softmax gate active and disables only geometry-dependent local-window/linear-complement work that cannot be interpreted on the mixed lattice. The fresh target-grid stage returns to normal VDN execution automatically.

See [docs/MIXED_SEQUENCE_API.md](docs/MIXED_SEQUENCE_API.md) for the exact fail-closed contract.

## Legacy Advanced-node workflow compatibility

The original Advanced node had 14 positional widgets. v1.5 added `retain_buffers` and `architecture_mode`, which would shift values in workflows saved before ComfyUI emitted `widgets_values_named`.

The included frontend migration:

- publishes the original 14 names through `fallbackWidgetsValuesNames`;
- restores old positional values by name;
- inserts `retain_buffers="auto"`;
- inserts `architecture_mode="override"` for old workflows because those architecture fields historically applied unconditionally;
- preserves the short-lived 16-value positional layout without changing its semantics;
- leaves workflows that already contain `widgets_values_named` untouched.

Newly created nodes still default to `architecture_mode="checkpoint"`.

## Compatibility notes

- `grouped` is the portable attention backend and remains the default. `flex` is opt-in and falls back to grouped if unavailable.
- VDN's retained local-window operator uses exact SDPA and deliberately does not inherit model-level `transformer_options` attention overrides such as Sage/Kitchen backends.
- VDN owns the H3 attention object patch. Another extension that tries to own the same `diffusion_model.blocks.*.attn.forward` target is rejected rather than ambiguously stacked.
- Runtime LoRA/DoRA providers using Comfy's ordinary `BypassForwardHook` mechanism may coexist with VDN bypass. VDN does not join or rewrite that provider's forward chain.
- The AIMDO malloc-graph compatibility guard is scoped only around VDN diffusion-model execution and restores the user's compiler setting afterward.
- Historical benchmark numbers in [Benchmarks.md](Benchmarks.md) predate the current lifecycle work and are not assigned to this runtime without matched measurement.

## Validation

v1.5.2 has explicit regression coverage for:

- exact ordinary post-forward LoRA math while `module.forward` remains externally owned;
- exact projected curve-AdaLN bypass math while the base curve weight/bias remain unchanged;
- VDN-first and external-provider-first coexistence with a real Comfy `BypassForwardHook`;
- VDN removal while the external provider remains live;
- clone-shared VDN replacement and stale-clone eject without accumulation;
- repeated pseudo-Continuum injection/ejection;
- injection-time adapter-factor staging;
- zero `weight_wrapper_patches` and zero VDN mutable-forward wrappers in active bypass mode;
- custom/quantized-like modules retaining their native weight path;
- cross-stream prefetch lifetime ownership under native and `cudaMallocAsync` allocators;
- exact retained-window SDPA semantics with no model-level attention-override leakage;
- current ComfyUI import/registration and the existing OpenVDN numerical/oracle suite.

CPU/oracle CI validates numerical and ownership contracts. The complete stacked RTX PRO 6000 workflow was also re-run on current ComfyUI with the corrected Untwist RoPE configuration and passed the release acceptance gate: no CUDA illegal-memory-access regression and normal artifact-free decoded video quality.

## Upstream and licensing

- OpenVDN reference implementation: https://github.com/OpenVDN/vdn-minimax-h3
- Original ComfyUI port: https://github.com/Saganaki22/ComfyUI-VDN-H3
- This maintained fork: https://github.com/xmarre/ComfyUI-VDN-H3-Plus

Repository code is distributed under [Apache-2.0](LICENSE). Third-party model weights and repositories retain their own licenses. See [NOTICE](NOTICE) for attribution details.
