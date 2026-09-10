# ComfyUI-VDN-H3 v1.5.2

v1.5.2 is the production-correctness repair for VDN bypass on the stacked quantized MiniMax-H3 workflow. It replaces the remaining VDN-owned mutable/weight-materializing bypass paths with non-mutating runtime residuals, fixes cross-stream prefetch lifetime under `cudaMallocAsync`, and restores the released exact-SDPA semantics for retained VDN local-window attention.

## Runtime-bypass ownership

- Ordinary VDN LoRA residuals use PyTorch forward **post-hooks**.
- VDN does **not** replace, splice, save, or restore `module.forward`.
- VDN does **not** use `ModelPatcher.weight_function` / `add_weight_wrapper`.
- One post-hook is registered per affected module, with all VDN terms fused into one exact low-rank residual for that module.
- Registration handles are generation-owned across clone-shared models: a newer VDN clone replaces the old registration, while stale ejects cannot remove the newer generation.
- Independently managed Comfy `BypassForwardHook` providers remain outside VDN's ownership; VDN never enters their mutable forward chain.
- Fused INT8 `mlp.fc2` targets retain native Comfy patch ownership because the H3 fused path bypasses `module.forward`.

This supersedes both failed v1.5.x bypass topologies: the v1.5.0 runtime weight-wrapper path and the v1.5.1 VDN-owned mutable `BypassForwardHook` chain.

## Pruned / curve AdaLN bypass

The exact pruning affine remains:

```text
dense(t) ≈ mean + curve(t) @ basis
A_pruned   = A @ basis.T
bias_delta = B @ (A @ mean)
```

In `bypass`, the projected low-rank curve-coordinate residual and required constant bias are added after `adaln_proj.linear` without mutating the pruned base AdaLN weight or bias. In `merge`, the exact projected weight+bias terms continue to use ordinary native Comfy patches.

This removes the remaining bypass-mode base-weight materialization that was still present in the first v1.5.2 candidate.

## Adapter staging

Runtime VDN factors are staged synchronously onto the intended compute device when the VDN `PatcherInjection` is injected, before the first H3 module forward. Ordinary VDN post-hooks therefore do not perform their normal adapter H2D setup during the first model call. An unexpected device/dtype change retains a synchronous correctness fallback.

## `cudaMallocAsync` prefetch lifetime

VDN's one-block branch prefetch allocates/copies branch weights on a producer CUDA stream and consumes them on the model stream. v1.5.2 records the actual consumer stream for every returned prefetched tensor and for quantized backing/scale tensors under both the native allocator and `cudaMallocAsync`.

This avoids premature storage reuse across the producer/consumer stream boundary while retaining the existing bounded one-block prefetch architecture.

## Exact VDN local-window attention

A separate reconciliation audit found that retained grouped-window execution had started forwarding model-level `transformer_options` into VDN's local-window SDPA helper. That was semantic drift: the released VDN local-window operator is exact SDPA, while Sage/Kitchen/model-level attention overrides belong to the native/base attention path.

v1.5.2 therefore deliberately ignores model-level attention overrides for retained VDN local windows and passes `None` to the local `_sdpa` calls. Regression coverage verifies both no override leakage and numerical parity with the exact grouped-window reference path.

## Model-aware runtime metadata

Read-only, zero-copy runtime adapter descriptors remain published for model-aware consumers such as Spectrum. They expose the effective runtime LoRA and constant-offset perturbations without taking execution ownership, changing adapter math, or mutating `module.forward`.

## Validation

PR #4's final pre-release head passed CI run **232**, including current-Comfy smoke, pinned Comfy/OpenVDN oracle coverage, bypass lifecycle/math regressions, projected curve-AdaLN coverage, allocator/prefetch lifetime checks, runtime introspection, and exact retained-window attention tests.

The corrected production RTX PRO 6000 workflow was then re-run on current ComfyUI with the intended Untwist RoPE configuration (`high_scale_end=1.00`) and completed successfully. The release acceptance criteria are satisfied:

- no CUDA illegal-memory-access regression;
- normal artifact-free decoded video quality in the production stack.

No speed or VRAM claim is inferred from the CPU/oracle suite; historical benchmark numbers remain historical until separately re-measured.

# ComfyUI-VDN-H3 v1.5.1

v1.5.1 is a hotfix for a production regression introduced by v1.5.0's `lora_mode=bypass` implementation.

## Fixed: stacked runtime adapters on quantized H3

v1.5.0 moved VDN bypass adapters from Comfy's activation-side `BypassForwardHook` mechanism to `ModelPatcher.add_weight_wrapper()` / `weight_function`. On quantized MiniMax-H3 layers, a weight function forces Comfy through a copied/dequantized compute-weight path.

In the production RTX PRO 6000 workflow using:

- an INT8 ConvRot MiniMax-H3 base;
- VDN `lora_mode=bypass`;
- an independent runtime-bypass DoRA/LoRA provider on the same H3 modules;
- `cudaMallocAsync`;
- Continuum + Flow Mixed-Grid + Spectrum/SA-PECE;

the first actual H3 evaluation hard-aborted with `CUDA error: an illegal memory access was encountered`. The fatal Python stack was inside the external Comfy `BypassForwardHook`/LoRA `F.linear` chain. CUDA errors are asynchronous, so that stack alone does not identify the exact originating kernel.

v1.5.1 restored the older VDN `BypassForwardHook` architecture, including stack-safe linked-list insertion/removal. A later production run showed that this was not sufficient: the same stacked workflow still aborted. v1.5.1 is therefore superseded by the v1.5.2 work above.

## Pruned / curve AdaLN behavior

The v1.5 exact pruning-affine work was retained. Full-width released AdaLN LoRAs were projected through the resolved `adaln_basis` + `adaln_mean` pair. v1.5.1 materialized those projected curve weight/bias terms as ordinary Comfy patches in both adapter modes; the revised v1.5.2 bypass path no longer does so.

---

# ComfyUI-VDN-H3 v1.5.0

v1.5.0 was the correctness/lifecycle and Flow-interoperability release of the xmarre fork. It added the mixed-grid API used by MiniMax-H3 Flow-Aligned Regenerate, current pruned/INT8 MiniMax-H3 support, exact curve-AdaLN projection, branch/runtime lifecycle hardening, upstream v1.4.3 reconciliation, and backwards-compatible Advanced-node workflow migration.

Its `lora_mode=bypass` weight-wrapper implementation is superseded because of the stacked runtime-adapter regression described above. `merge` behavior was not implicated.
