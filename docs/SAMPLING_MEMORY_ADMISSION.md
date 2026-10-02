# Retained sampling memory admission

Retained VDN preparation reserves finite GPU scratch headroom. An unconditional
all-model purge would discard reusable text encoders and VAEs even when their
weights and the current sampling workspace fit together. Changing conditioning
would then require those models to be transferred again.

Core preparation runs first, exactly once, with the original model, shape,
conditioning, options and load/offload flags. Core controls clone switching,
weight-patch reconciliation, multi-device preparation and the required
conditioning, nested and hook-provided models. Preparation failures propagate
unchanged and do not enter VDN admission. If admission fails after preparation,
VDN cleans the prepared additional-model execution state because Core has not
yet handed that set to the sampler lifetime; cleanup failure does not replace
the original admission exception.

After successful preparation, VDN requests free memory equal to:

```
max(Core minimum inference memory,
    Core sampling estimate + conditioning inference memory + Core reserved memory)
+ 10 GiB retained sampling headroom
```

The 10 GiB allowance is the existing automatic scratch-retention heuristic,
shared with that policy. It is added to Core's geometry-dependent estimate;
it is not a strict peak-memory bound. Weight loading has already completed, so
resident model bytes are not added again to the free-memory target.

If available memory meets that target, VDN does not call `free_memory`.
This matters because Core's disable-smart-memory option can ignore finite
targets and unload all eligible models. Under pressure, VDN delegates the finite
request to Core without changing that user-selected policy.

Admission checks the observed free memory after eviction. If a partial unload
recovers memory but leaves a shortfall, one further finite request reconciles
that shortfall. Lack of progress stops immediately; at most two eviction passes
run. If the observed memory still misses the target, admission raises an
out-of-memory error before model evaluation and cleans the prepared additional
models. The requested target and protected set stay unchanged across both passes.

Admission protects loaded patchers that Core identifies as clones of the
prepared diffusion model, its prepared additional models and their weight-patch
backing models on the sampling device. Dead weak references and other-device
entries are excluded. Unrelated models remain eligible under memory pressure;
Core may partially unload them. The keep list is preparation-local and retains
no model or tensor across calls. CPU execution and execution without retained
buffers use the original delegate path.

## Receipts

At custom-node import, `sampling admission source policy=bounded_headroom_v2`
reports the loaded `vdn_h3.hybrid` module path and the importing package path.
These are loaded-module paths, not a Git revision or a hash of files on disk.
If another package imported an older `vdn_h3` first, `policy=unversioned` and
the actual module path expose that shared-import case without assuming its cause.
Applying VDN emits `sampling admission installed`, including the policy,
retained-buffer setting and the branch's actual `fast_kernels` flag. Import
provenance alone does not establish that the current MODEL uses this wrapper;
the installation and sampling receipts provide that separate evidence.

The `sampling preparation` receipt reports `prepare_elapsed_ms` and `success`
for delegated Core/wrapper preparation. Successful retained CUDA preparation
then emits `sampling admission policy=bounded_headroom_v2`, including the stage,
requested and available MiB, whether eviction was requested, protected counts,
full-unload names/count, `eviction_passes`, `target_met` and `eviction_elapsed_ms`.

The two host wall-time intervals do not overlap and add no explicit CUDA
synchronization. They exclude earlier text encoding and later model evaluation.
`unloaded=0` can describe a partial unload; use `eviction_requested` and the free
memory fields to interpret it. A zero protected-H3 count is not itself an error.

The memory target cannot guarantee that arbitrary external kernels or concurrent
device users fit. Model residency can still change through Core's own memory
policy or other nodes. This admission policy changes no attention, denoising,
transfer, history or VDN buffer-release arithmetic.

Structural tests cover sufficient headroom, finite pressure, actual Core partial
eviction, bounded shortfall reconciliation and cleanup on an unmet target,
changed patch UUIDs, required additional/backing models, device/dead
reference exclusion, original argument/result identity and failure propagation.
Runtime timings and output comparison are required to qualify a speed or
generated-quality change.

## Adapter activation workspace

In inference, a VDN low-rank post-forward hook adds the module output into its
newly allocated projection delta when both are ordinary tensors with identical
shape, dtype and strides. This avoids a third output-sized allocation while
preserving the module output, including aliases retained by earlier hooks or
the caller. The factors and bias cached at injection remain read-only.

Gradient-enabled calls, tensor subclasses, differing layouts, dtype promotion,
broadcast outputs and bias-only hooks retain out-of-place addition. Bias-only
hooks cannot reuse the delta because that tensor belongs to the factor cache.
Projection math, compiled factor scaling, strengths and hook order are unchanged.
This local storage correction does not make admission a worst-case peak bound.

## Patcher application

For this stacked draft, retain overlays #33 -> #34 -> #35 -> #36 on the existing
tracked VDN base. Use the VDN repository card's **Update** action to rebuild the
enabled overlay stack, then restart ComfyUI. Refreshing installation details,
previewing an update or refreshing checkpoint history does not apply the stack.
The current Patcher source fetches each enabled PR head, validates its captured
declared-base snapshot and applies that PR's delta in the stored order.

The startup source receipt can be checked before queuing a generation. If it is
missing or unversioned, or points at another package, keep the Patcher Update
operation log and startup log for source diagnosis. A full sampling run is not
needed merely to establish import provenance. Successful retained CUDA sampling
must still publish `sampling admission policy=bounded_headroom_v2`.
