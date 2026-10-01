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

Admission protects loaded patchers that Core identifies as clones of the
prepared diffusion model, its prepared additional models and their weight-patch
backing models on the sampling device. Dead weak references and other-device
entries are excluded. Unrelated models remain eligible under memory pressure;
Core may partially unload them. The keep list is preparation-local and retains
no model or tensor across calls. CPU execution and execution without retained
buffers use the original delegate path.

## Receipts

The `sampling preparation` receipt reports `prepare_elapsed_ms` and `success`
for delegated Core/wrapper preparation. Successful retained CUDA preparation
then emits `sampling admission policy=bounded_headroom_v1`, including the stage,
requested and available MiB, whether eviction was requested, protected counts,
full-unload names/count and `eviction_elapsed_ms`.

The two host wall-time intervals do not overlap and add no explicit CUDA
synchronization. They exclude earlier text encoding and later model evaluation.
`unloaded=0` can describe a partial unload; use `eviction_requested` and the free
memory fields to interpret it. A zero protected-H3 count is not itself an error.

The memory target cannot guarantee that arbitrary external kernels or concurrent
device users fit. Model residency can still change through Core's own memory
policy or other nodes. This admission policy changes no attention, denoising,
transfer, history or VDN buffer-release arithmetic.

Structural tests cover sufficient headroom, finite pressure, actual Core partial
eviction, changed patch UUIDs, required additional/backing models, device/dead
reference exclusion, original argument/result identity and failure propagation.
Runtime timings and output comparison are required to qualify a speed or
generated-quality change.
