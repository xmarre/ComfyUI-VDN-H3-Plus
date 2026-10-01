# Retained sampling memory admission

Retained VDN sampling evicts unrelated GPU models before Core prepares the
diffusion model. This reserves attention/scratch headroom even when Core's
high-VRAM policy would otherwise leave a text encoder or VAE resident.

The incoming `ModelPatcher` can be a fresh clone of an already loaded diffusion
model. Core's `LoadedModel` equality compares patcher object identity, while
`ModelPatcher.is_clone` identifies their shared underlying model. Protecting only
`LoadedModel(incoming)` therefore fails to protect a resident sibling clone and
can force a full unload/reload before sampling starts.

Admission preserves the incoming patcher and loaded clones that Core identifies
as sharing its model. Unrelated models remain eligible for eviction; dead weak
references are excluded. Core still receives the original model, conditioning,
shape, options and load/offload flags and performs its normal clone detach,
weight-patch reconciliation and memory checks. Different patch UUIDs do not
permit bypassing Core preparation. The keep list is local to one preparation
call and retains no model or tensor across calls.

CPU execution and execution without retained buffers use the original delegate
path. Attention, denoising, transfer, sampler histories and VDN buffer-release
policies are unchanged.

The admission log includes `kept_resident_h3` and `eviction_elapsed_ms`. A
separate `sampling preparation` receipt reports `prepare_elapsed_ms` and
`success` for the delegated Core/wrapper preparation. These host wall-time
intervals do not overlap each other and introduce no CUDA synchronization.
They do not include text encoding before sampler admission or model evaluation
after preparation. A failed delegate is logged with `success=False` and its
original exception propagates.

Structural tests use Core's actual patcher/loaded-model classes to distinguish
object identity from clone identity, exercise equal and changed weight patches,
exclude unrelated/dead models, preserve delegate arguments and check failure
propagation. A hardware run is still required to measure saved preparation time
and confirm model outputs; passing these tests does not establish a speedup.
