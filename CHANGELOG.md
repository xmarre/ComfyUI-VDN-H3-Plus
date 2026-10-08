# Changelog

## Unreleased

- The AIMDO model-compiler guard now takes effect. It was a DIFFUSION_MODEL
  wrapper, but the native MiniMax-H3 forward decides whether to record a malloc
  graph, and opens it, before its DIFFUSION_MODEL wrappers run, and nothing in
  the forward reads the setting again. The guard therefore never prevented
  recording. It is now an APPLY_MODEL wrapper registered by Apply on the
  patched model, so the compiler is off when the forward makes that decision
  and is restored after each model evaluation. On builds with dynamic VRAM and
  the compiler enabled, VDN forwards now run without a recorded allocation
  graph. A user-supplied `--disable-comfy-compiler` is never changed.
- The import-time wrapper around `vdn_h3.hybrid.make_layout_wrapper` is removed.

## v1.5.8 — 2026-10-06

- The partitioned exact-prefix contract may declare `native_carrier_grid="target"`
  with `native_carrier_rows_per_frame`. The native (pre-partition) sequence is
  then the uniform target grid while the partition presented to VDN keeps its
  `[target-grid head | source-grid tail]` layout. VDN validates its native layout
  against the declared carrier. Contracts without the field keep the reduced-grid
  carrier, and their contracts and semantic digests are unchanged. The bridged
  forward advertises the accepted carriers in
  `_vdn_partitioned_native_carrier_grids`.
- Optional native carrier row counts require an explicit target carrier and
  integer values in the Flow and external-sequence contracts. Undeclared or
  non-integer counts are rejected before attention execution.

## v1.5.7 and earlier

- Partitioned exact-prefix attention keeps the first generated local-query group
  dense. This preserves one attention operator across the carried/generated
  boundary while later generated groups retain native sparse Sol routing. The
  boundary group uses the existing gathered Q/K/V domain and prefix measure and
  adds no model evaluations.
- Retained sampling prepares required models through Core before reserving
  finite scratch headroom. Sufficient memory preserves unrelated resident
  models; pressure uses Core's eviction policy while protecting prepared models
  and their patch backings. The existing automatic 10 GiB scratch allowance is
  shared with admission and added to Core's sampling estimate.
- Preparation and admission receipts report separate timings, the finite memory
  target, available memory, protected models and full unloads. Partial unloads
  remain under Core's control.
- Admission checks the observed free memory after eviction, retries one remaining
  shortfall after partial recovery, and fails before sampling if the finite
  target remains unmet. Receipts expose the pass count and target result.
- Partitioned attention releases QKV storage and softmax temporaries after their
  last use, before weight prefetch and learned-linear workspace allocation.
- Startup reports the loaded admission module and package paths. Applying VDN
  reports the installed policy, retained-buffer setting and actual fast-kernel
  flag, so an outdated checkout or shared Python import can be identified before
  measuring a complete generation.
