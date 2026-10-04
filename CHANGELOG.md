# Changelog

## Unreleased

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
