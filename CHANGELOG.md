# Changelog

## Unreleased

- Retained sampling prepares required models through Core before reserving
  finite scratch headroom. Sufficient memory preserves unrelated resident
  models; pressure uses Core's eviction policy while protecting prepared models
  and their patch backings. The existing automatic 10 GiB scratch allowance is
  shared with admission and added to Core's sampling estimate.
- Preparation and admission receipts report separate timings, the finite memory
  target, available memory, protected models and full unloads. Partial unloads
  remain under Core's control.
- Startup reports the loaded admission module and package paths. Applying VDN
  reports the installed policy, retained-buffer setting and actual fast-kernel
  flag, so an outdated checkout or shared Python import can be identified before
  measuring a complete generation.
