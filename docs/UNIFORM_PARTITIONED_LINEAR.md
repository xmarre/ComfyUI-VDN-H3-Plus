# Uniform partitioned linear readout

Partitioned exact-prefix execution normally supports different spatial grids and
physical measures for individual video frames. When every frame has the same
grid and unit measure, its learned-linear complement reduces to the released
fixed-grid VDN computation. Reuse that computation on an execution-local branch
copy after validating the partitioned geometry, QKV shapes, bounds and measures.

The shortcut preserves frame bounds, text-state injection and anchor trimming.
RuntimeLinearBranch retains its execution-owned scan banks; nested executions
receive separate banks. The shared branch's backend selector is unchanged.
No attention domain, physical grid, sampler step or provider is changed.

The shortcut requires contiguous hidden/raw-QKV tensors, native temporal-carrier
policy and released K/V short-convolution shapes (or no short convolution).
Q convolution, nonunit measures, mixed grids, diagnostic statistics, temporal
suppression, alternate carrier policies and feature witnesses retain the general
path. Equal area with different grid axes does not qualify. Unit measure is an
exact comparison, without a tolerance.

With `fast_kernels` enabled, the reused implementation requests its existing
compiled query activation, state gather and linear epilogue. Compilation failure
retains the existing eager fallback. With the flag disabled, the same fixed-grid
route runs eagerly. Fused GPU kernels can change floating-point rounding; CPU
equivalence does not establish rendered equivalence or GPU speed.

## Receipts

After a successful readout and shape check, Flow's metrics receive:

- `partitioned_vdn_uniform_linear_calls`: successful fixed-grid shortcut calls.
- `partitioned_vdn_uniform_fast_requested_calls`: those calls with fast kernels
  requested; this does not prove that compilation succeeded.

The optional component recorder reports `vdn_linear_native_uniform_host_wall_s`
and the enclosing `vdn_linear_api_host_wall_s`. CUDA diagnostics use the
`vdn_linear_native_uniform` span. These are overlapping intervals. The general
path keeps its existing component spans and witness observations.
