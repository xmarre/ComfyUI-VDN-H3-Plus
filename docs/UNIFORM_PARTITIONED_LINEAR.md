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

Native runtime and general partitioned execution release normalized K/V and
beta after frame statistics, then the statistics after the scans. Native runtime
also releases query features and gathered state after readout matmul. These
temporaries no longer overlap subsequent gate/output allocations. Retained scan
banks remain owned by their execution lease. Release follows the last tensor use
on the current stream and adds no synchronization or allocator purge.

The shortcut requires contiguous hidden tensors, native temporal-carrier
policy and released K/V short-convolution shapes (or no short convolution).
Raw QKV split views may retain the native projection's shared row stride;
the released feature operators support that layout without changing its values.
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
- `partitioned_vdn_uniform_pre_rope_calls`: uniform inference forwards that
  compute the complement before in-place QK normalization/RoPE.

The optional component recorder reports `vdn_linear_native_uniform_host_wall_s`
and the enclosing `vdn_linear_api_host_wall_s`. CUDA diagnostics use the
`vdn_linear_native_uniform` span. These are overlapping intervals. The general
path keeps its existing component spans and witness observations.

## Uniform projection lifetime

Normal inference on identical physical grids now computes the validated uniform
readout and its output projection before in-place RoPE. The projection is added
after the existing softmax output projection, preserving the arithmetic order of
the residual addition. Raw video Q/K/V and text K/V are consumed directly from the
native projection views; no preservation copies are created or retained. Any
previous activation-preservation scratch is released before this path starts.
Only the projected complement and, when enabled, the softmax gate survive through
attention. Branch weights are fetched once, with streamed lookahead kept after
attention as in native VDN.

This forward-lifetime shortcut requires normal linear mode, the native carrier
policy, released `vdn_solve` with K/V-only or absent short convolution, contiguous
hidden rows and disabled autograd. Mixed grids, diagnostic policies and witness
ownership retain their previous late readout and copy lifetimes. Query grouping,
prefix dense decisions, key measure, descriptors and Sol gates remain unchanged.
CPU comparisons cover raw strided projection views, text state, anchor trimming,
FP32/FP16/BF16 and adapter strengths 0, 0.5 and 1. GPU peak memory, latency and
rendered quality still require measurement.

Normalization epsilon keeps the preceding weight-dtype rounding of `1e-6`.
Its scalar is now computed once per dtype on CPU, avoiding the CUDA allocation
and scalar read previously performed at each linear epilogue. No normalization
strength or epsilon value changes.

## Applied adapter configuration

The Apply node publishes `vdn_h3_adapter_config_v1` in transformer options after
applying its adapters. It contains the checkpoint name, mode and named strengths
from the application report, with no tensor payload. Flow can include this
configuration in its trajectory receipt even when the Apply node itself is
cached. A missing receipt from older code means the applied strength is unknown;
it must not be inferred from node defaults.

## Partitioned attention gather workspace

The partitioned softmax path uses one block-local K/V workspace pair for its
temporal groups. Each view exposes exactly the current group's rows, in the
same order and with the same contiguous strides as the preceding independent
gathers. Global sink rows are copied once; each group replaces only its video
rows. Query gathers, mapped-neighbor descriptors, prefix key measure, forced
dense decisions and all arithmetic gates are unchanged.

The pair is released after the final local group, before anchor attention,
branch-weight retrieval and learned-linear work. It is not added to the retained
runtime pool and cannot carry capacity into the following MLP or sampling stage.
This removes repeated per-group K/V allocations. It does not reduce the number
of attention calls or full-resolution transformer evaluations, and no GPU
latency or rendered-quality improvement is established by CPU tests.
