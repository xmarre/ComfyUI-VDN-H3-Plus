# Post-forward LoRA memory and output ownership

VDN bypass mode applies ordinary LoRA terms with PyTorch forward post-hooks.
A large MLP projection can require several GiB for the adapter residual alone.
For eligible inference calls, VDN computes the down projection, up projection
and optional bias over token-row tiles, then adds each completed tile into the
linear activation. Every adapter contribution is retained.

## Ownership contract

Tensor contiguity and distinct input storage do not establish exclusive output
ownership. Before using tiled accumulation, VDN checks on every call that:

- Its plan is the sole local forward hook, with no keyword forward hooks,
  global forward hooks or backward hooks.
- Module call dispatch is the standard PyTorch implementation and has no
  compiled call replacement or instance-level forward replacement.
- Forward methods match the installed PyTorch `Linear` or Comfy
  `disable_weight_init.Linear`, `manual_cast.Linear`, or
  `mixed_precision_ops().Linear` implementations. The Comfy cast and internal
  projection methods must also match. Comparison uses function code and globals,
  including for classes produced by the mixed-precision operations factory.
- No Python tensor dispatch or function mode can observe intermediate outputs.
- Autograd is disabled and the tensor shapes, dtype, device and storage layout
  support the same additive operation. Input and output cannot share storage.

These standard linear implementations return fresh projection storage. A
contiguous reshape spanning the complete private GEMM result is also eligible;
such a reshape is common for batched inputs. Custom forward implementations and
wrappers that might retain an activation are excluded. Hook and method checks
are repeated so a provider installed after VDN injection disables mutation.

The functional fallback preserves another observer's activation. It still needs
one full residual allocation and can therefore encounter the original OOM.
VDN does not claim bounded memory for arbitrary provider wrappers. Small outputs,
gradient-enabled calls and unsupported tensor layouts also use the functional
path. VDN never replaces `module.forward` or modifies model weights for tiling.

## Allocation and retry behavior

The initial tile size targets at most 256 MiB of output-equivalent residual
storage across all batch dimensions. At least one token row is required, even
when that row exceeds the target. This is not a total workspace cap: the down
projection, optional bias addition, input packing and GEMM backend can require
additional memory.

On a tile OOM, VDN halves the tile size and retries only uncommitted rows. A
one-row failure propagates. Completed rows are never applied twice. A fatal
failure aborts the model call; VDN does not return a partially adapted output.
Changing GEMM batch sizes can change floating-point rounding, so tiled results
are numerically equivalent within dtype tolerances rather than bit-exact.

The first successful tiled call for each plan logs
`bounded post-forward LoRA active`. An unqualified ownership contract logs
`output ownership unqualified` and uses the functional residual.

A fatal post-forward CUDA OOM logs the allocator backend, device free/total
memory, `memory_stats()` and `memory_summary()` once per plan. Each diagnostic
is best effort, and unavailable fields do not replace the allocation exception.
Some allocator statistics are backend-specific, particularly with
`cudaMallocAsync`; free memory and these counters alone do not establish whether
fragmentation, pool limits or outstanding stream work caused an allocation to
fail. This diagnostic does not change allocator configuration.
