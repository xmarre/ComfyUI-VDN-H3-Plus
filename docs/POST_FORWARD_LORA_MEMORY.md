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
Unavailable dispatch/function-mode inspection or unexpected qualification
errors disable mutation and retain the functional residual. An absent optional
mixed-precision factory is omitted from the known linear contracts; independently
qualified PyTorch and Comfy linear implementations remain eligible.

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

When the complete rank-sized down projection fits within the same 256 MiB
budget, VDN computes it once. For qualified 2-D private Linear outputs, the
default fast path uses one `torch.addmm(C, down, up.T, beta=1, out=C)`
operation to accumulate directly into that output. This eliminates both the
multi-GiB full residual and per-tile up-projection GEMMs; it does not change
the low-rank mathematical formula. Bias, if any, is added afterwards. A
single direct GEMM can have different BF16 rounding than separate up-projection
and addition, and GPU throughput/quality require installed validation.

The existing bounded tile algorithm remains active when the rank workspace
cannot be allocated, the output is batched or direct GEMM is disabled. These
tiles reuse the rank projection and retry only uncommitted rows. The rank
workspace is additional to the output tile. A direct GEMM allocation failure
is fatal, not retried with tiles: after a launched aliased-output operation,
partial modification cannot safely be ruled out. The original exception and
OOM diagnostics are preserved. If its allocation fails before the direct GEMM,
VDN uses per-tile down projections.
If an up-projection cannot fit even one row, VDN releases the retained rank
workspace and retries that uncommitted row before propagating a further failure.
This reduces projection launches; GPU speed improvement requires measurement.

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
