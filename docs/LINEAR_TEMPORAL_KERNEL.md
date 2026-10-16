# Single-pass temporal short-conv

VDN's learned linear branch applies a separable short convolution to its key
and value features: a depthwise 5×5 spatial convolution per frame, then a
depthwise 5-tap temporal convolution across frames, then SiLU (plus L2 norm
for keys). The released arithmetic evaluates the temporal part as eager BF16
shift-multiply-add:

```python
xp = F.pad(x, (0, 0, 0, 0, 2, 2))          # one padded copy of the volume
out = xp[0:F] * w[:, 0]
out = out + xp[1:F + 1] * w[:, 1]           # ... five taps
```

On a [F, S, C] feature volume this is a padded copy plus a full-volume product
and a full-volume sum per tap: about 24 volume-sized memory passes and up to
three live volume-sized temporaries, twice per VDN block call (keys and
values).

`vdn_h3/temporal_shift.py` computes the same result in one pass on CUDA BF16:
every program holds a five-frame sliding window of one [16 × 128] tile in
registers and walks the frame axis, reading each input element once and writing
each output element once. It rounds to BF16 (round-to-nearest-even) after
every multiply and every add, in tap order, which is exactly the rounding
sequence of the eager tensor operations, so the output is bitwise identical
for finite normal-range values, infinities and overflow; zero padding uses +0
operands, so signed zeros match too. BF16 subnormals (below ~1.2e-38) may flush
on backends that flush FP32 denormals.

## Safety

* Only CUDA BF16 inputs with a 5-tap kernel use the kernel; everything else
  (CPU, FP16/FP32, other tap counts) runs the eager arithmetic unchanged.
* The first launch in a process is compared bitwise with the eager path on a
  bounded slice. A mismatch, compile error or launch error disables the kernel
  for the rest of the process with a warning; CUDA out-of-memory errors
  propagate unchanged.
* Both Apply VDN-H3 nodes expose an optional `fused_temporal` selector: `auto` (default), `on`, `off`. `auto` uses the existing `VDN_H3_FUSED_TEMPORAL` process override (unset = enabled; `0` disables); `on` explicitly requests fusion even if the environment variable is `0`; `off` always requests eager. The choice is stored per MODEL/VDN branch, not in mutable global environment state. Old workflows with no setting continue in `auto` mode.
* Eligibility is still checked for every call. `on` is **not** a force-unsafe switch: unsupported dtype/device, absent Triton, autograd, graph capture and compile tracing use eager; canary mismatch disables fusion for the process.
* The Apply node logs mode, environment and Triton availability. On the first successful fused dispatch the log says `fused temporal short-conv ACTIVE`; errors emit `disabled` and automatically fall back. A requested mode or a uniform-linear counter alone is not proof of kernel dispatch.

The variable-grid partitioned short-conv (`partitioned_linear.py`) keeps its
own per-run arithmetic; uniform-grid partitioned calls use the released
readout and therefore this kernel.

## Measuring

```bash
python tools/benchmark_vdn_linear_components.py --preset high --json vdn_linear_high.json
python tools/benchmark_vdn_linear_components.py --preset low  --json vdn_linear_low.json
```

The tool uses synthetic activations and random branch weights at H3 geometry
(56 heads × 128, configurable frames/grid). It reports the eager and fused
temporal times with a full-tensor bitwise check and incremental peak memory,
CUDA-event times of every learned-linear step, and the end-to-end
`LinearBranch.readout` with the kernel on and off (outputs compared bitwise).
These are per-block-call component times, not sampler wall time. It exits
non-zero if any comparison is not bitwise identical.
