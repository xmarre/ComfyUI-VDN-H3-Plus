# Partitioned boundary feature observation

This diagnostic observes the existing native-grid-then-map short convolution.
It introduces no destination-stencil production arithmetic. Promotion of a
different stencil requires matched rendered-media evidence.

Flow publishes `h3_flow_partitioned_boundary_witness_v1` as a non-dictionary,
stage-owned object with API 1. VDN advertises
`_vdn_partitioned_boundary_witness_api=1`. The first actual low-stage learned
linear block claims the observer; subsequent blocks and probe/high calls cannot
claim it. There is no projection replay, attention replay, recurrence replay or
extra H3/provider/VAE call.

VDN forwards only observations to this sink. The existing batched arithmetic,
temporal accumulation order, physical measure and recurrence remain unchanged.
Absent observation leaves the previous path active. The public external
sequence API4 and Sol provider-v4 mapping remain unchanged.

## Captured components

For each checkpoint-requested short-conv projection, capture heads zero and the
last head, including every channel of each head. Default checkpoints request K
and V; a checkpoint requesting Q convolution is supported. Observe the first
mixed-grid boundary after the existing skipped-anchor trimming. Record that
inner boundary index and `frame_index_origin` separately.
The captured bounds, frame sizes and measure scales all use that same readout
frame domain after optional anchor trimming. `full_measure_scales` separately
records the untrimmed runtime contract.

The receiving frames within temporal radius r of the boundary need raw input
support within radius 2r. The observer captures that bounded support, native
spatial-filtered features, actual unsuppressed cross-grid mapped/weighted taps,
preactivation sums and actual activated/L2 features. Radius >4 is explicitly
unsupported by this bounded diagnostic. Captured A/B/alpha norms refer to the
actual complete-head production frame statistics, after physical measure
weighting. The receiver owns CPU copies; no retained scratch or CUDA view is
kept across evaluations.

Suppression omits cross-grid contributions from the actual model computation.
Its witness therefore contains no applied cross-grid tap tensors. Raw inputs
and native filtered features remain available for offline comparison. The
suppression execution counters remain the evidence that the intervention ran.

## Offline component comparison

Run with the Python environment containing PyTorch:

```bash
python tools/analyze_boundary_witness.py /absolute/path/boundary-witness-ID.json \
  --output /absolute/path/stencil-component-comparison.json
```

The script validates the tensor-file SHA256 and uses `torch.load` with
`weights_only=True`. It independently constructs the H3 endpoint-excluded
physical lattice. It compares native-filter-then-map with map-then-destination-
filter on those same projected inputs, signed temporal weights and selected
complete heads. It preserves the original FP32 mapping/cast policy and border
extrapolation. It reports preactivation differences, 4×4 regional RMS,
activated differences and CPU legacy-replay disagreement with the actual
captured output. The latter is separate because CPU/CUDA low-precision
convolution can differ.

This is additional CPU component work. It is not a second production forward,
a calibrated checkpoint result or proof of rendered causality. Two observed
heads in one low block are a bounded witness, not a survey of all layers or
heads. Sparse attention, global recurrence and provider/frontier effects remain
competing causes until matched runtime media distinguishes them.

## Validation

Tests establish exact output equality with and without observation, unchanged
raw inputs, both suppression states and both skipped-anchor states. Feature
tests cover requested Q/K/V convolution combinations in FP32, FP16 and BF16.
Offline replay agrees with the FP32 actual legacy preactivation and detects a
nonzero stencil-order difference. The retained scalar production oracle is
unchanged. Full pinned VDN tests, the official OpenVDN oracle and current-Core
node/compiler smoke are separate checks; SM120 wall time and media acceptance
remain required.
