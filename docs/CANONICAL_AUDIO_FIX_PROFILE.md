# Canonical audio-fix profile

The canonical correction-training profile follows the released VDN Turbo stage:

```text
Stage-B                              1.0
Turbo                                1.0
global_gate_mode                     checkpoint
adapter_ablation                     none
audio_adapter_strength               1.0
conditioning_adapter_strength        1.0
audio_video_context_strength         1.0
conditioning_video_context_strength  1.0
sampler                              res_multistep
sigma schedule                       Comfy simple
sampler steps                        8
video shift                          12
audio shift                          3
video latent frames                  52
stereo audio latent frames           292
```

The 8-step requirement is not an arbitrary speed choice. OpenVDN's released Stage-DMD configuration defines the Turbo adapter with `num_steps: 8`, `video_shift: 12.0`, and `audio_shift: 3.0`; the DMD stage is explicitly described as learning the VDN trajectory down to eight steps.

A 10-step `res_multistep` run at reduced Turbo strength (for example Turbo `0.75`) is a separate deployment transfer/quality operating point. It must not be used as the canonical Turbo-1.0 correction-training trajectory.

For the low-host-RAM workstation path, use `tools/audio_fix_int8_train_gpu.py`. That entrypoint enforces the canonical eight-step grid and uses direct-CUDA H3 loading plus bounded streamed INT8/ConvRot VDN branch weights.

## Production-geometry memory contract

A completed optimizer step is not sufficient if the process enters shared-GPU-memory/page-migration thrash. The first 48x84 production-geometry smoke exposed exactly that condition during late backward even though the optimizer eventually completed.

The observed packed H3 hidden tensor was `[53026, 5376]` BF16. One full hidden boundary therefore occupies about `0.531 GiB`. The old one-checkpoint-per-block implementation retained approximately 50 such boundaries, or about `26.55 GiB`, before temporary backward/recompute activations.

The current trainer instead uses two-level segmented reentrant activation checkpointing with five H3 blocks per outer group. Its large hidden-boundary working set is approximately:

```text
10 outer group boundaries  ~= 5.31 GiB
 5 current inner boundaries ~= 2.65 GiB
--------------------------------------
approximate boundary set    ~= 7.96 GiB
```

That is about `18.6 GiB` less large-boundary residency than the old scheme. During outer-group recomputation, individual blocks are checkpointed again, and the inference-exact RMS/RoPE and fused-fc2 training surrogates are constructed lazily inside their operator backward rather than retained through block forward.

These changes alter backward recomputation and aliasing only. They do **not** change H3 forward arithmetic, latent geometry, sampler steps, sigma schedule, VDN/Stage-B/Turbo strengths, LoRA targets, or the training objective.

The full long run remains blocked until a fresh one-step production-geometry smoke shows no shared-memory migration or catastrophic late-backward slowdown, matched decoded-media tests pass, and the Spectrum trajectory mismatch is resolved.
