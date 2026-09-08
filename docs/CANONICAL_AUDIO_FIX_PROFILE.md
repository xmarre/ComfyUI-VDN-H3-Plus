# Canonical audio-fix profile

The canonical correction-training profile follows the released OpenVDN Turbo stage:

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

The 8-step requirement is the correction-training contract, not the preferred final deployment quality point. OpenVDN's released Stage-DMD/Turbo configuration is an 8-step trajectory with `video_shift=12` and `audio_shift=3`. A 10-step `res_multistep` run at reduced Turbo strength (for example Turbo `0.75`) is a separate deployment transfer/quality operating point and must not be silently substituted into canonical training.

For the low-host-RAM workstation path, use `tools/audio_fix_int8_train_gpu.py`. It is the authoritative production-geometry trainer: direct-CUDA H3 loading, bounded streamed INT8/ConvRot VDN branch weights, and the canonical eight-step grid.

## Memory/performance gate: passed

The original production-geometry implementation checkpointed all 50 H3 blocks independently and retained too many large activation boundaries. It completed mathematically but reached roughly 94.5 GB real GPU occupancy, spilled into shared memory, and suffered a severe late-backward paging cliff.

The current implementation uses two-level segmented reentrant activation checkpointing with five H3 blocks per outer group, inner block checkpointing only during group recomputation, ContextVar preservation, and lazy operator-local backward bridges for the inference-exact fused RMS/RoPE and INT8 `linear_input_act` paths. Progress accounting is 550 block-equivalents for the previously exercised train-index-6 smoke.

The corrected production-geometry `/550` smoke has already completed on the RTX PRO 6000 Blackwell with about 78 GB real GPU occupancy and no catastrophic paging cliff. **Do not repeat that smoke as a gate.** The memory/performance blocker is closed.

## Current learned-correction objective

The correction remains generated-audio-only at runtime. The trainable LoRA-format bank targets:

```text
blocks.*.attn.qkv_proj
blocks.*.attn.out_proj
blocks.*.mlp.fc1
```

It does not target the token refiner, AdaLN, final layer, or `fc2`. Rank and alpha are both 32; trainables are FP32. The LoRA B side is zero-initialized, so step 0 is an exact no-op.

At a selected canonical RES grid location the trainer:

1. rolls the current VDN + Stage-B + Turbo student to that state using all-actual H3 evaluations;
2. evaluates the same production H3 base with VDN/adapters disabled for the audio teacher x0;
3. evaluates the frozen VDN/Turbo student with `audio_fix` disabled for the video-preservation x0;
4. optimizes generated-audio x0 MSE to the dense base teacher plus the weighted video-preservation MSE.

This is a teacher-restoration objective, not a generic silence penalty. It can in principle preserve requested dialogue, whispering, ambience, and effects because the teacher remains prompt-conditioned. Decoded media is still required to establish that it actually does so.

## Step-1 empirical status

The one-update checkpoint loaded correctly at runtime (`audio_fix_strength=1.0`, 150 converted modules) and changed the joint trajectory enough to produce a slight video difference. On the permanent known-bad chatter seed, however, matched `audio_fix_strength=0` versus `1` media still chattered in both cases, with no perceptible reduction in chatter or loudness.

Therefore step 1 proves only that the adapter loads and influences inference. It does **not** establish corrective audio behavior.

A structural reason makes one update especially weak evidence: with LoRA `B=0` at initialization, the first backward pass has zero gradient for A and can update only B. The observed `nonzero_grad_tensors=150` is exactly consistent with one nonzero B-gradient tensor for each of the 150 target modules. A-side learning can begin only after B becomes nonzero.

## Spectrum deployment mismatch

The canonical trainer records:

```text
rollout_profile = exact_res_all_actual_no_spectrum_or_progressive_handoff
spectrum_forecasting_emulated = false
progressive_handoff_emulated = false
```

Live Spectrum H3 code confirms that a forecast call does not execute the normal H3 block stack. Spectrum predicts the compact target final-hidden state from history and applies the current timestep-conditioned H3 final/output head to that prediction. Because `audio_fix` lives inside H3 transformer blocks, it cannot execute directly on a forecast-only call; its influence reaches later forecasts through corrected actual hidden-state history.

This is a real trajectory mismatch, but it does not yet justify implementing forecast-aware training before demonstrating that more than one all-actual update moves the correction in the intended direction.

## Next gate

Do not start the 250-step run. The next experiment is a short eight-update canonical run from a fresh state, followed by decoded-media A/B on the permanent known-bad seed and control prompts.

Using trainer seed `378` is deliberate: with the current deterministic train-index selector, the first eight updates visit all eight canonical RES locations exactly once in this order:

```text
0, 3, 2, 7, 6, 1, 5, 4
```

That gives substantially more information than extending the previous seed-0 sequence, which does not cover the full grid early.

If the step-8 adapter shows no perceptible movement on the known-bad seed, do not continue to 16/32/250 merely because training loss is finite. The next work should then isolate objective/target insufficiency versus Spectrum transfer. If it clearly improves the known-bad case without damaging the controls, continue only through early checkpoints and then re-evaluate Spectrum transfer before any long run.
