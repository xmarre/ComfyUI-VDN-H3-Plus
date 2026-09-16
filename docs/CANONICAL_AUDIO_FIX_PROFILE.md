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

A structural reason makes one update especially weak evidence: with LoRA `B=0` at initialization, the first backward pass has zero gradient for A and can update only B. The observed `nonzero_grad_tensors=150` is exactly consistent with one nonzero B-gradient tensor for each of the 150 target modules. A-side learning can begin only after B becomes nonzero. The GPU trainer now reports A-side and B-side nonzero-gradient counts separately.

## Optimizer warm-up matters to the short-run gate

The trainer pins `prodigy-plus-schedule-free==2.0.1` and currently uses `d0=1e-6`, `d_limiter=True`, and `lr=1.0`. In that optimizer version, `d_limiter` caps each update of the adaptive `d` estimate to at most `2^(1/4)` times the previous value. Even under maximum allowed growth, the post-step ceilings are therefore approximately:

```text
step 8   4e-6
step 16  16e-6
step 24  64e-6
```

The step-1 checkpoint was consequently both B-only **and** in the deliberately tiny initial Prodigy regime. A null result at step 8 would still be weak evidence against the objective. The authoritative GPU trainer records `prodigy_d`, `prodigy_d_prev`, `nonzero_lora_a_grad_tensors`, and `nonzero_lora_b_grad_tensors` so the next short experiment can distinguish optimizer starvation from an ineffective learned direction.

## Spectrum deployment mismatch

The canonical trainer records:

```text
rollout_profile = exact_res_all_actual_no_spectrum_or_progressive_handoff
spectrum_forecasting_emulated = false
progressive_handoff_emulated = false
```

Live Spectrum H3 code confirms that a forecast call does not execute the normal H3 block stack. Spectrum predicts the compact target final-hidden state from history and applies the current timestep-conditioned H3 final/output head to that prediction. Because `audio_fix` lives inside H3 transformer blocks, it cannot execute directly on a forecast-only call; its influence reaches later forecasts through corrected actual hidden-state history.

This is a real trajectory mismatch, but it does not yet justify implementing forecast-aware training before demonstrating that a short all-actual run can move the correction in the intended direction at all.

## Next gate: short 25, not full 250

Do not start the 250-step run. The next experiment is a fresh **25-update** canonical run. The purpose is not to establish final quality; it is to determine whether the present teacher-restoration objective and target set produce any useful decoded-media direction once both LoRA factors are active and Prodigy's adaptive scale has had enough updates to leave the immediate `d0` regime.

Using trainer seed `378` is deliberate: the first eight updates visit all eight canonical RES locations exactly once:

```text
0, 3, 2, 7, 6, 1, 5, 4
```

The first 25 selected indices are:

```text
0, 3, 2, 7, 6, 1, 5, 4, 3, 7, 0, 6, 0, 1, 7, 7, 5, 7, 7, 3, 6, 5, 7, 6, 5
```

This is 11,750 H3 block-equivalents in the progress model, about `21.36x` the already-passed train-index-6 `/550` smoke, before one-time setup and checkpoint I/O. The corrected smoke's wall-clock duration is not preserved in the repository, so a tighter wall-clock prediction would be fabricated. Use the per-step `seconds` telemetry from the new run for the actual ETA; do not extrapolate from the obsolete paged ~20-minute run.

Checkpoints 8, 16, 24, and final 25 are retained. Step 8 is an **early-direction probe only**: improvement there is informative, but no improvement is not a rejection criterion because the optimizer scale is still tightly limited. Step 25 is the primary short-run media checkpoint.

Training must remain structurally healthy:

- finite loss and gradients;
- no recurrence of the paging cliff;
- step 1 should show B-side gradients and zero A-side gradient tensors;
- by steps 2-4, A-side and B-side gradients should both be active;
- `prodigy_d` must be recorded and interpreted together with media results;
- exported checkpoints must load normally.

If A-side gradients remain absent through step 4, stop. If `prodigy_d` remains effectively pinned to `d0` through the run, a negative media result is an optimizer-scaling result, not evidence that the semantic objective is wrong.

After step 25, perform matched decoded-media A/B on the permanent known-bad seed and control prompts in the real Spectrum + Progressive/Continuum workflow. Vary only `audio_fix_strength=0` versus `1`. Evaluate unwanted speech/gibberish, vocal loudness, silence compliance, whisper compliance, requested normal dialogue, ambience/effects, and video/action/composition preservation.

Decision after step 25:

1. Production Spectrum improves clearly and controls remain intact: the current objective/targets have evidence of the right direction. Continue only through early checkpoints; resolve Spectrum transfer before any long run.
2. An all-actual diagnostic improves but production Spectrum does not: prioritize Spectrum-trajectory emulation/robustness rather than widening LoRA targets.
3. Neither path improves **and** A/B gradients are healthy **and** `prodigy_d` has materially escaped its initial floor: evidence shifts toward an insufficient teacher/objective signal or target set.
4. `prodigy_d` remains near its initial floor: revise optimizer scaling or the diagnostic budget before drawing semantic conclusions.
5. Chatter drops only by suppressing requested speech, whispering, or ambience: reject the checkpoint; do not promote a generic speech suppressor.
6. Video/action/composition drifts materially: reject or strengthen preservation constraints before scaling training.
