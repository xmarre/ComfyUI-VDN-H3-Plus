# Audio-fix INT8/ConvRot validation status

This document tracks the direct Comfy production trainer and the empirical gates for the generated-audio correction. Structural correctness and decoded-media correctness are separate requirements.

## Canonical training stack

```text
Stage-B strength                     1.00
Turbo strength                       1.00
global_gate_mode                     checkpoint
adapter_ablation                     none
audio_adapter_strength               1.00
conditioning_adapter_strength        1.00
audio_video_context_strength         1.00
conditioning_video_context_strength  1.00
sampler                              8-step res_multistep
sigma table                          Comfy simple
video shift                          12
audio shift                          3
video latent frames                  52
stereo audio latent frames           292
Spectrum emulation                   no
Progressive emulation                no
```

Turbo `0.75` and 10-step `res_multistep` are deployment transfer/quality settings. They are not canonical correction-training defaults.

## Completed gates

The following have already passed on the installed production INT8/ConvRot graph:

- quantized input-gradient probe;
- zero-init B-gradient and subsequent A-gradient structural probes;
- dense-H3 training/inference forward parity;
- finished VDN/Stage-B/Turbo training/inference forward parity;
- passive Sol-H3 prefix/audio/video drift localization;
- production-geometry segmented-checkpoint optimizer smoke.

The forward-parity probe reached exact equality for dense H3 and the wrapped VDN/Turbo student while retaining a nonzero student-vs-dense delta, ruling out accidental VDN/Turbo bypass.

The passive drift trace showed broad released-stack trajectory divergence rather than one hidden sparse-audio topology bug: block 0 begins from identical hidden rows, released adapter computations create divergence immediately, video drift accumulates first, and dense joint attention feeds that altered state back into later prefix/audio computation. See `docs/SOL_H3_AUDIO_PREFIX_AUDIT.md`.

### Memory/performance gate is closed

The old 50-independent-checkpoint path reached roughly 94.5 GB real GPU occupancy and paged badly during late backward. The corrected two-level segmented reentrant path completed the `/550` production-geometry smoke at about 78 GB real GPU occupancy without the catastrophic paging cliff.

Do **not** repeat that one-step memory smoke as a prerequisite. The remaining blockers are semantic efficacy and deployment-trajectory transfer.

## What the current loss is actually teaching

At one selected RES grid location, the trainer rolls the current student to that state, then computes:

```text
audio_teacher_loss = MSE(student_audio_x0 / 4, dense_base_teacher_audio_x0 / 4)
video_preserve_loss = MSE(student_video_x0, frozen_VDN_Turbo_video_x0)
loss = audio_teacher_loss + 0.1 * video_preserve_loss
```

The audio target is the same production H3 base with VDN/adapters disabled. This is not a VAD loss, ASR loss, or speech-suppression loss. It asks the generated-audio rows to recover dense-base teacher behavior while preserving the finished VDN/Turbo video output.

This objective is intentionally safer than a blanket anti-speech penalty because requested dialogue, whispering, ambience, and effects remain part of the prompt-conditioned teacher target. Its limitation is that x0 MSE does not directly encode the semantic distinction between unwanted speech and desired non-speech audio. Decoded media must decide whether the teacher-restoration direction correlates with the actual failure.

## Why step 1 was not a meaningful capacity test

The runtime successfully loaded the one-update adapter (`audio_fix_strength=1.0`, 150 converted modules). Earlier training metrics were approximately:

```text
loss                  0.02547
audio_teacher_loss    0.02547
video_preserve_loss   0.0
grad_norm             0.00624
nonzero_grad_tensors  150
```

The matched production A/B on the permanent known-bad chatter seed did **not** perceptibly reduce chatter or loudness. Both `audio_fix_strength=0` and `1` chattered; the slight video divergence only proves that the adapter influenced the joint trajectory.

The first optimizer step is also structurally special. Each LoRA delta is `B(Ax)`, and B starts at exactly zero. On the first backward pass:

```text
dL/dA = B^T (...) = 0
```

so only B can learn. With 150 target modules, `nonzero_grad_tensors=150` is the expected B-only result. After the optimizer makes B nonzero, A can begin receiving gradient. One update therefore cannot distinguish "wrong objective" from "correct objective but effectively only initialized one side of the factorization".

The authoritative GPU trainer now reports `nonzero_lora_a_grad_tensors` and `nonzero_lora_b_grad_tensors` separately so the next run verifies this directly rather than inferring it from a total count.

## Prodigy short-run audit

The training dependency is pinned to `prodigy-plus-schedule-free==2.0.1`. The trainer configures:

```text
lr          1.0
d0          1e-6
d_limiter   true
d_coef      1.0
```

In that optimizer version, `d_limiter` caps the candidate adaptive scale at `d * 2^(1/4)` per optimizer step. Thus even if every update hits the limiter ceiling, the post-step upper bounds are only:

```text
step 8   4e-6
step 16  16e-6
step 24  64e-6
```

Actual growth can be slower. This changes the interpretation of the previous step-8 proposal: a null media result at eight updates cannot safely reject the teacher-restoration objective, because the adapter is still in a deliberately conservative optimizer-start regime. The GPU trainer now records `prodigy_d` and `prodigy_d_prev` in every metrics row.

## Spectrum audit

The live Spectrum implementation was rechecked before choosing the next experiment.

- Actual calls execute H3 and observe the target final hidden state into Spectrum history.
- Forecast calls do not execute the regular H3 blocks.
- A forecast predicts the compact target final hidden state from history, then applies H3's current timestep-conditioned final/output head and unpacks video/audio output.
- `audio_fix` therefore cannot run directly on forecast-only calls; it affects them only through corrected actual hidden-state history.

The canonical trainer's all-actual rollout is consequently not identical to the production Spectrum trajectory. This remains an open transfer question, not evidence that Spectrum caused the original yapping.

## Next experiment: 25 fresh canonical updates, not full250

The next run is a short 25-update experiment from a fresh training state. The information target is specific: once both LoRA factors are active, all eight canonical RES positions have been visited, and the adaptive optimizer scale has had materially more room to grow, does the current dense-teacher restoration objective produce any useful decoded-media direction?

Use seed `378`. Under the current deterministic selector its first eight rollout indices are:

```text
0, 3, 2, 7, 6, 1, 5, 4
```

so every canonical grid location is trained exactly once before repetition. The first 25 indices are:

```text
0, 3, 2, 7, 6, 1, 5, 4, 3, 7, 0, 6, 0, 1, 7, 7, 5, 7, 7, 3, 6, 5, 7, 6, 5
```

Under the trainer's progress accounting, these 25 updates represent 11,750 H3 block-equivalents. That is about `21.36x` the already-passed train-index-6 `/550` smoke. This is the defensible compute estimate. The corrected smoke's elapsed seconds were not preserved in the repository, so do not derive wall-clock time from the obsolete ~20-minute paged implementation. The new run logs `seconds` per update and provides its own real ETA after the first few steps.

### Training-side stop conditions

Do not interpret decreasing training loss as success. Require structural health:

- finite loss and gradients;
- no recurrence of the paging cliff;
- step 1 should report B-side gradients with zero A-side nonzero-gradient tensors;
- by step 2-4, both A-side and B-side nonzero-gradient counts should be positive;
- `prodigy_d` and `prodigy_d_prev` must be present in metrics;
- checkpoints 8, 16, 24, and final 25 must export and load normally.

If A-side gradients remain absent through step 4, stop and debug the factorized update path. If `prodigy_d` remains effectively pinned near `d0`, do not treat a negative media result as proof that the objective or target set is wrong; that would instead identify optimizer starvation.

Step 8 is only an early-direction checkpoint. Improvement there is positive evidence; no improvement is inconclusive. Step 25 is the primary short-run media checkpoint. Checkpoints 8 and 16 are retained so that an overshoot or late degradation at step 25 can be distinguished from a consistently wrong direction.

## Decoded-media acceptance criteria

The primary deployment acceptance remains a matched A/B in the real workflow where **only** `audio_fix_strength` changes.

Keep at least one permanent seed that reliably yaps. For the step-25 checkpoint compare:

```text
audio_fix_strength = 0.00
audio_fix_strength = 1.00
```

with the same seed, prompt, references, sampler/scheduler, Stage-B/Turbo strengths, Spectrum settings, Progressive/Continuum handoff, DiffAid, and RoPE configuration.

Evaluate all of:

- unwanted speech/gibberish incidence and duration;
- unwanted vocal loudness;
- no-speech/silence compliance;
- whisper compliance;
- requested normal dialogue preservation;
- ambient/effect-only audio preservation;
- video/action/composition preservation.

A clean seed is not evidence of success. The permanent known-bad seed must improve.

## Decision after step 25

Use the following decision rule before spending more training compute:

1. **Production Spectrum improves clearly and controls remain intact:** current objective/targets have evidence of the right direction. Continue only through early checkpoints; still resolve Spectrum transfer before a long run.
2. **All-actual diagnostic improves but production Spectrum does not:** prioritize Spectrum-trajectory emulation/robustness. Do not widen LoRA targets yet.
3. **Neither production nor all-actual improves, A/B gradients are healthy, and `prodigy_d` has materially escaped the initial floor:** the evidence shifts toward an insufficient teacher/objective signal or target set. Inspect semantic supervision and target coverage before more updates.
4. **`prodigy_d` remains near the initial floor:** revise optimizer scaling or the diagnostic budget before drawing semantic conclusions.
5. **Chatter drops only by suppressing requested speech/whisper/ambience:** reject the checkpoint and redesign the objective; do not promote a generic speech suppressor.
6. **Video/action/composition drifts materially:** reject or strengthen preservation constraints before scaling training.

For the all-actual diagnostic only, Spectrum's `enabled=false` path is the correct way to force actual H3 execution. That diagnostic is for causal isolation; it is not a deployment success criterion.

## Long-run gate

Do not start 250 steps until decoded-media evidence shows that the correction direction is useful and either:

- it transfers through the production Spectrum + Progressive/Continuum trajectory, or
- the trainer has been deliberately updated to reproduce the relevant forecast trajectory and that path has passed matched media validation.

A changed Stage-B/Turbo profile, sampler grid, global gate, geometry, rank/alpha, prompt corpus, optimizer profile, or rollout semantics belongs in a new training output directory. Do not resume optimizer state across those changes.
