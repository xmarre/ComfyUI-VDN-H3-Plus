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
 audio shift                         3
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

## Spectrum audit

The live Spectrum implementation was rechecked before choosing the next experiment.

- Actual calls execute H3 and observe the target final hidden state into Spectrum history.
- Forecast calls do not execute the regular H3 blocks.
- A forecast predicts the compact target final hidden state from history, then applies H3's current timestep-conditioned final/output head and unpacks video/audio output.
- `audio_fix` therefore cannot run directly on forecast-only calls; it affects them only through corrected actual hidden-state history.

The canonical trainer's all-actual rollout is consequently not identical to the production Spectrum trajectory. This remains an open transfer question, not evidence that Spectrum caused the original yapping.

## Next experiment: short canonical run, not full250

Run eight fresh optimizer updates before redesigning the objective. This experiment is justified because it answers a specific question that step 1 could not answer: once both LoRA factors can learn and all canonical RES locations have been touched, does the current teacher-restoration objective move decoded media in the correct direction at all?

Use seed `378`. Under the current deterministic selector its first eight rollout indices are:

```text
0, 3, 2, 7, 6, 1, 5, 4
```

so every canonical grid location is trained exactly once. The eight updates represent 3400 H3 block-equivalents in total (average 425/update), versus 550 block-equivalents for the prior train-index-6 smoke. Ignoring one-time setup/checkpoint I/O, the compute is therefore about `6.18x` that corrected one-step smoke, not eight copies of the obsolete ~20-minute paged run. The exact corrected smoke `seconds` field is not preserved in the repository, so do not fabricate a tighter wall-clock estimate; use the trainer's per-step `seconds` telemetry from this run.

### Training-side stop conditions

Do not interpret decreasing training loss as success. Require only structural health during the eight updates:

- finite loss and gradients;
- no recurrence of the paging cliff;
- by step 2-4, `nonzero_grad_tensors` should rise above the step-1 B-only count of 150, showing that A-side gradients are now active;
- checkpoints 2, 4, and 8 must export and load normally.

If A-side gradients remain absent through step 4, stop and debug the factorized update path rather than spending more GPU time.

## Decoded-media acceptance criteria

The primary deployment acceptance remains a matched A/B in the real workflow where **only** `audio_fix_strength` changes.

Keep at least one permanent seed that reliably yaps. For the step-8 checkpoint compare:

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

## Decision after step 8

Use the following decision rule before spending more training compute:

1. **Production Spectrum improves clearly and controls remain intact:** current objective/targets have evidence of the right direction. Continue only through early checkpoints; still resolve Spectrum transfer before a long run.
2. **All-actual diagnostic improves but production Spectrum does not:** prioritize Spectrum-trajectory emulation/robustness. Do not widen LoRA targets yet.
3. **Neither production nor all-actual improves:** the evidence shifts toward an insufficient objective/teacher signal or target set. Inspect semantic supervision and target coverage before more updates.
4. **Chatter drops only by suppressing requested speech/whisper/ambience:** reject the checkpoint and redesign the objective; do not promote a generic speech suppressor.
5. **Video/action/composition drifts materially:** reject or strengthen preservation constraints before scaling training.

For the all-actual diagnostic only, Spectrum's `enabled=false` path is the correct way to force actual H3 execution. That diagnostic is for causal isolation; it is not a deployment success criterion.

## Long-run gate

Do not start 250 steps until decoded-media evidence shows that the correction direction is useful and either:

- it transfers through the production Spectrum + Progressive/Continuum trajectory, or
- the trainer has been deliberately updated to reproduce the relevant forecast trajectory and that path has passed matched media validation.

A changed Stage-B/Turbo profile, sampler grid, global gate, geometry, rank/alpha, prompt corpus, or rollout semantics belongs in a new training output directory. Do not resume optimizer state across those changes.
