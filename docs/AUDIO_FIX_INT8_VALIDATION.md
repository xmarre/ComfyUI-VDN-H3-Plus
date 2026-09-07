# Audio-fix INT8/ConvRot validation gate

This document covers the **direct Comfy production trainer** in this repository. It is intentionally stricter than a normal training recipe because the correction must be proven on the actual installed MiniMax-H3 INT8/ConvRot + VDN + released-adapter graph before a long run is allowed.

The direct-trainer defaults are the canonical released training stack:

```text
Stage-B strength      1.00
Turbo strength        1.00
global_gate_mode      checkpoint
audio routing         1.00 / 1.00 / 1.00 / 1.00
sampler               10-step res_multistep
sigma table           Comfy simple
Spectrum emulation    no
Progressive emulation no
```

Turbo is required. A Turbo-off run is not an accepted production correction path. Turbo `0.75` is a deployment transfer/validation setting after canonical training, not a trainer default.

## Current gate status

Completed on the installed production INT8/ConvRot graph:

- quantized input-gradient probe;
- zero-init sidecar B-gradient probe and subsequent A-gradient probe;
- dense-H3 training/inference forward parity;
- finished VDN/Stage-B/Turbo training/inference forward parity;
- passive Sol-H3 prefix/audio/video drift localization.

The authoritative deployment-profile parity run produced exact forward equality:

```text
dense H3:
video_rel_rms=0 video_max=0
audio_rel_rms=0 audio_max=0

VDN/Turbo student:
video_rel_rms=0 video_max=0
audio_rel_rms=0 audio_max=0

student-vs-dense:
video_max=1.86064 audio_max=1.87091
```

The nonzero student-vs-dense delta proves that exact parity was not obtained by accidentally bypassing the VDN/Turbo student.

The passive drift trace then showed a broad released-stack trajectory difference rather than a hidden sparse-audio topology bug. Block 0 begins with identical hidden rows but immediately diverges in released QKV/attention/MLP computations; cumulative video drift appears first and later feeds prefix/audio through dense cross-stream attention. No native-prefix/native-attention production override is justified by that result. See `docs/SOL_H3_AUDIO_PREFIX_AUDIT.md`.

The **next gate is exactly one optimizer step** at production geometry using the canonical training stack below.

## 0. Install the training-only dependency

The Comfy node itself has no mandatory Python dependencies. The standalone trainer additionally needs Prodigy-Plus:

```bash
cd /home/toor/ComfyUI/custom_nodes/ComfyUI-VDN-H3-Plus
python -m pip install -e '.[training]'
```

The optional extra is pinned to `prodigy-plus-schedule-free==2.0.1`.

## 1. Pull the validation branch

```bash
cd /home/toor/ComfyUI/custom_nodes/ComfyUI-VDN-H3-Plus
git fetch origin
git switch fix/audio-fidelity-controls
git pull --ff-only
```

Record the commit before testing:

```bash
git rev-parse HEAD
```

Do not compare GPU results from different heads without recording that fact.

## 2. Structural + gradient + forward-parity probe — completed

The deployment-profile command is retained for reproducibility:

```bash
python tools/audio_fix_int8_probe.py \
  --comfy-root /home/toor/ComfyUI \
  --base-model MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors \
  --vdn-checkpoint vdn-minimax-h3-int8-convrot-comfyui \
  --stage-b-strength 1.0 \
  --turbo-strength 0.75 \
  --global-gate-mode video_only
```

This is deliberately a **deployment-profile parity probe**. It verifies that the inference-exact training bridge reproduces the installed deployment graph; the `0.75` / `video_only` values above do **not** define the canonical training profile.

The hard relative-RMS limit remains `0.02`; it was not relaxed. The current production result is exactly zero for both dense H3 and the wrapped VDN/Turbo student.

### Why the trainer has an inference-exact forward bridge

Before the bridge, a corrected GPU probe measured:

```text
video_rel_rms = 0.0650898
video_max     = 0.618967
audio_rel_rms = 0.0405348
audio_max     = 0.425354
```

That was a real blocker, not tolerance noise. The global Comfy `in_training=True` state required by the autograd-safe H3/VDN graph changed two production H3 primitives:

- MiniMax-H3 switched from its in-place fused RMS/RoPE inference kernel to the functional training kernel;
- `comfy.ops.linear_input_act` disabled the production fused INT8 activation + MLP down-projection path while the global training flag was set.

The trainer keeps the global training state required for differentiable H3/VDN execution but bridges those two primitives explicitly:

- **forward value:** exact production inference primitive;
- **backward:** Comfy's supported functional/eager training primitive as a straight-through surrogate.

The production RMS/RoPE kernel runs only on detached clones and cannot mutate live autograd inputs. The fused INT8 `linear_input_act` value is likewise obtained under `no_grad`; its supported eager training path supplies the input gradient. Both patches are scoped to `comfy_quant_training_mode()` and restored even on exceptions.

## 3. Passive Sol-H3 drift localization — completed

The deployment-profile trace is retained for reproducibility:

```bash
python tools/h3_prefix_drift_probe.py \
  --comfy-root /home/toor/ComfyUI \
  --base-model MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors \
  --vdn-checkpoint vdn-minimax-h3-int8-convrot-comfyui \
  --stage-b-strength 1.0 \
  --turbo-strength 0.75 \
  --global-gate-mode video_only
```

The result does not identify one incorrect native-row operation to patch. Released adapter residuals alter prefix/audio/video branch computations from block 0, video residual drift accumulates first, and late audio divergence is additionally amplified after the transformer stack. This supports continuing the learned generated-audio correction rather than adding a destructive full-prefix clamp.

## 4. Cache a small prompt set with the installed H3 text encoder

Prepare a small JSONL or line-based prompt file that includes at minimum:

- people present with no requested dialogue;
- explicit silence / no speech;
- whisper delivery;
- normal prompted speech controls;
- ambient/effect-only scenes.

Then cache the actual MiniMax-H3 conditioning:

```bash
python tools/audio_fix_cache_prompts.py \
  --comfy-root /home/toor/ComfyUI \
  --text-encoder <MINIMAX_H3_TEXT_ENCODER.safetensors> \
  --prompts <PROMPTS.jsonl> \
  --output-dir /home/toor/audio_fix_prompt_cache
```

The cache stores the real `cond` tensor and `minimax_token_tags`; the trainer does not download another text encoder.

For the current released workflow, the installed encoder filename is typically `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`; use the actual file loaded by the workflow rather than downloading a substitute.

## 5. Run exactly one production-geometry optimizer step

Use the target workflow's **video latent** height and width, not pixel dimensions. MiniMax-H3 video latents are spatially downscaled by 16, so a pixel geometry `W x H` corresponds to trainer arguments:

```text
latent_width  = W / 16
latent_height = H / 16
```

Both resulting latent axes must be even because the H3 DiT patchifies them by `2 x 2`.

Run the smoke at the canonical released stack:

```bash
python tools/audio_fix_int8_train.py \
  --comfy-root /home/toor/ComfyUI \
  --base-model MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors \
  --vdn-checkpoint vdn-minimax-h3-int8-convrot-comfyui \
  --prompt-cache-dir /home/toor/audio_fix_prompt_cache \
  --output-dir /home/toor/audio_fix_smoke \
  --latent-height <PRODUCTION_VIDEO_LATENT_H> \
  --latent-width <PRODUCTION_VIDEO_LATENT_W> \
  --stage-b-strength 1.0 \
  --turbo-strength 1.0 \
  --global-gate-mode checkpoint \
  --smoke \
  --no-resume
```

Those three profile flags match the command defaults and are shown explicitly so the training record is unambiguous.

The fixed production contract also enforces:

```text
video latent frames = 52
audio latent frames = 292
sampler steps        = 10
video shift          = 12
audio shift          = 3
```

A successful smoke run must produce:

```text
audio_fix_step_000000/
audio_fix_step_000001/
train_state.pt
metrics.jsonl
```

The step-1 metrics must show finite `loss`, `audio_teacher_loss`, `video_preserve_loss`, `grad_norm`, a positive `nonzero_grad_tensors`, and a sane `peak_gib` for the installed GPU.

## 6. Do not start the 250-step run yet

The direct trainer is explicit about its rollout scope:

```text
rollout_profile = exact_res_all_actual_no_spectrum_or_progressive_handoff
spectrum_forecasting_emulated = false
progressive_handoff_emulated = false
```

The one-step adapter must first be exercised through the real deployment graph to prove that the learned correction survives Spectrum forecasting and Progressive/Continuum boundaries. The full 250-step run remains blocked until the Spectrum trajectory mismatch is resolved or deliberately justified with strong evidence.

## 7. Deploy the step-1 adapter for matched A/B validation

Copy only the exported adapter directory into the selected VDN stage as:

```text
<VDN_STAGE>/adapters/audio_fix/
├── adapter_config.json
└── adapter_model.safetensors
```

First validate the checkpoint against the exact canonical stack it was trained on:

```text
lora_mode                           bypass
stage_b_strength                    1.00
turbo_strength                      1.00
global_gate_mode                    checkpoint
adapter_ablation                    none
audio_adapter_strength              1.00
conditioning_adapter_strength       1.00
audio_video_context_strength        1.00
conditioning_video_context_strength 1.00
sampler                             10-step res_multistep
```

Run matched seeds through the real workflow and compare only:

```text
audio_fix_strength = 0.00
audio_fix_strength = 1.00
```

Only after canonical validation should the same checkpoint be transfer-tested at deployment Turbo strength `0.75`. Record any deployment-only routing difference, such as `global_gate_mode=video_only`, separately rather than baking it into the training profile.

Do not set `audio_adapter_strength=0` for either trained-checkpoint test. That was a useful diagnostic/mitigation for the released checkpoint, but it removes part of the full adapter stack on top of which the correction is trained.

## 8. Acceptance criteria before long training

Check decoded media, not just training loss:

- false-speech/VAD incidence on no-dialogue prompts;
- ASR non-empty rate and transcript duration;
- unwanted vocal loudness;
- whisper-vs-normal delivery compliance;
- prompted-dialogue accuracy;
- ambient/effect audio quality;
- video quality and action fidelity;
- continuity across the real Progressive/Continuum boundary.

If the step-1 direction is structurally sound through deployment, continue with early checkpoints (`2, 4, 8, 16, 32`) before committing to the full run. If the correction behaves correctly in the all-actual trainer but fails specifically through Spectrum, the next trainer must reproduce Spectrum's final-hidden-feature forecast -> current MiniMax-H3 output-head path rather than pretending a skipped H3 call has a dense x0 target.

## Resume rule

A training output directory belongs to one recorded training profile. Do not resume an optimizer state after changing Stage-B strength, Turbo strength, global gate mode, geometry, sampler profile, rank, alpha, or prompt corpus. Start a new output directory for a changed experiment profile.