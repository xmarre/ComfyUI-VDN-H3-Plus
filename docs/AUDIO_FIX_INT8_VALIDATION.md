# Audio-fix INT8/ConvRot validation gate

This document covers the **direct Comfy production trainer** in this repository. It is intentionally stricter than a normal training recipe because the correction must be proven on the actual installed MiniMax-H3 INT8/ConvRot + VDN + released-adapter graph before a long run is allowed.

The current direct-trainer defaults are the production chatter-test profile:

```text
Stage-B strength      1.00
Turbo strength        0.75
global_gate_mode      video_only
audio routing         1.00 / 1.00 / 1.00 / 1.00
sampler               10-step res_multistep
sigma table           Comfy simple
Spectrum emulation    no
Progressive emulation no
```

Turbo is required. A Turbo-off run is not an accepted production correction path.

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

## 2. Run the structural + gradient + forward-parity probe

Use the same installed base model and VDN stage as the production workflow:

```bash
python tools/audio_fix_int8_probe.py \
  --comfy-root /home/toor/ComfyUI \
  --base-model MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors \
  --vdn-checkpoint vdn-minimax-h3-int8-convrot-comfyui \
  --stage-b-strength 1.0 \
  --turbo-strength 0.75 \
  --global-gate-mode video_only
```

This must pass **all** of the following before training:

1. the installed INT8/ConvRot qkv projection propagates a finite nonzero input gradient;
2. the zero-initialized `audio_fix` sidecar is an exact forward no-op;
3. its B matrix receives a finite first-step gradient while A remains zero at zero-init;
4. after one tiny B update, A receives a finite nonzero gradient;
5. dense H3 inference and autograd-safe training execution remain within the hard relative-RMS parity limit;
6. the direct MiniMax-H3 call can see the required `vdn_h3` and `vdn_h3_audio_adapter_scope` ModelPatcher wrappers;
7. the VDN/Stage-B/Turbo inference and autograd-safe training executions remain within the same hard parity limit;
8. the direct VDN/Stage-B/Turbo student produces finite output and is not bit-identical to the wrapper-free dense H3 call.

Failure of any item is a hard stop. Do not work around it by starting training anyway, and do not raise `--max-training-parity-rel-rms` merely to pass the probe. The default `0.02` gate is deliberate.

### Why the trainer has an inference-exact forward bridge

A corrected GPU probe reached the dense-H3 comparison and measured:

```text
video_rel_rms = 0.0650898
video_max     = 0.618967
audio_rel_rms = 0.0405348
audio_max     = 0.425354
```

That was a real blocker, not tolerance noise. The global Comfy `in_training=True` state required by the autograd-safe H3/VDN graph changes two production H3 primitives:

- MiniMax-H3 switches from its in-place fused RMS/RoPE inference kernel to the functional training kernel;
- `comfy.ops.linear_input_act` disables the production fused INT8 activation + MLP down-projection path while the global training flag is set.

Across 50 transformer blocks that changed the dense model output by several percent before VDN was even involved.

The trainer now keeps the global training state required for differentiable H3/VDN execution but bridges those two primitives explicitly:

- **forward value:** exact production inference primitive;
- **backward:** Comfy's supported functional/eager training primitive as a straight-through surrogate.

The production RMS/RoPE kernel runs only on detached clones and therefore cannot mutate the live autograd inputs. The fused INT8 `linear_input_act` value is likewise obtained under `no_grad`; its supported eager training path supplies the input gradient. Both patches are scoped to `comfy_quant_training_mode()` and are restored even on exceptions.

This is intentionally different from loosening the parity threshold: the probe still compares the complete dense and VDN/Turbo outputs against the same `0.02` hard limit after the bridge is active.

## 3. Cache a small prompt set with the installed H3 text encoder

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

## 4. Run exactly one production-geometry optimizer step

Use the target workflow's **video latent** height and width, not pixel dimensions:

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
  --turbo-strength 0.75 \
  --global-gate-mode video_only \
  --smoke \
  --no-resume
```

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

and the step-1 metrics must show finite `loss`, `audio_teacher_loss`, `video_preserve_loss`, `grad_norm`, a positive `nonzero_grad_tensors`, and a sane `peak_gib` for the installed GPU.

## 5. Do not start the 250-step run yet

The direct trainer is currently explicit about its rollout scope:

```text
rollout_profile = exact_res_all_actual_no_spectrum_or_progressive_handoff
spectrum_forecasting_emulated = false
progressive_handoff_emulated = false
```

That is intentional. The one-step adapter must first be exercised through the real deployment graph to prove that the learned correction survives Spectrum forecasting and Progressive/Continuum boundaries.

## 6. Deploy the step-1 adapter for a matched A/B

Copy only the exported adapter directory into the selected VDN stage as:

```text
<VDN_STAGE>/adapters/audio_fix/
├── adapter_config.json
└── adapter_model.safetensors
```

Keep the normal production routing restored:

```text
lora_mode                           bypass
stage_b_strength                    1.00
turbo_strength                      0.75
global_gate_mode                    video_only
adapter_ablation                    none
audio_adapter_strength              1.00
conditioning_adapter_strength       1.00
audio_video_context_strength        1.00
conditioning_video_context_strength 1.00
sampler                             10-step res_multistep
```

Run matched seeds through the **actual production Spectrum + Progressive/Continuum workflow** and compare only:

```text
audio_fix_strength = 0.00
audio_fix_strength = 1.00
```

Do not set `audio_adapter_strength=0` for this test. That was a useful mitigation for the released checkpoint, but it would remove part of the full adapter stack on top of which the correction is trained.

## 7. Acceptance criteria before long training

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
