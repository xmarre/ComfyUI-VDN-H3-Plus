# Canonical audio-fix profile

The canonical correction-training profile follows the released VDN Turbo stage:

```text
Stage-B                         1.0
Turbo                           1.0
global_gate_mode                checkpoint
adapter_ablation                none
audio_adapter_strength          1.0
conditioning_adapter_strength   1.0
audio_video_context_strength    1.0
conditioning_video_context_strength 1.0
sampler                         res_multistep
sigma schedule                  Comfy simple
sampler steps                   8
video shift                     12
 audio shift                    3
video latent frames             52
stereo audio latent frames      292
```

The 8-step requirement is not an arbitrary speed choice. OpenVDN's released Stage-DMD configuration defines the Turbo adapter with `num_steps: 8`, `video_shift: 12.0`, and `audio_shift: 3.0`; the DMD stage is explicitly described as learning the VDN trajectory down to eight steps.

A 10-step `res_multistep` run at reduced Turbo strength (for example Turbo `0.75`) is a separate deployment transfer/quality operating point. It must not be used as the canonical Turbo-1.0 correction-training trajectory.

For the low-host-RAM workstation path, use `tools/audio_fix_int8_train_gpu.py`. That entrypoint enforces the canonical eight-step grid and uses direct-CUDA H3 loading plus bounded streamed INT8/ConvRot VDN branch weights.

The full long run remains blocked until the one-step optimizer/media smoke succeeds and the Spectrum trajectory mismatch is resolved.