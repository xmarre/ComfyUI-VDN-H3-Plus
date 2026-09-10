# Audio-fix dense-H3 parity finding

The corrected full-stack GPU preflight reached the dense MiniMax-H3 parity gate and hard-stopped before VDN/Turbo comparison:

```text
dense H3 training/inference parity:
video_rel_rms=0.0650898 video_max=0.618967
audio_rel_rms=0.0405348 audio_max=0.425354
```

The `0.02` relative-RMS guard is retained. The mismatch is not accepted as tolerance noise.

## Cause

The direct trainer requires Comfy's global `model_management.in_training=True` for autograd-safe H3/VDN execution. On the installed production Comfy graph that flag also changes dense H3 forward numerics before VDN is involved:

1. MiniMax-H3 attention selects functional `rms_rope_split_half` instead of the production in-place `rms_rope_split_half_` kernel.
2. `comfy.ops.linear_input_act` leaves the production fused TensorWise-INT8 activation/down-projection path and executes the eager activation + Linear path.

Repeated across 50 blocks, those changes materially alter the model trajectory.

## Fix

`vdn_h3.audio_fix_forward_bridge` scopes two inference-exact-forward/autograd-surrogate bridges to `comfy_quant_training_mode()`:

- RMS/RoPE: production in-place kernel on detached clones supplies the forward value; the functional kernel supplies gradients.
- fused INT8 `linear_input_act`: the production fused call under `no_grad` supplies the forward value; Comfy's eager training path supplies gradients.

The live q/k graph inputs are never mutated by the inference kernel. All patched globals and Comfy training flags are restored on context exit, including exceptions.

This preserves the autograd-safe VDN branches while removing the known dense-H3 forward mismatch. It does **not** prove the full-stack parity gate passes on GPU; the same production probe must be rerun and remains authoritative.
