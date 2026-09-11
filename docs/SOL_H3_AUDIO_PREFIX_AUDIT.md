# Sol-H3 audio/prefix audit for VDN-H3

Status: diagnostic investigation complete. This document does **not** propose Sol-Attn as a VDN backend and does not change the production attention defaults.

Source snapshot audited:

- `xmarre/Sana` branch `sol-engine` at `2936c47637380842aaa4a4488fac5006cc542b70`;
- current Sol-H3 H3 integration lives in `models/minimax_h3/Sol-H3/h3_runtime/sparse_attention.py` (the current branch no longer has the older/requested `sol_attn_h3.py` path);
- `h3_runtime/sol_residual.py`, `h3_runtime/engine.py`, and the Sol-H3 README were checked together;
- native ComfyUI MiniMax-H3 was checked at `eb357862592aefe8e136031b9e3aa14e55abddaa`.

## What Sol-H3 adds to this investigation

The useful Sol-H3 result is a correctness lesson, not its sparse kernel. The H3-specific integration records a case where visual quality remained strong while dialogue failed, then treats target audio as generated state rather than passive conditioning. Its default T2V/I2V routing therefore makes the complete pre-target-video prefix an exact K/V sink and computes those prefix query rows densely. The implementation explicitly warns that a visual-only metric can rate a broken audio result too highly.

The current Sol-H3 implementation also differs materially from older generic Sol-Attn integration:

- no per-call Morton permutation for H3; the H3 target-video tail is already contiguous/grid ordered;
- exact sink K/V blocks are selected rather than approximated;
- the default T2V/I2V sink is the whole prefix, not text only;
- prefix query rows are exact dense queries (`sol`) or exact all-block BSA query blocks (`sol_bsa`);
- generated target audio is explicitly included in that correctness boundary;
- per-shape tau calibration was removed; the current path uses the fixed routing contract instead;
- sparse correctness gates compare all-block sparse/BSA execution against dense SDPA on real H3 Q/K/V.

One important nuance: current `engine.py` uses `sink_mode="prefix"` for T2V/I2V, but `sink_mode="text_audio"` for Ref2VA. Ref2VA therefore does **not** make every visual-reference row an exact sink. This is one reason not to translate the Sol policy into a blanket VDN rule.

## Current native H3 packed order

Current ComfyUI derives the order from `PackedLayout`, not from a fixed token count:

- T2VA / FL2VA: `[text | keyframe/conditioning rows | target audio | target video]`;
- Ref2VA: `[text | reference block 1 | reference block 2 | ... | target audio | target video]`;
- a reference-video soundtrack is packed immediately before that reference's video rows inside its reference block;
- target audio is always one contiguous segment immediately before the final target-video segment.

VDN derives `audio_start`, `audio_end`, `video_start`, and `video_end` from this authoritative layout and fails closed unless `audio_end == video_start`.

## Native H3 vs finished VDN + Stage-B + Turbo

The table below distinguishes direct row writes from indirect drift through later dense cross-modal attention.

| Component | Packed prefix (text / refs / conditioning) | Generated audio | Generated video |
|---|---|---|---|
| QKV projections | Native projection plus released adapter residuals | Native projection plus released adapter residuals | Native projection plus released adapter residuals |
| Q/K norm + RoPE | Same native operation | Same native operation | Same native operation |
| Allowed attention K/V | **Dense against every row** | **Dense against every row** | Every global/prefix/audio K/V row is present exactly; only video-video K/V is windowed/anchored |
| Softmax result | Dense | Dense | Windowed video softmax plus exact global context |
| VDN softmax gate | Released checkpoint gate unless `global_gate_mode=video_only` | Same | Released gate |
| Attention `out_proj` | Native projection plus released adapter residuals | Native projection plus released adapter residuals | Native projection plus released adapter residuals |
| VDN linear/readout branch | No direct output write | No direct output write | Direct branch output added only to target-video rows |
| Block residual | Native residual structure; input can already differ | Native residual structure; input can already differ | Native residual structure plus VDN attention change |
| MLP | Stage-B does not target it; released Turbo does | Same | Same |
| Stage-B adapter | DiT Q/K/V/out plus token-refiner Q/K/V/out | DiT Q/K/V/out | DiT Q/K/V/out |
| Turbo adapter | Checkpoint target set includes DiT Q/K/V/out, FF, AdaLN; token-refiner attention/FF; final `norm_out` | Same row-wise DiT/FF effects plus audio-modality AdaLN/final effects | Same row-wise DiT/FF effects plus video-modality AdaLN/final effects |
| Token refiner | Text-only preprocessing, modified by Stage-B and Turbo when real 5120-wide prompt conditioning is used | Not a packed-audio operation | Not a packed-video operation |
| AdaLN/modulation | Native structure; Turbo modifies the modulation projection | Native structure; Turbo modifies audio modulation rows/chunk | Native structure; Turbo modifies video modulation |
| Final layer/output head | Not emitted | Native audio head fed by the finished hidden trajectory; Turbo `norm_out`/final-AdaLN residual can modify its modulation | Native video head fed by finished hidden trajectory |
| Audio-fix training substitutions | N/A | Direct Comfy trainer bridges training-only RMS/RoPE and fused INT8 MLP forward differences back to production-exact forward values; differentiable VDN recurrence remains the training surrogate | Same bridge/recurrence contract |

### Released adapter scope matters

The released Stage-B training recipe targets only DiT attention Q/K/V/out and the two token-refiner attention Q/K/V/out projections. The released Turbo adapter is broader: its `model_spec.json` enumerates transformer attention, FF, AdaLN, token-refiner attention/FF, and final `norm_out.linear` targets.

Consequently, `conditioning_adapter_strength=0` was a useful packed sequence-linear diagnostic, but it was **not** a complete "native prefix" restoration: token-refiner work is outside the packed scope, and the current conditioning strength does not suppress the conditioning modality's AdaLN projection terms. That does not invalidate the prior decoded-media result; it narrows what that ablation proved.

## What VDN already does correctly relative to the Sol-H3 lesson

VDN does not have the specific generic-Sol failure "audio/prefix queries became sparse":

1. `window_softmax_grouped` and the Flex mask keep every row outside target video global.
2. Every global query (text, reference/conditioning, target audio) attends densely to **all** K/V rows, including target video.
3. Every target-video query sees every global K/V row exactly; only target-video-to-target-video attention is windowed.
4. The VDN linear complement writes only target-video rows.
5. `global_gate_mode=video_only` removes the learned VDN softmax gate from all pre-video rows while preserving the checkpoint gate on video.

Therefore "make audio dense" or "make the prefix an exact K/V sink" is not a new VDN fix. The relevant Sol-H3 question was whether the **finished hidden trajectory** of those rows drifts because the video rows and released adapters change what later dense prefix/audio queries see.

## Remaining ways target audio/prefix can diverge from native H3

Even with dense prefix/audio attention and `global_gate_mode=video_only`, the finished stack can differ from native H3 through:

1. Stage-B/Turbo Q/K/V projection residuals on the prefix/audio rows themselves.
2. Stage-B/Turbo Q/K/V residuals on video rows, which change the K/V seen by later dense audio/prefix queries.
3. VDN's direct video-row branch/window changes, which change generated-video hidden state and therefore future video K/V.
4. Stage-B/Turbo `out_proj` residuals on prefix/audio rows.
5. Turbo FF (`fc1` / fused `fc2`) residuals on prefix/audio rows.
6. Turbo block AdaLN modulation, including the audio modality.
7. Stage-B/Turbo token-refiner changes to real text conditioning before the packed DiT stack.
8. Turbo final `norm_out`/final-AdaLN modulation before the audio output head.

Prior decoded-media evidence already constrains several of these:

- `global_gate_mode=video_only` did not remove semantic yapping;
- `audio_adapter_strength=0` strongly reduced chatter loudness/incidence but did not remove the semantic prior;
- removing generated-video K/V from audio queries damaged legitimate audio while yapping remained;
- removing video feedback from conditioning radically changed the shot/action while yapping remained;
- packed conditioning sequence-linear adapter suppression did not remove yapping.

The Sol-H3 finding therefore does **not** overturn the learned-correction conclusion.

## Passive drift probe

`tools/h3_prefix_drift_probe.py` compares one small native-H3 inference call with the finished VDN/Stage-B/Turbo call at the same top-level latent/timestep/context input and reports, per block and per grouped PackedLayout segment:

- cumulative block-input hidden drift;
- raw Q, K, V drift after the finished projection hooks;
- attention result drift immediately before `out_proj`;
- attention-module output drift;
- MLP output drift;
- final block-output drift;
- relative RMS, cosine similarity, and max absolute delta.

Dense snapshots are moved to CPU in BF16 and the tool fails closed above a small packed-row limit. It is intentionally a structural probe, not a production-resolution tensor recorder. The hidden-width synthetic context skips the token refiner on purpose; real prompt-cache / trainer smoke exercises that path.

### Production GPU result

The production-profile run used Stage-B `1.0`, Turbo `0.75`, `global_gate_mode=video_only` and a 200-row synthetic layout:

```text
prefix [0,8)
audio  [8,72)
video  [72,200)
```

All three row classes enter block 0 exactly equal to dense H3. Divergence therefore begins inside the finished block computation rather than arriving from a previously altered residual stream.

Block-0 relative-RMS deltas already show direct released-stack perturbation:

```text
prefix: q_raw=0.01426  attn_out=0.02335  mlp_out=0.02709  block_out=0.009045
audio : q_raw=0.004615 attn_out=0.01355  mlp_out=0.02389  block_out=0.004195
video : q_raw=0.005662 attn_out=0.02754  mlp_out=0.03071  block_out=0.02371
```

Cumulative block-input drift then crosses the following thresholds:

```text
             >=1%   >=5%   >=10%
prefix       b05    b29    b31
audio        b05    b34    b36
video        b01    b22    b28
```

The ordering is important. Video hidden state diverges first because VDN directly changes the video path. Prefix/audio rows are nevertheless not native within each block: their QKV/attention/MLP branch results differ from block 0 because Stage-B/Turbo apply to those rows. Later dense cross-stream attention then exposes prefix/audio queries to the increasingly non-native video K/V trajectory.

At the final transformer block the audio hidden-state relative RMS is about `0.1602`, but the emitted audio output reaches `0.4172`, indicating substantial post-block amplification consistent with the broader Turbo final modulation/output path. The final synthetic call measured:

```text
video: rel_rms=0.3095 cosine=0.953429 max_abs=2.19
audio: rel_rms=0.4172 cosine=0.918066 max_abs=1.856
```

These tensor metrics are diagnostic evidence only; they do not prove decoded-media semantic quality by themselves.

## Exact-prefix / exact-audio diagnostics: decision

### Attention topology level

For generated-audio queries and all other pre-video queries, native **dense attention topology** is already present in VDN. With `global_gate_mode=video_only`, the VDN-specific softmax gate is also removed from those rows. No new sparse-attention replacement is needed.

### Whole native row-update level

A stronger shadow experiment could recompute the installed base-H3 attention or full block on the same current hidden state and replace only target-audio/all-prefix rows while leaving the VDN video rows intact. The passive trace does not justify promoting that into the runtime:

- divergence is already present in released QKV and MLP projections at block 0;
- the video trajectory then becomes strongly non-native and feeds the dense global rows;
- late audio is additionally amplified by the final path;
- prior decoded-media adapter/context ablations reduced severity or damaged valid semantics without eliminating yapping.

There is therefore no single incorrect native-row execution boundary isolated by this trace. A full native-prefix/full-block clamp would remove intended Stage-B/Turbo behavior and would be a destructive research ablation, not a supported production fix.

The Sol-H3 diagnostic gate is closed for this PR. No Sol sparse-attention dependency and no native-prefix replacement mode are added.

## Trainer implications

### Canonical `xmarre/vdn-minimax-h3` audio-fix trainer

The canonical trainer remains logically sound for its current T2VA corpus:

- row ownership comes from Diffusers' own packed sequence indices;
- rollout is on-policy in the frozen VDN + Stage-B + Turbo student;
- the teacher audio target is dense released H3 evaluated at the exact current student sampler state;
- the teacher is not Sol sparse attention;
- the audio-fix sidecar applies only to target-audio indices.

Its current packing helper explicitly builds T2VA with no conditioning/reference rows. Ref2VA/keyframe robustness therefore remains a deployment validation requirement rather than something the current training corpus proves.

### Direct Comfy INT8/ConvRot trainer

The direct trainer derives target-audio ownership from current Comfy `PackedLayout`, keeps the dense native-H3 teacher, and has now passed the production INT8/ConvRot training/inference forward-parity probe exactly for both dense H3 and the finished VDN/Turbo student. The Sol-H3 finding does not justify lowering that gate or substituting a sparse teacher.

Spectrum and Progressive/Continuum remain outside the all-actual rollout and still require real deployment validation.

## Decision before further training

The Sol-H3 investigation and production numerical-parity gates are complete. The next allowed step is exactly one production-geometry optimizer smoke using the canonical training stack:

```text
Stage-B = 1.0
Turbo = 1.0
global_gate_mode = checkpoint
10-step res_multistep
```

Do not begin the long run after that smoke alone. The generated checkpoint must be decoded in matched media tests, and the Spectrum trajectory mismatch plus real Progressive/Continuum/reference-heavy validation remain open gates.