# Sol-H3 audio/prefix audit for VDN-H3

Status: diagnostic investigation. This document does **not** propose Sol-Attn as a VDN backend and does not change the production attention defaults.

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
| Audio-fix training substitutions | N/A | Direct Comfy trainer now bridges training-only RMS/RoPE and fused INT8 MLP forward differences back to production-exact forward values; differentiable VDN recurrence remains the training surrogate | Same bridge/recurrence contract |

### Released adapter scope matters

The released Stage-B training recipe targets only DiT attention Q/K/V/out and the two token-refiner attention Q/K/V/out projections. The released Turbo adapter is broader: its `model_spec.json` enumerates transformer attention, FF, AdaLN, token-refiner attention/FF, and final `norm_out.linear` targets.

Consequently, `conditioning_adapter_strength=0` was a useful packed sequence-linear diagnostic, but it was **not** a complete "native prefix" restoration: token-refiner work is outside the packed scope, and the current conditioning strength does not suppress the conditioning modality's AdaLN projection terms. That does not invalidate the prior decoded-media result; it narrows what that ablation proved.

## What VDN already does correctly relative to the Sol-H3 lesson

VDN does not have the specific generic-Sol failure "audio/prefix queries became sparse":

1. `window_softmax_grouped` and the Flex mask keep every row outside target video global.
2. Every global query (text, reference/conditioning, target audio) attends densely to **all** K/V rows, including target video.
3. Every target-video query sees every global K/V row exactly; only target-video-to-target-video attention is windowed.
4. The VDN linear complement writes only target-video rows.
5. `global_gate_mode=video_only` already removes the learned VDN softmax gate from all pre-video rows while preserving the checkpoint gate on video.

Therefore "make audio dense" or "make the prefix an exact K/V sink" is not a new VDN fix. The relevant Sol-H3 question is instead whether the **finished hidden trajectory** of those rows drifts because the video rows and released adapters change what later dense prefix/audio queries see.

## Remaining ways target audio/prefix can diverge from native H3

Even with dense prefix/audio attention and `global_gate_mode=video_only`, the finished stack can still differ from native H3 through:

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

The Sol-H3 finding therefore does **not** currently overturn the learned-correction conclusion.

## New passive drift probe

`tools/h3_prefix_drift_probe.py` was added instead of another production knob. It compares one small native-H3 inference call with the finished VDN/Stage-B/Turbo call at the same top-level latent/timestep/context input and reports, per block and per grouped PackedLayout segment:

- cumulative block-input hidden drift;
- raw Q, K, V drift after the finished projection hooks;
- attention result drift immediately before `out_proj`;
- attention-module output drift;
- MLP output drift;
- final block-output drift;
- relative RMS, cosine similarity, and max absolute delta.

Dense snapshots are moved to CPU in BF16 and the tool fails closed above a small packed-row limit. It is intentionally a structural probe, not a production-resolution tensor recorder.

The hidden-width synthetic context skips the token refiner on purpose; real prompt-cache / one-step trainer smoke still exercises that path. This makes the 50 packed DiT blocks attributable without conflating the text preprocessor.

## Exact-prefix / exact-audio diagnostics: current status

The Sol-style tests separate into two levels:

### Attention topology level

For generated-audio queries and all other pre-video queries, native **dense attention topology** is already present in VDN. With `global_gate_mode=video_only`, the VDN-specific softmax gate is also removed from those rows. No new sparse-attention replacement is needed for tests A/B at this level.

### Whole native row-update level

A stronger test would recompute the underlying installed base-H3 attention or full block on the same current hidden state and replace only target-audio or all prefix rows while leaving the VDN video rows intact. That is not equivalent to destructive K/V masking and would preserve legitimate cross-modal context.

It is deliberately **not** added as a production/default mode yet. First use the passive drift trace to determine whether audio divergence begins in QKV/attention, MLP/AdaLN, or only accumulates after video-row divergence. If the trace points specifically at attention, add a narrowly scoped native-attention shadow diagnostic; if it points later, a full-block/boundary diagnostic is the correct next experiment instead.

## Trainer implications

### Canonical `xmarre/vdn-minimax-h3` audio-fix trainer

The canonical trainer remains logically sound for its current T2VA corpus:

- row ownership comes from Diffusers' own packed sequence indices;
- rollout is on-policy in the frozen VDN + Stage-B + Turbo student;
- the teacher audio target is dense released H3 evaluated at the exact current student sampler state;
- the teacher is not Sol sparse attention;
- the audio-fix sidecar applies only to target-audio indices.

However, its current packing helper explicitly builds T2VA with no conditioning/reference rows. Ref2VA/keyframe robustness therefore remains a deployment validation requirement rather than something the current training corpus proves.

### Direct Comfy INT8/ConvRot trainer

The direct trainer derives target-audio ownership from current Comfy `PackedLayout`, keeps the dense native-H3 teacher, and retains the strict training/inference forward-parity gate. The Sol-H3 finding does not justify lowering that gate or substituting a sparse teacher.

Spectrum and Progressive/Continuum remain outside the all-actual rollout and still require real deployment validation.

## Decision before further training

No long audio-fix run should start on the basis of this audit alone. The order remains:

1. pass the existing production INT8/ConvRot training/inference parity probe;
2. run the packed-row drift probe and locate where audio/prefix divergence starts;
3. only if the trace identifies an actionable native-row execution difference, add the corresponding exact preservation diagnostic and validate decoded media;
4. otherwise continue the learned generated-audio correction path, then validate real Spectrum + Progressive/Continuum media.
