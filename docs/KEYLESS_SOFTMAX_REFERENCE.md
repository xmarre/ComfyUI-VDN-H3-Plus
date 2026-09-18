# MiniMax H3 Keyless VDN softmax reference

This branch adds only the **softmax/window half** of the VDN architecture for
`h3_keyless_core50_v1`.

Released VDN-H3 checkpoints remain QKV-trained and are still rejected by the
ordinary Apply VDN nodes. Their learned linear complement consumes raw projected
Q/K/V, including raw K, and cannot be preserved by substituting V or by dropping
that branch.

The Keyless reference node therefore does **not** load a released VDN checkpoint,
VDN adapters, softmax-gate weights, or the learned linear complement. It exists to
establish the row-domain and provider contract that a future Keyless-native VDN
stage can reuse.

For every grouped local window the reference provider:

1. obtains the exact restricted VDN row domain from the existing grouped-window
   geometry;
2. selects raw V exactly once;
3. transports the same selection into Keyless routing positions through
   `RoutingSpecV1.select_value_rows`;
4. materializes `route(V_selected)` from those selected V rows;
5. attends with `Q_requested / route(V_selected) / V_selected`;
6. preserves VDN provider-v4 query-position maps for a downstream rectangular
   attention backend.

Global and anchor rows retain VDN's dense semantics. Full-coverage windows use the
ordinary Keyless dense fallback.

The reference path deliberately fails closed for currently unreviewed compositions:
external/reduced query or value domains, external routing-position domains,
Keyless masks/log measures, and the old `vdn_attention_preprocess_v1` hook.
Routing-only transforms such as Keyless Untwist belong in
`minimax_h3_keyless_routing_preprocessors_v1`; applying the native VDN Q/K/V
preprocessor again would risk transforming retrieval V or double-applying routing
semantics.

This is structural/reference infrastructure, not a released Keyless VDN model.
Production compatibility still requires a separately identified Keyless-native
checkpoint/schema, retraining or distillation of the learned linear branch,
preservation/revalidation of the trained softmax gate and adapters, and decoded
audio/video/runtime evidence. The reference node must not be used to claim parity
with an existing released VDN checkpoint.
