"""Partitioned gathers preserve dense attention while sharing block-local K/V."""
import sys
from types import ModuleType, SimpleNamespace
import weakref

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import comfy.quant_ops
from vdn_h3 import partitioned_runtime
from vdn_h3.hybrid import VDNLayout, VDNState
from vdn_h3.partitioned_sequence import (
    PARTITIONED_PREFIX_KEY, PartitionedSequence, make_vdn_partitioned_external_contract,
)


@pytest.mark.parametrize("same_grid,anchor,sink", [
    (True, "none", 1), (True, "both", 5),
    (False, "none", 5), (False, "both", 1),
    (False, "rows", 5), (True, "columns", 5),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("native_carrier", ["source", "target"])
def test_partitioned_kv_storage_reuse_and_dense_frame_oracle(monkeypatch, same_grid, anchor, sink, dtype, native_carrier):
    if same_grid and native_carrier == "target":
        pytest.skip("target carrier requires heterogeneous frame grids")
    torch.manual_seed(815)
    heads, head_dim = 2, 4
    width = heads * head_dim
    plan = PartitionedSequence(
        video_start=sink, temporal=7, prefix_t=3,
        source_grid_h=2, source_grid_w=3,
        target_grid_h=2 if same_grid else 3, target_grid_w=3 if same_grid else 4,
        native_carrier_grid=native_carrier,
    )
    cfg = {"radius": 1, "chunk": 2, "anchor_frames": anchor,
           "enable_softmax_gate": False, "linear_enabled": False}
    state = VDNState("partitioned-kv-workspace", cfg, [SimpleNamespace()], heads, head_dim,
                     retain_buffers=True)
    frame_size = (3, 4) if native_carrier == "target" else (2, 3)
    native_rows = plan.native_rows_per_frame
    layout = VDNLayout(
        video_start=sink, video_end=sink + 7 * native_rows, num_frames=7, tokens_per_frame=native_rows,
        frame_size=frame_size, text_start=0, text_len=sink, seq_len=sink + 7 * native_rows,
        radius=1, chunk=2, anchor_frames=anchor,
    )
    x = torch.randn(plan.sequence_rows, width).to(dtype)
    projection = nn.Linear(width, 3 * width, bias=False).to(dtype).requires_grad_(False)
    raw = projection(x).split(width, dim=-1)
    expected_q = (raw[0].reshape(-1, heads, head_dim) + 0.25) * 1.5
    expected_k = (raw[1].reshape(-1, heads, head_dim) - 0.5) * 0.75
    expected_v = raw[2].reshape(-1, heads, head_dim) * 2.0
    storages = {"k": [], "v": []}
    scratch_refs = []
    preprocessing_calls = []

    def rope(q, k, *_args, **_kwargs):
        q.add_(0.25)
        k.sub_(0.5)

    def preprocess(_options, q, k, v, _heads):
        preprocessing_calls.append(True)
        return q * 1.5, k * 0.75, v * 2.0

    def dense(q, k, v, bias):
        return F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
            attn_mask=None if bias is None else bias.reshape(1, 1, 1, -1), scale=head_dim ** -0.5,
        )[0].transpose(0, 1)

    def attention(q, k, v, **kwargs):
        if kwargs["kind"] == "local":
            # Keep storage objects so allocator address recycling cannot disguise
            # fresh per-group allocations; do not keep the Tensor views alive.
            for name, tensor in (("k", k), ("v", v)):
                storages[name].append(tensor.untyped_storage())
                root = tensor._base if tensor._base is not None else tensor
                scratch_refs.append(weakref.ref(root))
        bias = None
        if kwargs["prefix_log_key_measure"] != 0.0:
            bias = q.new_zeros(k.shape[0], dtype=torch.float32)
            start, end = kwargs["prefix_k_range"]
            bias[start:end] = kwargs["prefix_log_key_measure"]
        return dense(q, k, v, bias)

    def weights_on(*_args, **_kwargs):
        assert all(ref() is None for ref in scratch_refs)
        assert state.runtime.current().retained_counts()["kv"] == 0
        return {}

    package = ModuleType("sol_h3")
    request = ModuleType("sol_h3.partitioned_request")
    request.partitioned_request_attention = attention
    monkeypatch.setitem(sys.modules, "sol_h3", package)
    monkeypatch.setitem(sys.modules, "sol_h3.partitioned_request", request)
    monkeypatch.setattr(comfy.quant_ops.ck, "rms_rope_split_half_", rope)
    monkeypatch.setattr("vdn_h3.softmax_provider.preprocess", preprocess)
    state.weights_on = weights_on
    values = {
        "state": state, "base_branch": SimpleNamespace(enable_text_state=False, short_conv=()),
        "qkv_proj": projection, "out_proj": lambda flat: flat.clone(),
        "q_norm": nn.RMSNorm(head_dim), "k_norm": nn.RMSNorm(head_dim),
        "heads": heads, "head_dim": head_dim, "block_index": 0, "cfg": cfg,
    }
    options = {
        PARTITIONED_PREFIX_KEY: plan.canonical_contract(),
        partitioned_runtime.VDN_EXTERNAL_SEQUENCE_KEY: make_vdn_partitioned_external_contract(plan),
    }
    with state.runtime.execution(), torch.no_grad():
        token = state._layout.set(layout)
        try:
            got = partitioned_runtime._partitioned_vdn_forward(
                None, values, x, torch.zeros(1, plan.sequence_rows, 1, 1), options,
            )
        finally:
            state._layout.reset(token)

    frame_rows = []
    cursor = sink
    for frame in range(plan.temporal):
        size = plan.target_rows if frame < plan.prefix_t else plan.source_rows
        frame_rows.append(list(range(cursor, cursor + size)))
        cursor += size
    full_bias = torch.zeros(plan.sequence_rows)
    full_bias[sink:sink + plan.prefix_t * plan.target_rows] = plan.prefix_log_key_measure
    expected = torch.empty_like(expected_q)
    if sink:
        expected[:sink] = dense(expected_q[:sink], expected_k, expected_v, full_bias)
    for frame, rows in enumerate(frame_rows):
        if anchor in ("rows", "both") and frame in (0, plan.temporal - 1):
            keys = list(range(plan.sequence_rows))
        else:
            lo, hi = layout.bounds[frame]
            frames = set(range(max(0, lo), min(plan.temporal - 1, hi) + 1))
            if anchor in ("columns", "both"):
                frames.update((0, plan.temporal - 1))
            keys = list(range(sink)) + [row for f in sorted(frames) for row in frame_rows[f]]
        expected[rows] = dense(expected_q[rows], expected_k[keys], expected_v[keys], full_bias[keys])
    torch.testing.assert_close(got, expected.reshape(-1, width), rtol=1e-5 if dtype == torch.float32 else 0.01,
                               atol=1e-6 if dtype == torch.float32 else 0.01)
    assert preprocessing_calls == [True]
    for allocations in storages.values():
        assert len(allocations) > 1
        assert len({storage.data_ptr() for storage in allocations}) == 1
