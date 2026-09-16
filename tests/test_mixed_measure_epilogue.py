from types import SimpleNamespace

import pytest
import torch

from vdn_h3.mixed_measure_epilogue import (
    EPILOGUE_KEY,
    EPILOGUE_RECEIPTS_KEY,
    ExternalSoftmaxEpilogueCapability,
    attach_external_softmax_epilogue,
)


class CountingProjection(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(width), requires_grad=False)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.nn.functional.linear(x, self.weight)


class State:
    def __init__(self, *, gated=True, branch=True):
        self.name = "test"
        self.cfg = {"enable_softmax_gate": gated}
        self.branches = [object() if branch else None]
        self.managed_weights = None
        self._layout = SimpleNamespace(
            seq_len=18,
            video_start=6,
            num_frames=2,
            tokens_per_frame=6,
        )
        self.weight_calls = 0

    @property
    def layout(self):
        return self._layout

    def weights_on(self, index, device, dtype):
        self.weight_calls += 1
        assert index == 0
        # Two heads, each receives sigmoid(0)=0.5.
        return {
            "softmax_gate.up.weight": torch.zeros(2, 8, device=device, dtype=dtype),
            "softmax_gate.up.bias": torch.zeros(2, device=device, dtype=dtype),
        }


def options(**overrides):
    contract = {
        "api": 2,
        "mode": "dense_gate_no_linear",
        "topology": "mixed_grid_low_suffix",
        "native_sequence_rows": 18,
        "sequence_rows": 24,
        "video_start": 6,
        "temporal": 2,
        "prefix_t": 1,
        "source_rows_per_frame": 6,
        "prefix_rows_per_frame": 12,
    }
    contract.update(overrides)
    return {"vdn_h3_external_sequence_v1": contract}


def test_capability_attaches_to_concrete_vdn_forward_once():
    state = State()
    projection = CountingProjection(8)

    def forward(*args, **kwargs):
        raise AssertionError("attachment must not execute the forward")

    capability = attach_external_softmax_epilogue(
        forward, state, 0, projection, heads=2, head_dim=4
    )
    assert getattr(forward, EPILOGUE_KEY) is capability
    assert capability.state is state
    assert capability.out_proj is projection
    with pytest.raises(RuntimeError, match="already attached"):
        attach_external_softmax_epilogue(
            forward, state, 0, projection, heads=2, head_dim=4
        )


def test_nontrivial_gate_and_projection_are_applied_exactly_once():
    state = State(gated=True, branch=True)
    projection = CountingProjection(8)
    capability = ExternalSoftmaxEpilogueCapability(state, 0, projection, heads=2, head_dim=4)
    x = torch.randn(24, 8)
    rope = torch.zeros(1, 24, 1, 1)
    softmax = torch.randn(24, 2, 4)

    bound = capability.prepare(x, rope, options(), 0)
    got = bound.apply(softmax, x)
    expected = (softmax * 0.5).reshape(24, 8)
    torch.testing.assert_close(got, expected)
    assert state.weight_calls == 1
    assert projection.calls == 1
    assert bound.gate_calls == 1
    assert bound.projection_calls == 1
    assert bound.completed is True
    fields = dict(bound.receipt_fields())
    assert fields["vdn_gate_calls"] == 1
    assert fields["vdn_projection_calls"] == 1
    assert fields["vdn_completed"] is True

    with pytest.raises(RuntimeError, match="more than once"):
        bound.apply(softmax, x)
    assert projection.calls == 1


def test_completion_receipt_is_forwarded_only_after_success():
    state = State(gated=True, branch=True)
    projection = CountingProjection(8)
    capability = ExternalSoftmaxEpilogueCapability(state, 0, projection, heads=2, head_dim=4)
    x = torch.randn(24, 8)
    rope = torch.zeros(1, 24, 1, 1)
    softmax = torch.randn(24, 2, 4)
    receipt_sink = []
    prepared_options = options()
    prepared_options[EPILOGUE_RECEIPTS_KEY] = receipt_sink

    bound = capability.prepare(x, rope, prepared_options, 0)
    assert receipt_sink == []
    bound.apply(softmax, x)

    assert len(receipt_sink) == 1
    block_index, receipt_fields = receipt_sink[0]
    assert block_index == 0
    fields = dict(receipt_fields)
    assert fields["vdn_owner_generation"] == capability.owner_generation
    assert fields["vdn_config_digest"] == capability.config_digest
    assert fields["vdn_weight_owner_digest"] == capability.weight_owner_digest
    assert fields["vdn_gate_expected"] is True
    assert fields["vdn_gate_calls"] == 1
    assert fields["vdn_projection_calls"] == 1
    assert fields["vdn_completed"] is True

    with pytest.raises(RuntimeError, match="more than once"):
        bound.apply(softmax, x)
    assert len(receipt_sink) == 1


def test_failed_apply_does_not_forward_completion_receipt():
    state = State()
    projection = CountingProjection(8)
    capability = ExternalSoftmaxEpilogueCapability(state, 0, projection, heads=2, head_dim=4)
    x = torch.randn(24, 8)
    receipt_sink = []
    prepared_options = options()
    prepared_options[EPILOGUE_RECEIPTS_KEY] = receipt_sink
    bound = capability.prepare(x, torch.zeros(1, 24, 1, 1), prepared_options, 0)

    with pytest.raises(RuntimeError, match="softmax tensor"):
        bound.apply(torch.randn(24, 8), x)
    assert receipt_sink == []
    assert projection.calls == 0


def test_prepare_rejects_invalid_receipt_sink_before_binding():
    state = State()
    capability = ExternalSoftmaxEpilogueCapability(state, 0, CountingProjection(8), heads=2, head_dim=4)
    prepared_options = options()
    prepared_options[EPILOGUE_RECEIPTS_KEY] = ()
    with pytest.raises(RuntimeError, match="receipt sink must be a list"):
        capability.prepare(
            torch.randn(24, 8),
            torch.zeros(1, 24, 1, 1),
            prepared_options,
            0,
        )


def test_branchless_block_uses_native_projection_without_gate_or_weight_load():
    state = State(gated=True, branch=False)
    projection = CountingProjection(8)
    capability = ExternalSoftmaxEpilogueCapability(state, 0, projection, heads=2, head_dim=4)
    x = torch.randn(24, 8)
    rope = torch.zeros(1, 24, 1, 1)
    softmax = torch.randn(24, 2, 4)

    bound = capability.prepare(x, rope, options(), 0)
    got = bound.apply(softmax, x)
    torch.testing.assert_close(got, softmax.reshape(24, 8))
    assert bound.gate_expected is False
    assert bound.gate_calls == 0
    assert state.weight_calls == 0
    assert projection.calls == 1


def test_disabled_gate_still_projects_once_and_does_not_read_gate_weights():
    state = State(gated=False, branch=True)
    projection = CountingProjection(8)
    capability = ExternalSoftmaxEpilogueCapability(state, 0, projection, heads=2, head_dim=4)
    x = torch.randn(24, 8)
    rope = torch.zeros(1, 24, 1, 1)
    softmax = torch.randn(24, 2, 4)

    bound = capability.prepare(x, rope, options(), 0)
    got = bound.apply(softmax, x)
    torch.testing.assert_close(got, softmax.reshape(24, 8))
    assert state.weight_calls == 0
    assert bound.gate_calls == 0
    assert projection.calls == 1


def test_prepare_rejects_wrong_block_stale_geometry_and_missing_execution_lifetime():
    state = State()
    capability = ExternalSoftmaxEpilogueCapability(state, 0, CountingProjection(8), heads=2, head_dim=4)
    x = torch.randn(24, 8)
    rope = torch.zeros(1, 24, 1, 1)

    with pytest.raises(RuntimeError, match="wrong block"):
        capability.prepare(x, rope, options(), 1)
    with pytest.raises(RuntimeError, match="does not match"):
        capability.prepare(x, rope, options(sequence_rows=25), 0)
    with pytest.raises(RuntimeError, match="explicit RoPE"):
        capability.prepare(x, torch.zeros(1, 23, 1, 1), options(), 0)

    state._layout = None
    with pytest.raises(RuntimeError, match="outside the VDN execution lifetime"):
        capability.prepare(x, rope, options(), 0)


def test_prepare_rejects_mutated_branch_weight_owner_and_config():
    x = torch.randn(24, 8)
    rope = torch.zeros(1, 24, 1, 1)

    state = State()
    capability = ExternalSoftmaxEpilogueCapability(state, 0, CountingProjection(8), heads=2, head_dim=4)
    state.branches[0] = object()
    with pytest.raises(RuntimeError, match="weight ownership changed"):
        capability.prepare(x, rope, options(), 0)

    state = State()
    capability = ExternalSoftmaxEpilogueCapability(state, 0, CountingProjection(8), heads=2, head_dim=4)
    state.managed_weights = object()
    with pytest.raises(RuntimeError, match="weight ownership changed"):
        capability.prepare(x, rope, options(), 0)

    state = State()
    capability = ExternalSoftmaxEpilogueCapability(state, 0, CountingProjection(8), heads=2, head_dim=4)
    state.cfg["enable_softmax_gate"] = False
    with pytest.raises(RuntimeError, match="configuration changed"):
        capability.prepare(x, rope, options(), 0)


def test_apply_rejects_wrong_softmax_shape_before_projection():
    state = State()
    projection = CountingProjection(8)
    capability = ExternalSoftmaxEpilogueCapability(state, 0, projection, heads=2, head_dim=4)
    x = torch.randn(24, 8)
    bound = capability.prepare(x, torch.zeros(1, 24, 1, 1), options(), 0)
    with pytest.raises(RuntimeError, match="softmax tensor"):
        bound.apply(torch.randn(24, 8), x)
    assert projection.calls == 0
