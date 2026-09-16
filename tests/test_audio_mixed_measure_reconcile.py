from types import SimpleNamespace

import pytest
import torch

from vdn_h3.mixed_measure_epilogue import ExternalSoftmaxEpilogueCapability


class CountingProjection(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(width), requires_grad=False)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.nn.functional.linear(x, self.weight)


class State:
    def __init__(self, branch=True, **cfg):
        self.name = "audio-mixed-grid-test"
        self.cfg = {
            "enable_softmax_gate": True,
            "global_gate_mode": "checkpoint",
            "audio_video_context_strength": 1.0,
            "conditioning_video_context_strength": 1.0,
            **cfg,
        }
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
        return {
            "softmax_gate.up.weight": torch.zeros(2, 8, device=device, dtype=dtype),
            "softmax_gate.up.bias": torch.zeros(2, device=device, dtype=dtype),
        }


def options():
    return {
        "vdn_h3_external_sequence_v1": {
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
    }


def test_video_only_gate_scope_is_preserved_by_external_mixed_grid_epilogue():
    state = State(global_gate_mode="video_only")
    projection = CountingProjection(8)
    capability = ExternalSoftmaxEpilogueCapability(
        state, 0, projection, heads=2, head_dim=4
    )
    x = torch.randn(24, 8)
    rope = torch.zeros(1, 24, 1, 1)
    softmax = torch.randn(24, 2, 4)

    bound = capability.prepare(x, rope, options(), 0)
    got = bound.apply(softmax, x)

    expected = softmax.clone()
    expected[6:] *= 0.5
    torch.testing.assert_close(got, expected.reshape(24, 8))
    assert state.weight_calls == 1
    assert projection.calls == 1
    assert bound.gate_calls == 1
    assert bound.projection_calls == 1
    assert bound.completed is True


@pytest.mark.parametrize(
    "name",
    ["audio_video_context_strength", "conditioning_video_context_strength"],
)
def test_external_mixed_grid_fails_closed_for_unrepresentable_context_diagnostics(name):
    state = State(**{name: 0.5})
    capability = ExternalSoftmaxEpilogueCapability(
        state, 0, CountingProjection(8), heads=2, head_dim=4
    )
    x = torch.randn(24, 8)
    rope = torch.zeros(1, 24, 1, 1)

    with pytest.raises(RuntimeError, match=name):
        capability.prepare(x, rope, options(), 0)


def test_branchless_block_ignores_branch_only_audio_diagnostics_like_normal_vdn_forward():
    state = State(
        branch=False,
        audio_video_context_strength=0.0,
        conditioning_video_context_strength=0.0,
    )
    projection = CountingProjection(8)
    capability = ExternalSoftmaxEpilogueCapability(
        state, 0, projection, heads=2, head_dim=4
    )
    x = torch.randn(24, 8)
    rope = torch.zeros(1, 24, 1, 1)
    softmax = torch.randn(24, 2, 4)

    bound = capability.prepare(x, rope, options(), 0)
    got = bound.apply(softmax, x)

    torch.testing.assert_close(got, softmax.reshape(24, 8))
    assert state.weight_calls == 0
    assert projection.calls == 1
    assert bound.gate_calls == 0
