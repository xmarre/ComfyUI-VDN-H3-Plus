import importlib.util
from pathlib import Path

import pytest
import torch

from vdn_h3.boundary_witness import FeatureWitness
from vdn_h3.partitioned_linear import _frame_offsets, _variable_features, partitioned_linear_readout
from test_partitioned_linear import _branch, _inputs, _weights


class Sink:
    def __init__(self):
        self.tensors = {}

    def add(self, name, tensor):
        assert name not in self.tensors
        self.tensors[name] = tensor.detach().clone()


@pytest.mark.parametrize("suppress", [False, True])
@pytest.mark.parametrize("skip_ends", [False, True])
def test_actual_witness_is_output_neutral_and_anchor_indices_are_explicit(suppress, skip_ends):
    weights = _weights()
    branch = _branch(weights)
    sizes = ((4, 5),) * 4 + ((3, 4),) * 5
    offsets = _frame_offsets(sizes)
    x, q, k, v = _inputs(offsets[-1][-1])
    before = [t.clone() for t in (x, q, k, v)]
    kwargs = dict(frame_sizes=sizes, bounds=((0, 8),) * 9,
                  measure_scales=(0.6,) * 4 + (1.0,) * 5,
                  skip_ends=skip_ends, suppress_cross_grid_temporal_taps=suppress)
    reference = partitioned_linear_readout(branch, weights, x, q, k, v, **kwargs)
    sink = Sink()
    witness = FeatureWitness(sink, {"diagnostic_mode": "suppress_cross_grid_temporal_taps" if suppress else "normal"})
    observed = partitioned_linear_readout(branch, weights, x, q, k, v, observation=witness, **kwargs)
    assert torch.equal(reference, observed)
    assert all(torch.equal(a, b) for a, b in zip(before, (x, q, k, v)))
    assert sink.tensors
    assert witness.context["boundary_inner_frame"] == 4 - int(skip_ends)
    assert all(t.device.type == "cpu" for t in sink.tensors.values())
    assert bool(witness.cross_taps) != suppress


def test_offline_replay_matches_actual_legacy_preactivation():
    weights = _weights()
    branch = _branch(weights)
    sizes = ((4, 5),) * 4 + ((3, 4),) * 5
    offsets = _frame_offsets(sizes)
    _, q, k, v = _inputs(offsets[-1][-1])
    sink = Sink()
    witness = FeatureWitness(sink, {"diagnostic_mode": "normal"})
    _variable_features(branch, weights, q, k, v, sizes, offsets, observation=witness)
    path = Path(__file__).resolve().parents[1] / "tools" / "analyze_boundary_witness.py"
    spec = importlib.util.spec_from_file_location("offline_witness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.compare(sink.tensors, witness.context)
    assert all(row["finite"] for row in report)
    assert max(row["cpu_legacy_vs_observed_max_abs"] for row in report) < 1e-6
    assert max(row["ordering_rms"] for row in report) > 1e-5


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("conv", [("q",), ("k",), ("v",), ("q", "k", "v")])
def test_feature_observation_preserves_actual_precision_and_requested_conv_branches(dtype, conv):
    weights = _weights()
    weights["short_conv.q_sp.weight"] = weights["short_conv.k_sp.weight"].clone()
    weights["short_conv.q_tm.weight"] = weights["short_conv.k_tm.weight"].clone()
    weights = {key: value.to(dtype) for key, value in weights.items()}
    branch = _branch(weights)
    branch.short_conv = conv
    sizes = ((4, 5),) * 4 + ((3, 4),) * 5
    offsets = _frame_offsets(sizes)
    _, q, k, v = _inputs(offsets[-1][-1])
    q, k, v = [item.to(dtype) for item in (q, k, v)]
    reference = _variable_features(branch, weights, q, k, v, sizes, offsets)
    sink = Sink()
    witness = FeatureWitness(sink, {"diagnostic_mode": "normal"})
    observed = _variable_features(branch, weights, q, k, v, sizes, offsets, observation=witness)
    assert all(torch.equal(old, new) for old, new in zip(reference, observed))
    assert tuple(witness.context["features"]) == conv
    assert all(item.dtype == dtype for item in sink.tensors.values())
