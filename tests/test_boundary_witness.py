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
