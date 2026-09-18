from types import SimpleNamespace

import torch

from vdn_h3.branch import LinearBranch
from vdn_h3.partitioned_linear import partitioned_frame_contract, partitioned_linear_readout


def _weights(*, hidden=5, heads=2, head_dim=3, rank=4, seed=91):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    channels = heads * head_dim

    def randn(shape, scale=0.1):
        return torch.randn(shape, generator=generator, dtype=torch.float32) * scale

    return {
        "short_conv.k_sp.weight": randn((channels, 1, 5, 5), 0.05),
        "short_conv.k_tm.weight": randn((channels, 1, 5), 0.05),
        "short_conv.v_sp.weight": randn((channels, 1, 5, 5), 0.05),
        "short_conv.v_tm.weight": randn((channels, 1, 5), 0.05),
        "beta_proj.weight": randn((heads, hidden)),
        "alpha.down.weight": randn((rank, hidden)),
        "alpha.up.weight": randn((channels, rank)),
        "alpha.dt_bias": randn((channels,), 0.05),
        "alpha.A_log": randn((heads,), 0.05),
        "output_gate.down.weight": randn((rank, hidden)),
        "output_gate.up.weight": randn((channels, rank)),
        "output_gate.up.bias": randn((channels,), 0.05),
        "norm.weight": torch.ones(head_dim, dtype=torch.float32),
    }


def _branch(weights, *, heads=2, head_dim=3):
    return LinearBranch(
        weights,
        heads,
        head_dim,
        delta_rule="vdn_solve",
        bridge="alpha",
        a_fp32=True,
        short_conv=("k", "v"),
        enable_text_state=False,
    )


def _inputs(rows, *, hidden=5, heads=2, head_dim=3, seed=17):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((rows, hidden), generator=generator, dtype=torch.float32)
    q = torch.randn((rows, heads, head_dim), generator=generator, dtype=torch.float32)
    k = torch.randn((rows, heads, head_dim), generator=generator, dtype=torch.float32)
    v = torch.randn((rows, heads, head_dim), generator=generator, dtype=torch.float32)
    return x, q, k, v


def test_variable_grid_linear_reduces_to_released_readout_on_uniform_grid():
    weights = _weights()
    branch = _branch(weights)
    frames = 4
    grid = (2, 2)
    rows_per_frame = grid[0] * grid[1]
    rows = frames * rows_per_frame
    bounds = ((0, 1), (0, 2), (1, 3), (2, 3))
    x, q, k, v = _inputs(rows)

    released = branch._readout(
        weights,
        x,
        (q, k, v),
        frames,
        rows_per_frame,
        bounds,
        grid,
        None,
        None,
        None,
    )
    partitioned = partitioned_linear_readout(
        branch,
        weights,
        x,
        q,
        k,
        v,
        frame_sizes=(grid,) * frames,
        bounds=bounds,
        measure_scales=(1.0,) * frames,
    )

    assert partitioned.shape == released.shape
    assert torch.allclose(partitioned, released, rtol=2e-5, atol=2e-5)


def test_variable_grid_linear_runs_target_prefix_and_source_suffix_with_physical_measure():
    weights = _weights(seed=113)
    branch = _branch(weights)
    plan = SimpleNamespace(
        target_grid_h=2,
        target_grid_w=2,
        source_grid_h=1,
        source_grid_w=2,
        prefix_t=2,
        temporal=4,
        target_rows=4,
        source_rows=2,
    )
    frame_sizes, measures = partitioned_frame_contract(plan)
    assert frame_sizes == ((2, 2), (2, 2), (1, 2), (1, 2))
    assert measures == (0.5, 0.5, 1.0, 1.0)
    rows = sum(height * width for height, width in frame_sizes)
    x, q, k, v = _inputs(rows, seed=29)
    bounds = ((0, 1), (0, 2), (1, 3), (2, 3))

    weighted = partitioned_linear_readout(
        branch,
        weights,
        x,
        q,
        k,
        v,
        frame_sizes=frame_sizes,
        bounds=bounds,
        measure_scales=measures,
    )
    unweighted = partitioned_linear_readout(
        branch,
        weights,
        x,
        q,
        k,
        v,
        frame_sizes=frame_sizes,
        bounds=bounds,
        measure_scales=(1.0,) * plan.temporal,
    )

    assert weighted.shape == (rows, branch.num_heads * branch.head_dim)
    assert torch.isfinite(weighted).all()
    assert not torch.allclose(weighted, unweighted, rtol=0.0, atol=0.0)


def test_variable_grid_linear_skip_ends_matches_released_anchor_contract_shape():
    weights = _weights(seed=131)
    branch = _branch(weights)
    frame_sizes = ((2, 2), (2, 2), (1, 2), (1, 2))
    measures = (0.5, 0.5, 1.0, 1.0)
    bounds = ((0, 1), (0, 2), (1, 3), (2, 3))
    rows = sum(height * width for height, width in frame_sizes)
    x, q, k, v = _inputs(rows, seed=41)

    result = partitioned_linear_readout(
        branch,
        weights,
        x,
        q,
        k,
        v,
        frame_sizes=frame_sizes,
        bounds=bounds,
        measure_scales=measures,
        skip_ends=True,
    )
    first_rows = frame_sizes[0][0] * frame_sizes[0][1]
    last_rows = frame_sizes[-1][0] * frame_sizes[-1][1]
    assert torch.count_nonzero(result[:first_rows]) == 0
    assert torch.count_nonzero(result[-last_rows:]) == 0
    assert torch.isfinite(result[first_rows:-last_rows]).all()


def test_variable_grid_linear_component_recorder_is_observational():
    weights = _weights(seed=211)
    branch = _branch(weights)
    frame_sizes = ((2, 2), (2, 2), (1, 2), (1, 2))
    measures = (0.5, 0.5, 1.0, 1.0)
    bounds = ((0, 1), (0, 2), (1, 3), (2, 3))
    rows = sum(height * width for height, width in frame_sizes)
    x, q, k, v = _inputs(rows, seed=47)
    originals = tuple(t.clone() for t in (x, q, k, v))
    recorded = {}

    def record(name, elapsed_s):
        recorded[name] = recorded.get(name, 0.0) + float(elapsed_s)

    reference = partitioned_linear_readout(
        branch,
        weights,
        x,
        q,
        k,
        v,
        frame_sizes=frame_sizes,
        bounds=bounds,
        measure_scales=measures,
    )
    observed = partitioned_linear_readout(
        branch,
        weights,
        x,
        q,
        k,
        v,
        frame_sizes=frame_sizes,
        bounds=bounds,
        measure_scales=measures,
        record_component=record,
    )

    assert torch.equal(observed, reference)
    for before, after in zip(originals, (x, q, k, v), strict=True):
        assert torch.equal(before, after)
    expected = {
        "vdn_linear_features_host_wall_s",
        "vdn_linear_statistics_host_wall_s",
        "vdn_linear_scans_host_wall_s",
        "vdn_linear_gather_host_wall_s",
        "vdn_linear_output_host_wall_s",
        "vdn_linear_api_host_wall_s",
    }
    assert expected.issubset(recorded)
    assert all(recorded[name] >= 0.0 for name in expected)
