from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vdn_h3.branch import LinearBranch
from vdn_h3.partitioned_linear import (
    _batched_frame_statistics,
    _batched_output_readout,
    _framewise_output_reference,
    _framewise_statistics_reference,
    _h3_axis_coordinates,
    _heterogeneous_conv_features,
    _heterogeneous_conv_features_reference,
    _map_temporal_neighbor,
    partitioned_frame_contract,
    partitioned_linear_readout,
)


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


def test_batched_heterogeneous_short_conv_matches_scalar_reference():
    # Two long domains mirror exact-prefix continuation while keeping the CPU
    # oracle small enough for hosted CI.
    frame_sizes = ((4, 5),) * 5 + ((3, 4),) * 9
    offsets = []
    cursor = 0
    for grid_h, grid_w in frame_sizes:
        next_cursor = cursor + grid_h * grid_w
        offsets.append((cursor, next_cursor))
        cursor = next_cursor
    offsets = tuple(offsets)

    generator = torch.Generator(device="cpu").manual_seed(701)
    tokens = torch.randn((cursor, 2, 3), generator=generator, dtype=torch.float32)
    channels = 6
    spatial = torch.randn((channels, 1, 5, 5), generator=generator, dtype=torch.float32) * 0.05
    temporal = torch.randn((channels, 1, 5), generator=generator, dtype=torch.float32) * 0.05

    reference = _heterogeneous_conv_features_reference(
        tokens,
        spatial,
        temporal,
        frame_sizes,
        offsets,
        l2norm=True,
    )
    batched = _heterogeneous_conv_features(
        tokens,
        spatial,
        temporal,
        frame_sizes,
        offsets,
        l2norm=True,
    )

    assert batched.shape == reference.shape
    assert torch.allclose(batched, reference, rtol=2e-5, atol=2e-5)


def test_batched_heterogeneous_suppression_matches_scalar_reference_and_stats():
    frame_sizes = ((3, 4),) * 4 + ((2, 3),) * 6
    offsets = []
    cursor = 0
    for grid_h, grid_w in frame_sizes:
        next_cursor = cursor + grid_h * grid_w
        offsets.append((cursor, next_cursor))
        cursor = next_cursor
    offsets = tuple(offsets)

    generator = torch.Generator(device="cpu").manual_seed(702)
    tokens = torch.randn((cursor, 1, 2), generator=generator, dtype=torch.float32)
    spatial = torch.randn((2, 1, 5, 5), generator=generator, dtype=torch.float32) * 0.05
    temporal = torch.randn((2, 1, 5), generator=generator, dtype=torch.float32) * 0.05
    reference_stats = {}
    batched_stats = {}

    reference = _heterogeneous_conv_features_reference(
        tokens,
        spatial,
        temporal,
        frame_sizes,
        offsets,
        l2norm=False,
        suppress_cross_grid_temporal_taps=True,
        diagnostic_stats=reference_stats,
    )
    batched = _heterogeneous_conv_features(
        tokens,
        spatial,
        temporal,
        frame_sizes,
        offsets,
        l2norm=False,
        suppress_cross_grid_temporal_taps=True,
        diagnostic_stats=batched_stats,
    )

    assert torch.allclose(batched, reference, rtol=2e-5, atol=2e-5)
    assert batched_stats == reference_stats
    assert batched_stats["suppressed_taps"] > 0


@pytest.mark.parametrize("suppress", [False, True])
def test_batched_noncontiguous_same_grid_runs_match_scalar_reference(suppress):
    # A grid reappears after a B run. Temporal taps from the first A run into
    # the second A run are same-grid but not same-run, so they must not be
    # dropped by the domain-batched implementation.
    frame_sizes = ((3, 4), (3, 4), (2, 3), (3, 4), (3, 4))
    offsets = []
    cursor = 0
    for grid_h, grid_w in frame_sizes:
        next_cursor = cursor + grid_h * grid_w
        offsets.append((cursor, next_cursor))
        cursor = next_cursor
    offsets = tuple(offsets)

    generator = torch.Generator(device="cpu").manual_seed(706)
    tokens = torch.randn((cursor, 1, 2), generator=generator, dtype=torch.float32)
    spatial = torch.randn((2, 1, 5, 5), generator=generator, dtype=torch.float32) * 0.05
    temporal = torch.randn((2, 1, 5), generator=generator, dtype=torch.float32) * 0.05
    reference_stats = {}
    batched_stats = {}

    reference = _heterogeneous_conv_features_reference(
        tokens,
        spatial,
        temporal,
        frame_sizes,
        offsets,
        l2norm=False,
        suppress_cross_grid_temporal_taps=suppress,
        diagnostic_stats=reference_stats,
    )
    batched = _heterogeneous_conv_features(
        tokens,
        spatial,
        temporal,
        frame_sizes,
        offsets,
        l2norm=False,
        suppress_cross_grid_temporal_taps=suppress,
        diagnostic_stats=batched_stats,
    )

    assert torch.allclose(batched, reference, rtol=2e-5, atol=2e-5)
    assert batched_stats == reference_stats


def test_batched_heterogeneous_statistics_match_scalar_reference():
    weights = _weights(seed=703)
    branch = _branch(weights)
    frame_sizes = ((4, 5),) * 5 + ((3, 4),) * 9
    offsets = []
    cursor = 0
    for grid_h, grid_w in frame_sizes:
        next_cursor = cursor + grid_h * grid_w
        offsets.append((cursor, next_cursor))
        cursor = next_cursor
    offsets = tuple(offsets)

    x, _q, key, value = _inputs(cursor, seed=704)
    beta_rows = torch.sigmoid(torch.nn.functional.linear(x, weights["beta_proj.weight"]))
    measures = tuple(
        0.55 + 0.45 * (index / max(1, len(frame_sizes) - 1))
        for index in range(len(frame_sizes))
    )

    reference = _framewise_statistics_reference(
        branch,
        x,
        key,
        value,
        beta_rows,
        offsets,
        measures,
    )
    batched = _batched_frame_statistics(
        branch,
        x,
        key,
        value,
        beta_rows,
        frame_sizes,
        offsets,
        measures,
    )

    for actual, expected in zip(batched, reference, strict=True):
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batched_statistics_preserve_scalar_measure_opmath(dtype):
    weights = _weights(seed=707)
    branch = _branch(weights)
    frame_sizes = ((2, 3),) * 3 + ((2, 2),) * 2
    offsets = []
    cursor = 0
    for grid_h, grid_w in frame_sizes:
        next_cursor = cursor + grid_h * grid_w
        offsets.append((cursor, next_cursor))
        cursor = next_cursor
    offsets = tuple(offsets)

    x, _q, key, value = _inputs(cursor, seed=708)
    beta_rows = torch.sigmoid(torch.nn.functional.linear(x, weights["beta_proj.weight"])).to(dtype)
    key = key.to(dtype)
    value = value.to(dtype)
    x = x.to(dtype)
    # Non-binary fractions expose premature fp16/bf16 quantization of the
    # batched measure tensor.
    measures = (1.0 / 3.0, 0.1, 0.7, 0.55, 0.95)

    reference = _framewise_statistics_reference(
        branch,
        x,
        key,
        value,
        beta_rows,
        offsets,
        measures,
    )
    batched = _batched_frame_statistics(
        branch,
        x,
        key,
        value,
        beta_rows,
        frame_sizes,
        offsets,
        measures,
    )

    for actual, expected in zip(batched, reference, strict=True):
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)


def test_batched_heterogeneous_output_matches_scalar_reference():
    frame_sizes = ((4, 5),) * 5 + ((3, 4),) * 9
    offsets = []
    cursor = 0
    for grid_h, grid_w in frame_sizes:
        next_cursor = cursor + grid_h * grid_w
        offsets.append((cursor, next_cursor))
        cursor = next_cursor
    offsets = tuple(offsets)

    generator = torch.Generator(device="cpu").manual_seed(705)
    heads = 2
    head_dim = 3
    frames = len(frame_sizes)
    query = torch.randn((cursor, heads, head_dim), generator=generator, dtype=torch.float32)
    linear_state = torch.randn(
        (frames, heads, head_dim, head_dim),
        generator=generator,
        dtype=torch.float32,
    )
    gate = torch.sigmoid(
        torch.randn((cursor, heads * head_dim), generator=generator, dtype=torch.float32)
    )
    norm_weight = torch.randn((head_dim,), generator=generator, dtype=torch.float32).abs() + 0.5
    eps = 1e-6

    reference = _framewise_output_reference(
        query,
        linear_state,
        gate,
        offsets,
        norm_weight,
        eps,
    )
    batched = _batched_output_readout(
        query,
        linear_state,
        gate,
        frame_sizes,
        offsets,
        norm_weight,
        eps,
    )

    assert batched.shape == reference.shape
    assert torch.allclose(batched, reference, rtol=2e-5, atol=2e-5)


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
    cuda_components = []

    def record(name, elapsed_s):
        recorded[name] = recorded.get(name, 0.0) + float(elapsed_s)

    @contextmanager
    def cuda_span(name):
        cuda_components.append(name)
        yield

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
        cuda_span=cuda_span,
    )

    assert torch.equal(observed, reference)
    for before, after in zip(originals, (x, q, k, v), strict=True):
        assert torch.equal(before, after)
    expected = {
        "vdn_linear_features_host_wall_s",
        "vdn_linear_statistics_host_wall_s",
        "vdn_linear_scans_host_wall_s",
        "vdn_linear_gate_host_wall_s",
        "vdn_linear_gather_host_wall_s",
        "vdn_linear_epsilon_scalar_host_wall_s",
        "vdn_linear_output_host_wall_s",
        "vdn_linear_api_host_wall_s",
    }
    assert expected.issubset(recorded)
    assert all(recorded[name] >= 0.0 for name in expected)
    assert cuda_components == [
        "vdn_linear_features",
        "vdn_linear_statistics",
        "vdn_linear_scans",
        "vdn_linear_gate",
        "vdn_linear_gather",
        "vdn_linear_epsilon_scalar",
        "vdn_linear_output",
    ]


def test_cross_grid_temporal_map_follows_h3_physical_rope_lattice():
    source_hw = (28, 38)
    target_hw = (20, 27)
    y = _h3_axis_coordinates(*source_hw, 0, device=torch.device("cpu"))
    x = _h3_axis_coordinates(*source_hw, 1, device=torch.device("cpu"))
    source = (y[:, None] + 0.25 * x[None, :]).unsqueeze(0).unsqueeze(0)

    mapped = _map_temporal_neighbor(source, target_hw)

    target_y = _h3_axis_coordinates(*target_hw, 0, device=torch.device("cpu"))
    target_x = _h3_axis_coordinates(*target_hw, 1, device=torch.device("cpu"))
    expected = (target_y[:, None] + 0.25 * target_x[None, :]).unsqueeze(0).unsqueeze(0)

    # The first target row extends slightly beyond the denser source lattice for
    # this aligned H3 geometry and is deliberately border-clamped. All interior
    # samples must represent the exact H3 physical coordinates.
    assert torch.allclose(mapped[..., 1:-1, 1:-1], expected[..., 1:-1, 1:-1], rtol=1e-5, atol=2e-5)

    half_pixel = torch.nn.functional.interpolate(
        source,
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    assert torch.max(torch.abs(half_pixel[..., 1:-1, 1:-1] - expected[..., 1:-1, 1:-1])) > 0.1


def test_cross_grid_temporal_map_identity_returns_original_tensor():
    source = torch.randn(1, 3, 4, 5)
    assert _map_temporal_neighbor(source, (4, 5)) is source


def test_h3_axis_coordinates_match_pinned_comfy_physical_rope_grid():
    from comfy.ldm.minimax.model import _axis_from_sqrt_area

    for grid_h, grid_w in ((20, 27), (28, 38), (2, 3)):
        latent_h = grid_h * 2
        latent_w = grid_w * 2
        sqrt_area = float(latent_h * latent_w) ** 0.5
        expected_y = _axis_from_sqrt_area(latent_h, 2, sqrt_area).to(torch.float32)
        expected_x = _axis_from_sqrt_area(latent_w, 2, sqrt_area).to(torch.float32)
        actual_y = _h3_axis_coordinates(
            grid_h,
            grid_w,
            0,
            device=torch.device("cpu"),
        )
        actual_x = _h3_axis_coordinates(
            grid_h,
            grid_w,
            1,
            device=torch.device("cpu"),
        )
        assert torch.allclose(actual_y, expected_y, rtol=0.0, atol=2e-6)
        assert torch.allclose(actual_x, expected_x, rtol=0.0, atol=2e-6)


def test_cross_grid_temporal_diagnostic_suppresses_only_cross_domain_taps():
    # Three 2x2 target-grid frames followed by three 1x2 source-grid frames.
    # A five-tap temporal kernel has radius two, so the outermost frames never
    # see the other grid domain while frames adjacent to the boundary do.
    frame_sizes = ((2, 2), (2, 2), (2, 2), (1, 2), (1, 2), (1, 2))
    offsets = ((0, 4), (4, 8), (8, 12), (12, 14), (14, 16), (16, 18))
    tokens = torch.arange(18, dtype=torch.float32).view(18, 1, 1) + 1.0
    spatial = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
    spatial[0, 0, 2, 2] = 1.0
    temporal = torch.ones((1, 1, 5), dtype=torch.float32)

    normal = _heterogeneous_conv_features(
        tokens,
        spatial,
        temporal,
        frame_sizes,
        offsets,
        l2norm=False,
    )
    stats = {}
    suppressed = _heterogeneous_conv_features(
        tokens,
        spatial,
        temporal,
        frame_sizes,
        offsets,
        l2norm=False,
        suppress_cross_grid_temporal_taps=True,
        diagnostic_stats=stats,
    )

    assert normal.shape == suppressed.shape == tokens.shape
    assert stats["suppressed_taps"] > 0
    assert stats["suppressed_rows"] > 0

    # Same-domain temporal neighborhoods stay byte-identical.
    first = slice(*offsets[0])
    last = slice(*offsets[-1])
    assert torch.equal(normal[first], suppressed[first])
    assert torch.equal(normal[last], suppressed[last])

    # Boundary-adjacent frames change because only cross-grid contributions are
    # removed; the spatial conv, same-grid temporal taps and the rest of the
    # learned-linear branch remain active.
    before = slice(*offsets[2])
    after = slice(*offsets[3])
    assert not torch.equal(normal[before], suppressed[before])
    assert not torch.equal(normal[after], suppressed[after])


def test_cross_grid_temporal_diagnostic_is_noop_on_uniform_grid():
    frame_sizes = ((2, 2),) * 4
    offsets = ((0, 4), (4, 8), (8, 12), (12, 16))
    tokens = torch.randn(16, 1, 1, generator=torch.Generator().manual_seed(5))
    spatial = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
    spatial[0, 0, 2, 2] = 1.0
    temporal = torch.randn((1, 1, 5), generator=torch.Generator().manual_seed(6))
    stats = {}

    normal = _heterogeneous_conv_features(
        tokens,
        spatial,
        temporal,
        frame_sizes,
        offsets,
        l2norm=False,
    )
    suppressed = _heterogeneous_conv_features(
        tokens,
        spatial,
        temporal,
        frame_sizes,
        offsets,
        l2norm=False,
        suppress_cross_grid_temporal_taps=True,
        diagnostic_stats=stats,
    )

    assert torch.equal(normal, suppressed)
    assert stats == {}
