from contextlib import contextmanager
from types import SimpleNamespace
import weakref

import pytest
import torch

from vdn_h3.branch import LinearBranch
from vdn_h3 import branch as B
from vdn_h3.retained import RuntimeLinearBranch
from vdn_h3.runtime import RuntimeBufferOwner, current_runtime_buffers
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
from vdn_h3.partitioned_sequence import (
    PartitionedSequence,
    VDN_TEMPORAL_CARRIER_DESTINATION,
    VDN_TEMPORAL_CARRIER_NATIVE,
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


def test_destination_grid_stencil_matches_independent_scalar_reference():
    frame_sizes = ((4, 5),) * 4 + ((3, 4),) * 6
    offsets = []
    cursor = 0
    for grid_h, grid_w in frame_sizes:
        next_cursor = cursor + grid_h * grid_w
        offsets.append((cursor, next_cursor))
        cursor = next_cursor
    offsets = tuple(offsets)

    generator = torch.Generator(device="cpu").manual_seed(709)
    tokens = torch.randn((cursor, 2, 3), generator=generator, dtype=torch.float32)
    channels = 6
    spatial = torch.randn((channels, 1, 5, 5), generator=generator, dtype=torch.float32) * 0.05
    temporal = torch.randn((channels, 1, 5), generator=generator, dtype=torch.float32) * 0.05
    stats = {}

    reference = _heterogeneous_conv_features_reference(
        tokens,
        spatial,
        temporal,
        frame_sizes,
        offsets,
        l2norm=True,
        temporal_carrier_policy=VDN_TEMPORAL_CARRIER_DESTINATION,
    )
    candidate = _heterogeneous_conv_features(
        tokens,
        spatial,
        temporal,
        frame_sizes,
        offsets,
        l2norm=True,
        temporal_carrier_policy=VDN_TEMPORAL_CARRIER_DESTINATION,
        carrier_stats=stats,
    )

    assert torch.allclose(candidate, reference, rtol=2e-5, atol=2e-5)
    assert stats == {
        "cross_grid_taps": 6,
        "mapped_carriers": 4,
        "mapped_carrier_rows": 64,
    }


def test_destination_grid_stencil_is_exact_noop_without_cross_grid_taps():
    frame_sizes = ((3, 4),) * 5
    offsets = tuple((index * 12, (index + 1) * 12) for index in range(5))
    generator = torch.Generator(device="cpu").manual_seed(710)
    tokens = torch.randn((60, 1, 2), generator=generator, dtype=torch.float32)
    spatial = torch.randn((2, 1, 5, 5), generator=generator, dtype=torch.float32) * 0.05
    temporal = torch.randn((2, 1, 5), generator=generator, dtype=torch.float32) * 0.05
    stats = {}

    native = _heterogeneous_conv_features(
        tokens,
        spatial,
        temporal,
        frame_sizes,
        offsets,
        l2norm=False,
        temporal_carrier_policy=VDN_TEMPORAL_CARRIER_NATIVE,
    )
    candidate = _heterogeneous_conv_features(
        tokens,
        spatial,
        temporal,
        frame_sizes,
        offsets,
        l2norm=False,
        temporal_carrier_policy=VDN_TEMPORAL_CARRIER_DESTINATION,
        carrier_stats=stats,
    )

    assert torch.equal(candidate, native)
    assert stats == {}


def test_destination_grid_stencil_changes_only_noncommuting_cross_grid_order():
    frame_sizes = ((5, 7),) * 3 + ((3, 4),) * 4
    offsets = []
    cursor = 0
    for grid_h, grid_w in frame_sizes:
        next_cursor = cursor + grid_h * grid_w
        offsets.append((cursor, next_cursor))
        cursor = next_cursor
    offsets = tuple(offsets)
    generator = torch.Generator(device="cpu").manual_seed(711)
    tokens = torch.randn((cursor, 1, 3), generator=generator, dtype=torch.float32)
    spatial = torch.randn((3, 1, 5, 5), generator=generator, dtype=torch.float32) * 0.1
    temporal = torch.randn((3, 1, 5), generator=generator, dtype=torch.float32) * 0.1

    native = _heterogeneous_conv_features(
        tokens,
        spatial,
        temporal,
        frame_sizes,
        offsets,
        l2norm=False,
        temporal_carrier_policy=VDN_TEMPORAL_CARRIER_NATIVE,
    )
    candidate = _heterogeneous_conv_features(
        tokens,
        spatial,
        temporal,
        frame_sizes,
        offsets,
        l2norm=False,
        temporal_carrier_policy=VDN_TEMPORAL_CARRIER_DESTINATION,
    )

    # Frames beyond the five-tap radius never receive another grid and therefore
    # remain bitwise on the audited native path; boundary-adjacent rows differ.
    assert torch.equal(native[slice(*offsets[0])], candidate[slice(*offsets[0])])
    assert torch.equal(native[slice(*offsets[-1])], candidate[slice(*offsets[-1])])
    boundary_rows = slice(offsets[1][0], offsets[4][1])
    assert not torch.equal(native[boundary_rows], candidate[boundary_rows])
    assert torch.isfinite(candidate).all()


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


@pytest.mark.parametrize("prefix_t", (1, 2, 3))
@pytest.mark.parametrize("grid", ((2, 2), (2, 3)))
@pytest.mark.parametrize("skip_ends", (False, True))
def test_variable_grid_linear_reduces_to_released_readout_on_uniform_grid(prefix_t, grid, skip_ends):
    weights = _weights()
    branch = _branch(weights)
    frames = 4
    plan = PartitionedSequence(
        video_start=1,
        temporal=frames,
        prefix_t=prefix_t,
        source_grid_h=grid[0],
        source_grid_w=grid[1],
        target_grid_h=grid[0],
        target_grid_w=grid[1],
    )
    frame_sizes, measures = partitioned_frame_contract(plan)
    assert frame_sizes == (grid,) * frames
    assert measures == (1.0,) * frames
    rows_per_frame = grid[0] * grid[1]
    rows = frames * rows_per_frame
    bounds = ((0, 1), (0, 2), (1, 3), (2, 3))
    x, q, k, v = _inputs(rows)

    released = branch.readout(
        weights,
        x,
        q,
        k,
        v,
        frames,
        rows_per_frame,
        bounds,
        grid,
        None,
        None,
        None,
        skip_ends=skip_ends,
    )
    partitioned = partitioned_linear_readout(
        branch,
        weights,
        x,
        q,
        k,
        v,
        frame_sizes=frame_sizes,
        bounds=bounds,
        measure_scales=measures,
        skip_ends=skip_ends,
    )
    # Force the original general path without changing uniform-grid arithmetic,
    # so dispatch to released readout cannot make this oracle tautological.
    general = partitioned_linear_readout(
        branch, weights, x, q, k, v,
        frame_sizes=frame_sizes, bounds=bounds, measure_scales=measures,
        skip_ends=skip_ends, diagnostic_stats={},
    )

    assert partitioned.shape == released.shape
    assert torch.allclose(partitioned, released, rtol=2e-5, atol=2e-5)
    assert torch.allclose(partitioned, general, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("fast_kernels", (False, True))
@pytest.mark.parametrize("skip_ends", (False, True))
@pytest.mark.parametrize("strided", [False, True])
def test_uniform_dispatch_preserves_text_anchors_and_execution_owned_scans(monkeypatch, fast_kernels, skip_ends, strided):
    weights = _weights(seed=311)
    branch = RuntimeLinearBranch(weights, 2, 3, enable_text_state=True)
    branch.fuse_epilogue = fast_kernels
    shared_backend = branch._delta_backend(97)
    shared_key = branch._backend_key
    inputs = _inputs(24)
    if strided:
        # The actual QKV projection's split views share a 3*H*d row stride.
        x, q, k, v = inputs
        packed = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)
        inputs = (x, *(part.view_as(q) for part in packed.split(q.shape[1] * q.shape[2], dim=-1)))
    text_x, _text_q, text_k, text_v = _inputs(3, seed=313)
    originals = tuple(t.clone() for t in (*inputs, text_x, text_k, text_v))
    kwargs = dict(
        frame_sizes=((2, 3),) * 4, measure_scales=(1.0,) * 4,
        bounds=((-1, 1), (0, 2), (1, 3), (2, 4)), skip_ends=skip_ends,
        text_x=text_x, text_k_raw=text_k, text_v_raw=text_v,
    )
    reference = partitioned_linear_readout(branch, weights, *inputs, diagnostic_stats={}, **kwargs)
    compiled_keys = []
    epilogue_flags = []
    epilogue = B.linear_epilogue

    def compiled_body(key, body, *args, **kw):
        compiled_keys.append(key)
        return body(*args, **kw)

    def checked_epilogue(*args, fuse=False):
        epilogue_flags.append(fuse)
        return epilogue(*args, fuse=False)

    # Test flag routing and math without compiling a CPU kernel that would not
    # qualify the CUDA implementation.
    monkeypatch.setattr(B, "_run_compiled", compiled_body)
    monkeypatch.setattr(B, "linear_epilogue", checked_epilogue)
    recorded = {}
    spans = []

    @contextmanager
    def cuda_span(name):
        spans.append(name)
        yield

    owner = RuntimeBufferOwner(True)
    with owner.execution() as outer:
        stats = {}
        got = partitioned_linear_readout(
            branch, weights, *inputs, **kwargs, execution_stats=stats,
            record_component=lambda name, elapsed: recorded.__setitem__(name, elapsed),
            cuda_span=cuda_span,
        )
        assert outer.retained_counts()["scan"] == 1
        outer_banks = next(iter(outer._scan.values()))
        outer_snapshot = tuple(bank.clone() for bank in outer_banks)
        with owner.execution() as inner:
            nested = partitioned_linear_readout(branch, weights, *inputs, **kwargs)
            assert current_runtime_buffers() is inner and inner is not outer
        assert current_runtime_buffers() is outer
        for bank, before in zip(outer_banks, outer_snapshot, strict=True):
            assert torch.equal(bank, before)
    assert current_runtime_buffers() is None
    assert stats == {
        "native_uniform_calls": 1,
        **({"native_uniform_fast_requested_calls": 1} if fast_kernels else {}),
    }
    assert epilogue_flags == [fast_kernels, fast_kernels]
    assert bool(compiled_keys) is fast_kernels
    if fast_kernels:
        assert any(key[0] == "act_fhsd" for key in compiled_keys)
        assert any(key[0] == "gather" for key in compiled_keys)
    assert spans == ["vdn_linear_native_uniform"]
    assert set(recorded) == {"vdn_linear_native_uniform_host_wall_s", "vdn_linear_api_host_wall_s"}
    assert all(elapsed >= 0 for elapsed in recorded.values())
    assert torch.allclose(got, reference, rtol=2e-5, atol=2e-5)
    assert torch.equal(got, nested)
    if skip_ends:
        assert torch.count_nonzero(got[:6]) == torch.count_nonzero(got[-6:]) == 0
    assert branch._backend is shared_backend and branch._backend_key == shared_key
    for before, after in zip(originals, (*inputs, text_x, text_k, text_v), strict=True):
        assert torch.equal(before, after)


@pytest.mark.parametrize("fallback", (
    "same_area_different_axes", "nonunit_measure", "near_unit_measure", "noncontiguous",
    "q_convolution", "cross_grid_diagnostic", "diagnostic_stats", "destination_carrier",
    "carrier_stats", "observation",
))
def test_uniform_dispatch_keeps_general_path_for_distinct_contracts(monkeypatch, fallback):
    weights = _weights()
    branch = _branch(weights)
    inputs = _inputs(24)
    kwargs = dict(frame_sizes=((2, 3),) * 4, bounds=((0, 1), (0, 2), (1, 3), (2, 3)), measure_scales=(1.0,) * 4)
    if fallback == "same_area_different_axes":
        kwargs["frame_sizes"] = ((2, 3), (2, 3), (3, 2), (3, 2))
    elif fallback in {"nonunit_measure", "near_unit_measure"}:
        kwargs["measure_scales"] = (0.5 if fallback == "nonunit_measure" else 1.0 - 1e-12,) + (1.0,) * 3
    elif fallback == "noncontiguous":
        inputs = tuple(torch.stack((t, t), dim=-1)[..., 0] for t in inputs)
    elif fallback == "q_convolution":
        branch.short_conv = ("q", "k", "v")
    elif fallback == "cross_grid_diagnostic":
        kwargs["suppress_cross_grid_temporal_taps"] = True
    elif fallback == "diagnostic_stats":
        kwargs["diagnostic_stats"] = {}
    elif fallback == "destination_carrier":
        kwargs["temporal_carrier_policy"] = VDN_TEMPORAL_CARRIER_DESTINATION
    elif fallback == "carrier_stats":
        kwargs["carrier_stats"] = {}
    else:
        kwargs["observation"] = object()
    sentinel = torch.empty((24, 6))
    from vdn_h3 import partitioned_linear

    def general(*args, **kw):
        assert kw.get("observation") is kwargs.get("observation")
        return sentinel

    def unexpected_native(*args, **kw):
        pytest.fail("a distinct partitioned contract reached the native uniform readout")

    monkeypatch.setattr(partitioned_linear, "_core_readout", general)
    monkeypatch.setattr(LinearBranch, "readout", unexpected_native)
    stats = {}
    assert partitioned_linear_readout(branch, weights, *inputs, **kwargs, execution_stats=stats) is sentinel
    assert stats == {}


@pytest.mark.parametrize("bad_input, message", (
    ("measure", "physical measure"), ("bounds", "bound"), ("rows", "hidden rows"),
    ("anchor_bounds", "bound"),
    ("even_temporal_kernel", "odd depthwise kernel"),
))
def test_uniform_dispatch_preserves_validation(monkeypatch, bad_input, message):
    weights = _weights()
    branch = _branch(weights)
    inputs = _inputs(24)
    kwargs = dict(frame_sizes=((2, 3),) * 4, bounds=((0, 1), (0, 2), (1, 3), (2, 3)), measure_scales=(1.0,) * 4)
    if bad_input == "measure":
        kwargs["measure_scales"] = (1.0, float("nan"), 1.0, 1.0)
    elif bad_input == "bounds":
        kwargs["bounds"] = ((2, 1), (0, 2), (1, 3), (2, 3))
    elif bad_input == "rows":
        inputs = (inputs[0][:-1], *inputs[1:])
    elif bad_input == "anchor_bounds":
        kwargs["skip_ends"] = True
        kwargs["bounds"] = ((0, 1), (0, 0), (1, 3), (2, 3))
    else:
        weights["short_conv.k_tm.weight"] = weights["short_conv.k_tm.weight"][..., :4]

    def unexpected_native(*args, **kw):
        pytest.fail("malformed input reached the native uniform readout")

    monkeypatch.setattr(LinearBranch, "readout", unexpected_native)
    with pytest.raises(RuntimeError, match=message):
        partitioned_linear_readout(branch, weights, *inputs, **kwargs)


@pytest.mark.parametrize("general", (False, True))
@pytest.mark.parametrize("retain", (False, True))
def test_linear_features_and_statistics_do_not_overlap_later_workspaces(monkeypatch, general, retain):
    from vdn_h3 import partitioned_linear, retained

    weights = _weights(seed=317)
    branch = RuntimeLinearBranch(weights, 2, 3, enable_text_state=False)
    inputs = _inputs(24)
    bounds = ((0, 1), (0, 2), (1, 3), (2, 3))
    with torch.no_grad():
        reference = _branch(weights).readout(weights, *inputs, 4, 6, bounds, frame_size=(2, 3))
    refs = {}
    native_features = RuntimeLinearBranch._features
    variable_features = partitioned_linear._variable_features
    statistics = B.frame_statistics
    alpha_gate = B.alpha_gate
    scans = retained.run_scans_runtime
    gather = B.gather_linear_state
    epilogue = B.linear_epilogue

    def track_features(values):
        refs.update({name: weakref.ref(t) for name, t in zip(("query", "key", "value"), values, strict=True)})
        return values

    def checked_statistics(*args, **kwargs):
        values = statistics(*args, **kwargs)
        refs.update({name: weakref.ref(t) for name, t in zip(("a", "b"), values, strict=True)})
        return values

    def checked_alpha(*args, **kwargs):
        assert refs["key"]() is None and refs["value"]() is None
        return alpha_gate(*args, **kwargs)

    def checked_scans(*args, **kwargs):
        values = scans(*args, **kwargs)
        refs.update({name: weakref.ref(t) for name, t in zip(("prefix", "suffix"), values, strict=True)})
        return values

    def checked_gather(*args, **kwargs):
        assert refs["a"]() is None and refs["b"]() is None
        value = gather(*args, **kwargs)
        refs["state"] = weakref.ref(value)
        return value

    def checked_epilogue(*args, **kwargs):
        assert (refs["prefix"]() is not None) is retain
        assert (refs["suffix"]() is not None) is retain
        if not general:
            assert refs["query"]() is None and refs["state"]() is None
        return epilogue(*args, **kwargs)

    monkeypatch.setattr(RuntimeLinearBranch, "_features", lambda *a, **kw: track_features(native_features(*a, **kw)))
    monkeypatch.setattr(partitioned_linear, "_variable_features", lambda *a, **kw: track_features(variable_features(*a, **kw)))
    monkeypatch.setattr(B, "frame_statistics", checked_statistics)
    monkeypatch.setattr(B, "alpha_gate", checked_alpha)
    monkeypatch.setattr(retained, "run_scans_runtime", checked_scans)
    monkeypatch.setattr(partitioned_linear, "run_scans_runtime", checked_scans)
    monkeypatch.setattr(B, "gather_linear_state", checked_gather)
    monkeypatch.setattr(B, "linear_epilogue", checked_epilogue)
    with RuntimeBufferOwner(retain).execution(), torch.no_grad():
        result = partitioned_linear_readout(
            branch, weights, *inputs, frame_sizes=((2, 3),) * 4,
            bounds=bounds, measure_scales=(1.0,) * 4,
            diagnostic_stats={} if general else None,
        )
    assert torch.allclose(result, reference, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("prefix_t, source_rows", ((0, 4), (4, 4), (2, 0), (2, 5)))
def test_partitioned_frame_contract_rejects_invalid_geometry(prefix_t, source_rows):
    plan = SimpleNamespace(
        target_grid_h=2,
        target_grid_w=2,
        source_grid_h=2,
        source_grid_w=2,
        prefix_t=prefix_t,
        temporal=4,
        target_rows=4,
        source_rows=source_rows,
    )
    with pytest.raises(RuntimeError, match="invalid Flow geometry"):
        partitioned_frame_contract(plan)


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


def test_variable_grid_readout_honors_selected_fused_gather_and_epilogue(monkeypatch):
    weights = _weights()
    branch = _branch(weights)
    grids = ((4, 5),) * 3 + ((3, 4),) * 4
    rows = sum(h * w for h, w in grids)
    inputs = _inputs(rows)
    kwargs = dict(frame_sizes=grids, bounds=tuple((0, 6) for _ in grids),
                  measure_scales=(0.6,) * 3 + (1.0,) * 4)
    reference = partitioned_linear_readout(branch, weights, *inputs, **kwargs)
    gathers, epilogues = [], []
    gather, epilogue = B.gather_linear_state, B.linear_epilogue

    def observed_gather(*args, fuse=False, **kwargs):
        gathers.append(fuse)
        return gather(*args, fuse=False, **kwargs)

    def observed_epilogue(*args, fuse=False):
        epilogues.append(fuse)
        return epilogue(*args, fuse=False)

    monkeypatch.setattr(B, "gather_linear_state", observed_gather)
    monkeypatch.setattr(B, "linear_epilogue", observed_epilogue)
    branch.fuse_epilogue = True
    result = partitioned_linear_readout(branch, weights, *inputs, **kwargs)
    assert gathers == [True]
    assert epilogues == [True, True]
    torch.testing.assert_close(result, reference)


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
