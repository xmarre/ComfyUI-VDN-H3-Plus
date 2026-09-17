from vdn_h3.partitioned_grouped import build_partitioned_grouped_plan
from vdn_h3.partitioned_sequence import PartitionedSequence
from vdn_h3.query_positions import WIRE_SCHEMA, WIRE_TAG
from vdn_h3.window import window_bounds


def test_partitioned_grouped_geometry_keeps_target_prefix_and_source_suffix_rows():
    flow = PartitionedSequence(
        video_start=7,
        temporal=5,
        prefix_t=2,
        source_grid_h=2,
        source_grid_w=3,
        target_grid_h=3,
        target_grid_w=4,
    )
    bounds = ((0, 1), (0, 2), (1, 3), (2, 4), (3, 4))
    grouped = build_partitioned_grouped_plan(
        flow,
        bounds=bounds,
        anchor_frames="none",
        semantic_digest="a" * 64,
    )

    assert grouped.frame_ranges == (
        (7, 19),
        (19, 31),
        (31, 37),
        (37, 43),
        (43, 49),
    )
    assert grouped.sequence_rows == 49
    assert grouped.full_prefix_k_range == (7, 31)
    assert grouped.source_rows == 6
    assert grouped.target_rows == 12

    # Prefix and suffix query frames are never collapsed into one query group,
    # even when their temporal bounds happen to match.
    assert all(
        all((frame < flow.prefix_t) == group.query_prefix_domain for frame in group.query_frames)
        for group in grouped.groups
    )


def test_partitioned_grouped_geometry_clamps_released_window_bounds_identically():
    flow = PartitionedSequence(
        video_start=7,
        temporal=5,
        prefix_t=2,
        source_grid_h=2,
        source_grid_w=3,
        target_grid_h=3,
        target_grid_w=4,
    )
    raw = window_bounds(flow.temporal, radius=1, chunk=1)
    assert raw[0][0] < 0
    assert raw[-1][1] >= flow.temporal

    grouped = build_partitioned_grouped_plan(
        flow,
        bounds=raw,
        anchor_frames="none",
        semantic_digest="d" * 64,
    )
    assert grouped.bounds == ((0, 1), (0, 2), (1, 3), (2, 4), (3, 4))


def test_partitioned_group_wire_maps_requested_rows_into_gathered_domain():
    flow = PartitionedSequence(
        video_start=5,
        temporal=4,
        prefix_t=1,
        source_grid_h=2,
        source_grid_w=2,
        target_grid_h=4,
        target_grid_w=2,
    )
    grouped = build_partitioned_grouped_plan(
        flow,
        bounds=((0, 1), (0, 2), (1, 3), (2, 3)),
        anchor_frames="columns",
        semantic_digest="b" * 64,
    )
    owner = "vdn-test-owner"

    for group, wire in zip(grouped.groups, grouped.wires(owner), strict=True):
        assert wire[0] == WIRE_TAG
        assert wire[1] == WIRE_SCHEMA
        assert wire[2] == owner
        assert wire[3] == grouped.plan_digest
        assert wire[4] == group.group_index
        assert wire[5] == group.q_rows
        assert wire[6] == group.kv_rows
        assert wire[7] == group.sink_rows
        assert wire[8] == group.query_position_runs
        assert wire[8][0][0] == 0
        assert wire[8][-1][1] == group.q_rows
        assert all(k_begin >= group.sink_rows for _, _, k_begin in wire[8])


def test_partitioned_grouped_plan_keeps_prefix_measure_range_contiguous_after_global_sink():
    flow = PartitionedSequence(
        video_start=11,
        temporal=6,
        prefix_t=2,
        source_grid_h=2,
        source_grid_w=2,
        target_grid_h=4,
        target_grid_w=4,
    )
    grouped = build_partitioned_grouped_plan(
        flow,
        bounds=((0, 2), (0, 2), (0, 3), (1, 4), (2, 5), (3, 5)),
        anchor_frames="none",
        semantic_digest="c" * 64,
    )

    for group in grouped.groups:
        prefix_frames = [frame for frame in group.key_frames if frame < flow.prefix_t]
        if not prefix_frames:
            assert group.prefix_k_range is None
            continue
        start, end = group.prefix_k_range
        assert start == group.sink_rows
        assert end - start == len(prefix_frames) * flow.target_rows
