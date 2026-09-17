import torch

from vdn_h3.runtime import RuntimeBufferOwner


def test_retained_partition_index_cache_reuses_identical_execution_geometry():
    owner = RuntimeBufferOwner(True)
    identity = ("partitioned_exact_prefix_v1", "a" * 64, "q", 0)
    ranges = ((0, 3), (5, 9))

    with owner.execution() as resources:
        first = resources.partition_indices(identity, ranges, "cpu")
        second = resources.partition_indices(identity, ranges, "cpu")
        assert torch.equal(first, torch.tensor([0, 1, 2, 5, 6, 7, 8]))
        assert first.data_ptr() == second.data_ptr()
        counts = resources.retained_counts()
        assert counts["partition_indices"] == 1
        assert counts["partition_index_bytes"] == first.numel() * first.element_size()

    with owner.execution() as resources:
        third = resources.partition_indices(identity, ranges, "cpu")
        assert third.data_ptr() == first.data_ptr()
        assert resources.retained_counts()["partition_indices"] == 1


def test_transient_partition_index_cache_is_scoped_to_one_execution():
    owner = RuntimeBufferOwner(False)
    identity = ("partitioned_exact_prefix_v1", "b" * 64, "k", 1)
    ranges = ((1, 4), (7, 10))

    with owner.execution() as first_resources:
        first = first_resources.partition_indices(identity, ranges, "cpu")
        second = first_resources.partition_indices(identity, ranges, "cpu")
        assert first.data_ptr() == second.data_ptr()
        assert first_resources.retained_counts()["partition_indices"] == 1

    with owner.execution() as second_resources:
        assert second_resources is not first_resources
        assert second_resources.retained_counts()["partition_indices"] == 0
        again = second_resources.partition_indices(identity, ranges, "cpu")
        assert torch.equal(again, first)


def test_partition_index_cache_is_bounded_and_rejects_invalid_ranges():
    owner = RuntimeBufferOwner(True)
    with owner.execution() as resources:
        for index in range(80):
            resources.partition_indices(("plan", index), ((index, index + 1),), "cpu")
        counts = resources.retained_counts()
        assert counts["partition_indices"] <= 64
        assert counts["partition_index_bytes"] <= 4 * 1024 * 1024

        for ranges in ((), ((2, 2),), ((3, 4), (2, 3)), ((-1, 1),)):
            try:
                resources.partition_indices(("bad", ranges), ranges, "cpu")
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"invalid partition ranges were accepted: {ranges!r}")
