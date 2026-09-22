import math

import pytest

from vdn_h3.partitioned_sequence import (
    PartitionedSequence,
    make_vdn_partitioned_external_contract,
    validate_flow_partition_contract,
)


def test_flow_partition_contract_roundtrips_and_binds_vdn_external_mode():
    plan = PartitionedSequence(
        video_start=7,
        temporal=5,
        prefix_t=2,
        source_grid_h=3,
        source_grid_w=4,
        target_grid_h=5,
        target_grid_w=6,
    )
    contract = plan.canonical_contract()
    parsed = validate_flow_partition_contract(contract, sequence_rows=103)
    assert parsed == plan
    assert parsed.prefix_log_key_measure == pytest.approx(math.log(12 / 30))

    external = make_vdn_partitioned_external_contract(parsed)
    assert external == {
        "api": 4,
        "mode": "partitioned_attention_variable_grid_linear",
        "topology": "target_prefix_source_suffix",
        "sequence_rows": 103,
        "video_start": 7,
        "temporal": 5,
        "prefix_t": 2,
        "source_rows_per_frame": 12,
        "target_rows_per_frame": 30,
        "flow_semantic_digest": contract["semantic_digest"],
    }


def test_flow_partition_contract_rejects_tampering_and_stale_rows():
    plan = PartitionedSequence(
        video_start=5,
        temporal=4,
        prefix_t=1,
        source_grid_h=2,
        source_grid_w=3,
        target_grid_h=4,
        target_grid_w=4,
    )
    contract = plan.canonical_contract()
    contract["prefix_range"] = [5, 999]
    with pytest.raises(ValueError, match="prefix_range"):
        validate_flow_partition_contract(contract, sequence_rows=69)

    contract = plan.canonical_contract()
    with pytest.raises(ValueError, match="current hidden sequence"):
        validate_flow_partition_contract(contract, sequence_rows=70)


def test_partitioned_sequence_requires_smaller_source_and_generated_suffix():
    with pytest.raises(ValueError, match="generated suffix"):
        PartitionedSequence(
            video_start=5,
            temporal=2,
            prefix_t=2,
            source_grid_h=2,
            source_grid_w=2,
            target_grid_h=4,
            target_grid_w=4,
        )
    with pytest.raises(ValueError, match="strictly smaller"):
        PartitionedSequence(
            video_start=5,
            temporal=3,
            prefix_t=1,
            source_grid_h=4,
            source_grid_w=4,
            target_grid_h=4,
            target_grid_w=4,
        )
