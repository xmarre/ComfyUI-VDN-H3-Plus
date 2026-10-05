"""Target native carrier for Flow's progressive target-band continuation."""

from types import SimpleNamespace

import pytest
import torch

from vdn_h3.partitioned_runtime import (
    VDN_EXTERNAL_SEQUENCE_KEY,
    _wrap_vdn_forward,
    validate_partitioned_external_execution,
)
from vdn_h3.partitioned_sequence import (
    PARTITIONED_NATIVE_CARRIER_GRIDS,
    PARTITIONED_NATIVE_CARRIER_SOURCE,
    PARTITIONED_NATIVE_CARRIER_TARGET,
    PARTITIONED_PREFIX_KEY,
    PartitionedSequence,
    make_vdn_partitioned_external_contract,
    validate_flow_partition_contract,
)

_GEOMETRY = dict(
    video_start=7,
    temporal=5,
    prefix_t=3,
    source_grid_h=2,
    source_grid_w=3,
    target_grid_h=3,
    target_grid_w=4,
)


def test_source_carrier_contract_and_digest_are_unchanged_by_the_optional_field():
    legacy = PartitionedSequence(**_GEOMETRY)
    explicit = PartitionedSequence(**_GEOMETRY, native_carrier_grid=PARTITIONED_NATIVE_CARRIER_SOURCE)
    assert legacy == explicit
    contract = legacy.canonical_contract()
    assert "native_carrier_grid" not in contract
    assert "native_carrier_rows_per_frame" not in contract
    assert "native_carrier_rows_per_frame" not in make_vdn_partitioned_external_contract(legacy)
    assert legacy.native_rows_per_frame == legacy.source_rows


def test_target_carrier_contract_roundtrips_with_a_distinct_digest():
    legacy = PartitionedSequence(**_GEOMETRY)
    band = PartitionedSequence(**_GEOMETRY, native_carrier_grid=PARTITIONED_NATIVE_CARRIER_TARGET)
    contract = band.canonical_contract()
    assert contract["native_carrier_grid"] == "target"
    assert contract["native_carrier_rows_per_frame"] == band.target_rows == 12
    assert contract["semantic_digest"] != legacy.canonical_contract()["semantic_digest"]
    # The partitioned sequence itself is unchanged: [target head | source tail].
    assert contract["sequence_rows"] == legacy.canonical_contract()["sequence_rows"]
    assert validate_flow_partition_contract(contract, sequence_rows=band.sequence_rows) == band
    external = make_vdn_partitioned_external_contract(band)
    assert external["native_carrier_rows_per_frame"] == 12
    assert PARTITIONED_NATIVE_CARRIER_GRIDS == ("source", "target")


def test_target_carrier_contract_fails_closed_on_tampering():
    band = PartitionedSequence(**_GEOMETRY, native_carrier_grid=PARTITIONED_NATIVE_CARRIER_TARGET)
    contract = band.canonical_contract()

    explicit_source = dict(contract, native_carrier_grid="source")
    with pytest.raises(ValueError, match="only for the target carrier"):
        validate_flow_partition_contract(explicit_source, sequence_rows=band.sequence_rows)

    wrong_rows = dict(contract, native_carrier_rows_per_frame=6)
    with pytest.raises(ValueError, match="native_carrier_rows_per_frame"):
        validate_flow_partition_contract(wrong_rows, sequence_rows=band.sequence_rows)

    stripped = {key: value for key, value in contract.items() if not key.startswith("native_carrier")}
    with pytest.raises(ValueError, match="semantic_digest"):
        validate_flow_partition_contract(stripped, sequence_rows=band.sequence_rows)

    with pytest.raises(ValueError, match="heterogeneous"):
        PartitionedSequence(
            **dict(_GEOMETRY, source_grid_h=3, source_grid_w=4),
            native_carrier_grid=PARTITIONED_NATIVE_CARRIER_TARGET,
        )
    with pytest.raises(ValueError, match="unsupported partitioned native carrier"):
        PartitionedSequence(**_GEOMETRY, native_carrier_grid="mixed")


def _native_layout(plan, rows_per_frame):
    native_rows = plan.video_start + plan.temporal * rows_per_frame
    return SimpleNamespace(
        seq_len=native_rows,
        video_start=plan.video_start,
        video_end=native_rows,
        num_frames=plan.temporal,
        tokens_per_frame=rows_per_frame,
    )


@pytest.mark.parametrize("carrier", PARTITIONED_NATIVE_CARRIER_GRIDS)
def test_external_execution_binds_the_declared_native_carrier(carrier):
    plan = PartitionedSequence(**_GEOMETRY, native_carrier_grid=carrier)
    options = {
        PARTITIONED_PREFIX_KEY: plan.canonical_contract(),
        VDN_EXTERNAL_SEQUENCE_KEY: make_vdn_partitioned_external_contract(plan),
    }
    rope = torch.zeros(1, plan.sequence_rows, 2)
    declared = _native_layout(plan, plan.native_rows_per_frame)
    assert validate_partitioned_external_execution(options, declared, plan.sequence_rows, rope) == plan

    other = plan.source_rows if carrier == PARTITIONED_NATIVE_CARRIER_TARGET else plan.target_rows
    with pytest.raises(RuntimeError, match="native carrier layout"):
        validate_partitioned_external_execution(options, _native_layout(plan, other), plan.sequence_rows, rope)


def test_bridge_advertises_native_carrier_capability():
    base_branch = SimpleNamespace(short_conv=(), delta_rule="vdn_solve", num_heads=1, head_dim=8, a_fp32=True)
    block_index = 0
    cfg = {}
    head_dim = 8
    heads = 1
    k_norm = q_norm = out_proj = qkv_proj = state = SimpleNamespace()

    def current(x, rope_freqs=None, transformer_options=None):
        _ = (base_branch, block_index, cfg, head_dim, heads, k_norm, out_proj, q_norm, qkv_proj, state)
        return x

    current._vdn_forward = True
    wrapped = _wrap_vdn_forward(current)
    assert tuple(wrapped._vdn_partitioned_native_carrier_grids) == PARTITIONED_NATIVE_CARRIER_GRIDS
