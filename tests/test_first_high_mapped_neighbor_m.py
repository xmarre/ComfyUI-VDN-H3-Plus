from __future__ import annotations

import pytest
import torch

from vdn_h3 import first_high_mapped_neighbor_diagnostic as m


def test_group10_saved_position_geometry_maps_to_physical_k_neighbours():
    positions = torch.arange(9245, 10269, dtype=torch.long)
    domain = torch.arange(11293, dtype=torch.long)
    q_idx = domain.index_select(0, positions)

    intervals = m._mapped_intervals(q_idx, domain, positions)

    assert len(intervals) == 16
    assert intervals[0] == (143, 147)
    assert intervals[-1] == (158, 162)
    assert all(1 <= end - start <= 4 for start, end in intervals)


def test_mapping_is_derived_from_domain_identity_not_local_q_ordinal():
    domain = torch.arange(5000, 5200, dtype=torch.long)
    positions = torch.arange(65, 129, dtype=torch.long)
    q_idx = domain.index_select(0, positions)

    intervals = m._mapped_intervals(q_idx, domain, positions)

    # The requested Q tile is local-Q ordinal 0 but physically occupies K
    # blocks 1/2 in the gathered domain, so M must protect blocks 0..3.
    assert intervals == ((0, 4),)


def test_mapping_fails_closed_on_noncontiguous_or_inconsistent_plan():
    domain = torch.arange(256, dtype=torch.long)
    positions = torch.arange(64, 128, dtype=torch.long)
    q_idx = domain.index_select(0, positions)

    broken = positions.clone()
    broken[32:] += 1
    with pytest.raises(RuntimeError, match="contiguous"):
        m._mapped_intervals(q_idx, domain, broken)

    wrong_q = q_idx.clone()
    wrong_q[0] += 1
    with pytest.raises(RuntimeError, match="map back"):
        m._mapped_intervals(wrong_q, domain, positions)
