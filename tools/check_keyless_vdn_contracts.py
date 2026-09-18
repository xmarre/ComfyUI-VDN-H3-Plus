#!/usr/bin/env python3
"""Execute the pinned MiniMax-H3-Keyless row-domain contract consumed by VDN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyless", type=Path, required=True)
    args = parser.parse_args()

    root = args.keyless.resolve()
    sys.path.insert(0, str(root))
    try:
        from minimax_h3_keyless.contracts import (
            ARCHITECTURE,
            CONTRACT_KEY,
            PROVIDER_KEY,
            QV_ORDER,
            RowDomain,
            RoutingSpecV1,
        )
    finally:
        sys.path.pop(0)

    assert ARCHITECTURE == "h3_keyless_core50_v1"
    assert CONTRACT_KEY == "minimax_h3_keyless_contract_v1"
    assert PROVIDER_KEY == "minimax_h3_keyless_provider_v1"
    assert QV_ORDER == "q_effective;v"

    rows = 4
    value = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 1, 2)
    rope = torch.arange(rows * 2, dtype=torch.float32).reshape(1, rows, 1, 1, 2)
    measure = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    spec = RoutingSpecV1(
        api=1,
        block_index=7,
        norm_weight=torch.ones(2),
        norm_epsilon=1e-5,
        rope_freqs=rope,
        value_domain=RowDomain(start=100, stop=104, identity="value-parent"),
        routing_position_domain=RowDomain(start=200, stop=204, identity="route-parent"),
    )
    selector = torch.tensor([3, 1], dtype=torch.long)
    selected, child, selected_measure = spec.select_value_rows(
        value,
        selector,
        log_measure=measure,
        identity="vdn-restricted",
    )

    torch.testing.assert_close(selected, value.index_select(0, selector))
    torch.testing.assert_close(selected_measure, measure.index_select(0, selector))
    torch.testing.assert_close(child.rope_freqs, rope.index_select(1, selector))
    assert child.value_domain.indices == (103, 101)
    assert child.routing_position_domain.indices == (203, 201)
    assert child.value_domain.identity == "vdn-restricted"
    assert child.routing_position_domain.identity == "vdn-restricted"

    print("MiniMax-H3-Keyless VDN row-domain contract: OK")


if __name__ == "__main__":
    main()
