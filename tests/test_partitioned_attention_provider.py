"""Partitioned dispatch transports weighted rectangular domains independently of Sol."""
import builtins

import pytest
import torch

from vdn_h3.softmax_provider import PARTITIONED_PROVIDER_KEY, partitioned_attention


def test_partitioned_provider_receives_complete_request_without_sol(monkeypatch):
    original_import = builtins.__import__

    def no_sol(name, *args, **kwargs):
        if name == "sol_h3" or name.startswith("sol_h3."):
            raise AssertionError("native partitioned provider must not import Sol")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sol)
    seen = []
    q = torch.randn(3, 2, 8)
    k, v = torch.randn(7, 2, 8), torch.randn(7, 2, 8)

    def provider(q_arg, k_arg, v_arg, **kwargs):
        seen.append((q_arg, k_arg, v_arg, kwargs))
        return q_arg.clone()

    options = {PARTITIONED_PROVIDER_KEY: provider}
    kwargs = dict(transformer_options=options, block_index=2, kind="local", scale=8**-0.5,
                  sink_rows=1, prefix_k_range=(1, 4), prefix_log_key_measure=-0.7,
                  semantic_digest="a" * 64, query_position_map={"layout": "mapped"}, force_dense=False)
    result = partitioned_attention(q, k, v, **kwargs)
    assert seen[0][0] is q and seen[0][1] is k and seen[0][2] is v
    assert seen[0][3] == kwargs
    torch.testing.assert_close(result, q)


@pytest.mark.parametrize("provider", [None, lambda q, *_a, **_kw: q[:1],
                                     lambda q, *_a, **_kw: q.to(torch.float64)])
def test_invalid_partitioned_provider_does_not_reenter_sol(provider):
    q = torch.randn(3, 1, 8)
    with pytest.raises(RuntimeError, match="provider"):
        partitioned_attention(q, q, q, transformer_options={PARTITIONED_PROVIDER_KEY: provider})
