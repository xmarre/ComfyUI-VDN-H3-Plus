from __future__ import annotations

from types import SimpleNamespace

import pytest

from vdn_h3 import keyless_compat
from vdn_h3 import nodes


class Contract:
    api = 1
    architecture = "h3_keyless_core50_v1"
    core_blocks = 50
    token_refiner = "native_qkv"
    token_refiner_blocks = 2
    heads = 56
    head_dim = 128
    inner_dim = 7168
    hidden_size = 5376
    routing_source = "value"
    retrieval_source = "raw_projected_value"
    routing_norm = "rmsnorm"
    routing_norm_epsilon = 1e-5
    rope_policy = "h3_split_half_96_v1"
    qv_order = "q_effective;v"
    projection_attr = "qv_proj"
    checkpoint_format_version = 1
    provenance_identity = "keyless-stage-a"

    def identity(self):
        return (
            self.api,
            self.architecture,
            self.checkpoint_format_version,
            self.qv_order,
            self.heads,
            self.head_dim,
            self.inner_dim,
            self.routing_source,
            self.retrieval_source,
            self.rope_policy,
            self.provenance_identity,
        )


def _keyless_model(contract=None):
    model = SimpleNamespace()
    setattr(model, keyless_compat.KEYLESS_CONTRACT_KEY, contract or Contract())
    return model


def _native_model(blocks=2):
    return SimpleNamespace(
        blocks=[
            SimpleNamespace(
                attn=SimpleNamespace(
                    qkv_proj=object(),
                    q_norm=object(),
                    k_norm=object(),
                )
            )
            for _ in range(blocks)
        ]
    )


def test_recognized_keyless_contract_is_rejected_as_qkv_checkpoint_base():
    model = _keyless_model()
    semantic = keyless_compat.advertised_keyless_identity(model)
    assert semantic[0] == keyless_compat.KEYLESS_CONTRACT_KEY
    assert semantic[-1] == "keyless-stage-a"

    with pytest.raises(
        keyless_compat.VDNBaseCompatibilityError,
        match="separately trained checkpoint",
    ):
        keyless_compat.require_released_qkv_base(model)


def test_malformed_advertised_keyless_contract_fails_closed():
    contract = Contract()
    contract.routing_source = "key"
    with pytest.raises(
        keyless_compat.VDNBaseCompatibilityError,
        match="routing_source",
    ):
        keyless_compat.advertised_keyless_identity(_keyless_model(contract))


def test_native_qkv_base_remains_accepted():
    model = _native_model(3)
    blocks = keyless_compat.require_released_qkv_base(model)
    assert blocks is model.blocks
    assert len(blocks) == 3


def test_partial_qkv_base_fails_on_exact_block_index():
    model = _native_model(3)
    del model.blocks[1].attn.k_norm
    with pytest.raises(
        keyless_compat.VDNBaseCompatibilityError,
        match="block 1 is missing q_norm/k_norm",
    ):
        keyless_compat.require_released_qkv_base(model)


def test_apply_node_rejects_keyless_before_checkpoint_io(monkeypatch):
    inner = _keyless_model()
    model = SimpleNamespace(get_model_object=lambda name: inner)
    resolved = []

    def forbidden_resolve(name):
        resolved.append(name)
        raise AssertionError("checkpoint discovery must not run for an incompatible base")

    monkeypatch.setattr(nodes.spec, "resolve_vdn_checkpoint", forbidden_resolve)
    with pytest.raises(
        keyless_compat.VDNBaseCompatibilityError,
        match="cannot be applied to h3_keyless_core50_v1",
    ):
        nodes._apply_vdn(
            model,
            "released-stage",
            1.0,
            "merge",
            "stream",
            "grouped",
            False,
            apply_turbo_adapter=True,
            retain_buffers="off",
        )
    assert resolved == []
