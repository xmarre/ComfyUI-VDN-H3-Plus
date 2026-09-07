from __future__ import annotations

import contextvars
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vdn_h3.audio_fix_progress import install_scope_safe_block_checkpointing


class _Bank:
    def __init__(self):
        self.scope_var = contextvars.ContextVar("test_audio_fix_scope", default=None)

    @contextmanager
    def scope(self, audio_start, audio_end, *, enabled=True, checkpoint_blocks=False):
        scope = SimpleNamespace(
            audio_start=int(audio_start),
            audio_end=int(audio_end),
            enabled=bool(enabled),
            checkpoint_blocks=bool(checkpoint_blocks),
        )
        token = self.scope_var.set(scope)
        try:
            yield
        finally:
            self.scope_var.reset(token)


class _Block(nn.Module):
    def __init__(self, bank, layout_var, calls, index=0):
        super().__init__()
        self.bank = bank
        self.layout_var = layout_var
        self.calls = calls
        self.index = int(index)
        self.weight = nn.Parameter(torch.tensor(0.75 + 0.03 * self.index))

    def forward(self, x):
        scope = self.bank.scope_var.get()
        if scope is None:
            raise RuntimeError("missing audio-fix scope")
        layout = self.layout_var.get()
        if layout is None:
            raise RuntimeError("missing VDN layout context")
        self.calls.append((
            self.index,
            scope.audio_start,
            scope.audio_end,
            scope.checkpoint_blocks,
            layout,
        ))
        # Nontrivial intermediates ensure checkpoint backward has work to recompute.
        scale = 1.25 if layout == "production-layout" else 0.5
        return torch.sin(x * self.weight + scale + self.index * 0.01) * x


class _Progress:
    def __init__(self):
        self.blocks = 0
        self.recompute_passes = None

    def block_done(self):
        self.blocks += 1

    def set_checkpoint_recompute_passes(self, passes):
        self.recompute_passes = int(passes)


def _run_blocks(model, x):
    for block in model.blocks:
        x = block(x)
    return x


def _restore(originals):
    for wrapped_block, original in originals:
        wrapped_block.forward = original


def test_checkpoint_recompute_restores_complete_forward_context_after_outer_wrappers_exit():
    bank = _Bank()
    layout_var = contextvars.ContextVar("test_vdn_layout", default=None)
    calls = []
    block = _Block(bank, layout_var, calls)
    model = SimpleNamespace(blocks=[block])
    progress = _Progress()

    originals = install_scope_safe_block_checkpointing(
        model,
        bank,
        progress=progress,
        scope_getter=bank.scope_var.get,
        group_size=1,
    )

    x = torch.randn(8, requires_grad=True)
    layout_token = layout_var.set("production-layout")
    try:
        with bank.scope(11, 19, enabled=True, checkpoint_blocks=True):
            y = _run_blocks(model, x)
    finally:
        layout_var.reset(layout_token)

    # Leave both outer ContextVar owners before backward. The outer-group and inner
    # block recomputations must each restore the captured production context.
    assert bank.scope_var.get() is None
    assert layout_var.get() is None
    y.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert block.weight.grad is not None
    assert torch.isfinite(block.weight.grad).all()
    assert progress.recompute_passes == 2
    # Initial forward + outer-group recompute + inner-block recompute.
    assert len(calls) >= 3
    assert all(call == (0, 11, 19, True, "production-layout") for call in calls)
    assert progress.blocks >= 3
    _restore(originals)


def test_segmented_checkpoint_matches_reference_forward_and_gradients():
    torch.manual_seed(17)

    ref_bank = _Bank()
    ref_layout = contextvars.ContextVar("reference_layout", default=None)
    ref_calls = []
    ref_blocks = [_Block(ref_bank, ref_layout, ref_calls, i) for i in range(6)]
    ref_model = SimpleNamespace(blocks=ref_blocks)

    ckpt_bank = _Bank()
    ckpt_layout = contextvars.ContextVar("checkpoint_layout", default=None)
    ckpt_calls = []
    ckpt_blocks = [_Block(ckpt_bank, ckpt_layout, ckpt_calls, i) for i in range(6)]
    ckpt_model = SimpleNamespace(blocks=ckpt_blocks)
    for ref_block, ckpt_block in zip(ref_blocks, ckpt_blocks):
        ckpt_block.load_state_dict(ref_block.state_dict())

    x_ref = torch.randn(16, requires_grad=True)
    x_ckpt = x_ref.detach().clone().requires_grad_(True)

    ref_token = ref_layout.set("production-layout")
    try:
        with ref_bank.scope(3, 9, enabled=True, checkpoint_blocks=False):
            y_ref = _run_blocks(ref_model, x_ref)
            y_ref.square().mean().backward()
    finally:
        ref_layout.reset(ref_token)

    progress = _Progress()
    originals = install_scope_safe_block_checkpointing(
        ckpt_model,
        ckpt_bank,
        progress=progress,
        scope_getter=ckpt_bank.scope_var.get,
        group_size=2,
    )
    ckpt_token = ckpt_layout.set("production-layout")
    try:
        with ckpt_bank.scope(3, 9, enabled=True, checkpoint_blocks=True):
            y_ckpt = _run_blocks(ckpt_model, x_ckpt)
    finally:
        ckpt_layout.reset(ckpt_token)

    # Backward happens after the publishing contexts have unwound, exactly like the
    # workstation trainer.
    y_ckpt.square().mean().backward()

    assert torch.equal(y_ckpt.detach(), y_ref.detach())
    assert torch.allclose(x_ckpt.grad, x_ref.grad, rtol=1e-6, atol=1e-7)
    for ref_block, ckpt_block in zip(ref_blocks, ckpt_blocks):
        assert torch.allclose(
            ckpt_block.weight.grad, ref_block.weight.grad, rtol=1e-6, atol=1e-7
        )

    # Every block executes once in the initial group forward, once in the outer group
    # recompute, and once more in its inner checkpoint recompute.
    assert progress.recompute_passes == 2
    assert progress.blocks == 18
    assert len(ckpt_calls) == 18
    assert all(call[1:] == (3, 9, True, "production-layout") for call in ckpt_calls)
    _restore(originals)


def test_segmented_checkpoint_keeps_parameter_gradients_when_first_input_is_frozen():
    bank = _Bank()
    layout_var = contextvars.ContextVar("frozen_input_layout", default=None)
    calls = []
    blocks = [_Block(bank, layout_var, calls, i) for i in range(4)]
    model = SimpleNamespace(blocks=blocks)
    originals = install_scope_safe_block_checkpointing(
        model,
        bank,
        scope_getter=bank.scope_var.get,
        group_size=2,
    )

    # Production H3 block-0 input comes from frozen embedding/projection modules and
    # need not require grad. The checkpoint carrier must still allow LoRA/sidecar-like
    # parameters inside the blocks to receive gradients.
    x = torch.randn(12, requires_grad=False)
    token = layout_var.set("production-layout")
    try:
        with bank.scope(2, 7, enabled=True, checkpoint_blocks=True):
            y = _run_blocks(model, x)
    finally:
        layout_var.reset(token)
    y.sum().backward()

    for block in blocks:
        assert block.weight.grad is not None
        assert torch.isfinite(block.weight.grad).all()
        assert float(block.weight.grad.abs()) > 0.0
    _restore(originals)


def test_segmented_checkpoint_rejects_dit_block_replacements():
    bank = _Bank()
    layout_var = contextvars.ContextVar("replacement_layout", default=None)
    block = _Block(bank, layout_var, [], 0)
    model = SimpleNamespace(blocks=[block])
    originals = install_scope_safe_block_checkpointing(
        model,
        bank,
        scope_getter=bank.scope_var.get,
        group_size=1,
    )
    options = {
        "patches_replace": {"dit": {("double_block", 0): object()}},
    }
    token = layout_var.set("production-layout")
    try:
        with bank.scope(0, 1, enabled=True, checkpoint_blocks=True):
            with pytest.raises(RuntimeError, match="cannot run with dit block replacements"):
                block(torch.randn(4), transformer_options=options)
    finally:
        layout_var.reset(token)
        _restore(originals)
