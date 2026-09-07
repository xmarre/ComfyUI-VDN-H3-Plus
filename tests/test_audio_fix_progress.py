from __future__ import annotations

import contextvars
from contextlib import contextmanager
from types import SimpleNamespace

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
    def __init__(self, bank, layout_var, calls):
        super().__init__()
        self.bank = bank
        self.layout_var = layout_var
        self.calls = calls

    def forward(self, x):
        scope = self.bank.scope_var.get()
        if scope is None:
            raise RuntimeError("missing audio-fix scope")
        layout = self.layout_var.get()
        if layout is None:
            raise RuntimeError("missing VDN layout context")
        self.calls.append((
            scope.audio_start,
            scope.audio_end,
            scope.checkpoint_blocks,
            layout,
        ))
        # Nontrivial intermediates ensure checkpoint backward has work to recompute.
        # The extra layout-dependent branch mimics VDN choosing a different attention
        # graph when its published layout disappears.
        scale = 1.25 if layout == "production-layout" else 0.5
        return torch.sin(x * x + scale) * x


class _Progress:
    def __init__(self):
        self.blocks = 0

    def block_done(self):
        self.blocks += 1


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
    )

    x = torch.randn(8, requires_grad=True)
    layout_token = layout_var.set("production-layout")
    try:
        with bank.scope(11, 19, enabled=True, checkpoint_blocks=True):
            y = block(x)
    finally:
        layout_var.reset(layout_token)

    # Deliberately leave *both* outer ContextVar owners before backward. This models
    # the real trainer: the diffusion-model layout wrapper and audio-fix scope have
    # unwound before a checkpointed block is recomputed.
    assert bank.scope_var.get() is None
    assert layout_var.get() is None
    y.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert len(calls) >= 2
    assert all(call == (11, 19, True, "production-layout") for call in calls)
    # One forward invocation plus at least one checkpoint recomputation invocation.
    assert progress.blocks >= 2

    for wrapped_block, original in originals:
        wrapped_block.forward = original
