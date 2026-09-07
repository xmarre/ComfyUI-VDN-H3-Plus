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
    def __init__(self, bank, calls):
        super().__init__()
        self.bank = bank
        self.calls = calls

    def forward(self, x):
        scope = self.bank.scope_var.get()
        if scope is None:
            raise RuntimeError("missing scope")
        self.calls.append((scope.audio_start, scope.audio_end, scope.checkpoint_blocks))
        # Nontrivial intermediates ensure checkpoint backward has work to recompute.
        return torch.sin(x * x + 0.25) * x


class _Progress:
    def __init__(self):
        self.blocks = 0

    def block_done(self):
        self.blocks += 1


def test_checkpoint_recompute_reenters_captured_audio_scope_after_outer_scope_exits():
    bank = _Bank()
    calls = []
    block = _Block(bank, calls)
    model = SimpleNamespace(blocks=[block])
    progress = _Progress()

    originals = install_scope_safe_block_checkpointing(
        model,
        bank,
        progress=progress,
        scope_getter=bank.scope_var.get,
    )

    x = torch.randn(8, requires_grad=True)
    with bank.scope(11, 19, enabled=True, checkpoint_blocks=True):
        y = block(x)

    # Deliberately leave the caller's scope before backward. The checkpoint helper must
    # restore the captured scope for recomputation instead of weakening fail-closed hooks.
    assert bank.scope_var.get() is None
    y.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert len(calls) >= 2
    assert all(call == (11, 19, True) for call in calls)
    # One forward invocation plus at least one checkpoint recomputation invocation.
    assert progress.blocks >= 2

    for wrapped_block, original in originals:
        wrapped_block.forward = original
