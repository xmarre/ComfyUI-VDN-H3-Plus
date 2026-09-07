"""Progress and checkpoint helpers for the standalone audio-fix GPU trainer.

The MiniMax-H3 training hooks are deliberately fail-closed: every target projection
must execute under an explicit ``TrainAudioScope``. PyTorch non-reentrant activation
checkpointing can recompute a block later after the caller's ContextVar scope has
unwound, so the recompute function explicitly re-enters the scope captured when the
checkpoint was created.

This module also exposes a lightweight tqdm reporter that counts actual H3 transformer
block invocations. That makes long production-geometry rollouts/backward passes visible
instead of leaving the terminal silent for minutes.
"""
from __future__ import annotations

from dataclasses import dataclass

from tqdm.auto import tqdm
from torch.utils.checkpoint import checkpoint

from vdn_h3.audio_fix_train import current_train_scope


@dataclass
class _StepProgress:
    step: int
    total_steps: int
    train_index: int
    bar: object


class TrainingProgress:
    """Two-level progress display for optimizer steps and real H3 block work."""

    def __init__(self, *, num_blocks: int, total_steps: int, initial_step: int = 0,
                 disable: bool = False):
        self.num_blocks = int(num_blocks)
        self.total_steps = int(total_steps)
        self.disable = bool(disable)
        self._current: _StepProgress | None = None
        self.steps = tqdm(
            total=self.total_steps,
            initial=int(initial_step),
            desc="audio-fix train",
            unit="step",
            dynamic_ncols=True,
            leave=True,
            position=0,
            disable=self.disable,
        )

    def start_step(self, *, step: int, train_index: int):
        self.close_work()
        # Work counted by the block wrapper:
        #   train_index rollout H3 calls
        #   + dense teacher
        #   + frozen student
        #   + train forward
        #   + checkpoint recompute during backward
        calls = int(train_index) + 4
        total_blocks = calls * self.num_blocks
        bar = tqdm(
            total=total_blocks,
            desc=f"step {step}/{self.total_steps}",
            unit="block",
            dynamic_ncols=True,
            leave=False,
            position=1,
            mininterval=0.25,
            disable=self.disable,
        )
        self._current = _StepProgress(
            step=int(step), total_steps=self.total_steps,
            train_index=int(train_index), bar=bar,
        )
        self.phase(f"rollout 0→{train_index} ({train_index} H3 evals)")

    def phase(self, text: str):
        if self._current is None:
            return
        self._current.bar.set_postfix_str(str(text), refresh=True)

    def block_done(self):
        if self._current is not None:
            self._current.bar.update(1)

    def finish_step(self, *, loss: float, peak_gib: float, seconds: float):
        self.close_work()
        self.steps.update(1)
        self.steps.set_postfix(
            loss=f"{float(loss):.5g}",
            peak=f"{float(peak_gib):.1f}GiB",
            sec=f"{float(seconds):.1f}",
            refresh=True,
        )

    def close_work(self):
        if self._current is not None:
            self._current.bar.close()
            self._current = None

    def write(self, text: str):
        tqdm.write(str(text))

    def close(self):
        self.close_work()
        self.steps.close()


def install_scope_safe_block_checkpointing(model, bank, progress=None,
                                            scope_getter=current_train_scope):
    """Checkpoint H3 blocks while preserving the audio-fix scope on recompute.

    The scope source is the authoritative ``vdn_h3.audio_fix_train`` ContextVar, not
    an incidental attribute on whichever CLI module imported the training helpers.
    ``scope_getter`` is injectable only for focused CPU tests.

    The returned ``(block, original_forward)`` list is compatible with
    ``audio_fix_train.restore_block_checkpointing``.
    """
    originals = []
    for block in model.blocks:
        original = block.forward

        def wrapped(*args, _original=original, **kwargs):
            scope = scope_getter()

            def run_scoped(*inner_args, **inner_kwargs):
                # Count the actual block invocation, including checkpoint recomputation.
                # Do it before the body so a failing block still leaves useful progress.
                if progress is not None:
                    progress.block_done()
                if scope is None:
                    return _original(*inner_args, **inner_kwargs)
                with bank.scope(
                    scope.audio_start,
                    scope.audio_end,
                    enabled=scope.enabled,
                    checkpoint_blocks=scope.checkpoint_blocks,
                ):
                    return _original(*inner_args, **inner_kwargs)

            if (scope is not None and scope.enabled and scope.checkpoint_blocks):
                return checkpoint(run_scoped, *args, use_reentrant=False, **kwargs)
            return run_scoped(*args, **kwargs)

        block.forward = wrapped
        originals.append((block, original))
    return originals


__all__ = [
    "TrainingProgress",
    "install_scope_safe_block_checkpointing",
]
