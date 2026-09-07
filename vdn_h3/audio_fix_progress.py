"""Progress and checkpoint helpers for the standalone audio-fix GPU trainer.

The MiniMax-H3 training hooks are deliberately fail-closed: every target projection
must execute under an explicit ``TrainAudioScope``. PyTorch non-reentrant activation
checkpointing recomputes a block later, after the diffusion-model wrapper that owns
VDN's ContextVars has unwound. The recompute therefore has to restore the *complete*
forward Context, not only the audio-fix scope. In particular, VDN's published layout
and execution-local runtime state live in ContextVars too; losing the layout makes the
attention wrapper silently take the native dense fallback and produces a different
autograd graph on recompute.

This module also exposes a lightweight tqdm reporter that counts actual H3 transformer
block invocations. Phase transitions are written as durable log lines so a traceback
still says where a long production-geometry run failed even after tqdm clears its
transient work bar.
"""
from __future__ import annotations

import contextvars
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
    phase: str | None = None


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
        text = str(text)
        if self._current.phase == text:
            return
        self._current.phase = text
        self._current.bar.set_postfix_str(text, refresh=True)
        self.write(
            f"[audio-fix step {self._current.step}/{self._current.total_steps}] "
            f"{text} ({self._current.bar.n}/{self._current.bar.total} blocks)"
        )

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

    def fail(self, exc: BaseException):
        if self._current is None:
            self.write(f"[audio-fix] FAILED before step work: {type(exc).__name__}: {exc}")
            return
        phase = self._current.phase or "unknown phase"
        self.write(
            f"[audio-fix step {self._current.step}/{self._current.total_steps}] FAILED "
            f"during {phase} at {self._current.bar.n}/{self._current.bar.total} blocks: "
            f"{type(exc).__name__}: {exc}"
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
    """Checkpoint H3 blocks while restoring the complete forward Context on recompute.

    The audio-fix scope is used to decide whether this block should be checkpointed,
    but it is not the only state that must survive. MiniMax-H3 VDN also publishes its
    packed layout and execution-local runtime lease through ContextVars owned by outer
    ModelPatcher wrappers. Those wrappers are no longer on the Python stack when
    backward asks checkpointing to recompute an individual block.

    Capture ``contextvars.copy_context()`` at checkpoint creation and run both the
    original forward and any later recomputation inside that captured Context. This
    preserves the exact same VDN-vs-native branch selection and fail-closed audio-fix
    scope without teaching this helper about every individual VDN ContextVar.

    ``scope_getter`` remains injectable for focused CPU tests. The returned
    ``(block, original_forward)`` list is compatible with
    ``audio_fix_train.restore_block_checkpointing``.
    """
    originals = []
    for block in model.blocks:
        original = block.forward

        def wrapped(*args, _original=original, **kwargs):
            scope = scope_getter()

            if (scope is not None and scope.enabled and scope.checkpoint_blocks):
                forward_context = contextvars.copy_context()

                def run_captured(*inner_args, **inner_kwargs):
                    # Count the actual block invocation, including checkpoint
                    # recomputation. Do it before the body so a failing block still
                    # leaves useful progress.
                    if progress is not None:
                        progress.block_done()
                    return forward_context.run(
                        _original, *inner_args, **inner_kwargs
                    )

                return checkpoint(
                    run_captured, *args, use_reentrant=False, **kwargs
                )

            if progress is not None:
                progress.block_done()
            return _original(*args, **kwargs)

        block.forward = wrapped
        originals.append((block, original))
    return originals


__all__ = [
    "TrainingProgress",
    "install_scope_safe_block_checkpointing",
]
