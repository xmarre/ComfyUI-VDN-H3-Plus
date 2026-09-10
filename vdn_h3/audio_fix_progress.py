"""Progress, CUDA-memory telemetry, and checkpoint helpers for audio-fix training.

The production MiniMax-H3 hidden stream is enormous: at the validated 48x84 latent
geometry one packed block boundary is about 0.53 GiB.  A naive independent checkpoint
around every one of the 50 H3 blocks therefore retains roughly 26.5 GiB of boundary
tensors before backward even starts.  The temporary attention/MLP graphs created while
recomputing a block then push a 96-GiB card into WDDM/shared-memory migration.

The trainer uses two-level segmented reentrant activation checkpointing instead:

* the initial forward checkpoints a short group of blocks as one unit, so only group
  boundaries survive to backward;
* when a group is recomputed with gradients enabled, each block inside that group is
  itself checkpointed, so only the current group's block boundaries coexist;
* every actual recomputation runs inside a copy of the complete forward Context so VDN
  layout/runtime ContextVars and the fail-closed audio-fix scope remain identical.

For 50 blocks with the default group size of five this reduces the large hidden-state
boundary set from 50 tensors to about 10 outer + at most 5 inner boundaries.  It adds
one extra block-forward equivalent during backward, but avoids the page-migration cliff
without changing model arithmetic, geometry, targets, or sampler behavior.
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm
from torch.utils.checkpoint import checkpoint

from vdn_h3.audio_fix_train import current_train_scope


DEFAULT_CHECKPOINT_GROUP_SIZE = 5


@dataclass
class _StepProgress:
    step: int
    total_steps: int
    train_index: int
    bar: object
    phase: str | None = None


def _gib(value: int | float) -> float:
    return float(value) / float(1 << 30)


def cuda_memory_snapshot() -> dict[str, float] | None:
    """Return allocator/driver-visible CUDA memory counters without synchronizing."""
    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    row = {
        "allocated_gib": _gib(torch.cuda.memory_allocated(device)),
        "reserved_gib": _gib(torch.cuda.memory_reserved(device)),
        "peak_allocated_gib": _gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": _gib(torch.cuda.max_memory_reserved(device)),
    }
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        row["free_gib"] = _gib(free_bytes)
        row["total_gib"] = _gib(total_bytes)
    except (RuntimeError, NotImplementedError):
        pass
    return row


def _memory_suffix() -> str:
    row = cuda_memory_snapshot()
    if row is None:
        return ""
    parts = [
        f"alloc={row['allocated_gib']:.1f}GiB",
        f"reserved={row['reserved_gib']:.1f}GiB",
        f"peak_alloc={row['peak_allocated_gib']:.1f}GiB",
        f"peak_reserved={row['peak_reserved_gib']:.1f}GiB",
    ]
    if "free_gib" in row:
        parts.append(f"cuda_free={row['free_gib']:.1f}GiB")
    return " | CUDA " + " ".join(parts)


class TrainingProgress:
    """Two-level progress display for optimizer steps and real H3 block work."""

    def __init__(self, *, num_blocks: int, total_steps: int, initial_step: int = 0,
                 disable: bool = False):
        self.num_blocks = int(num_blocks)
        self.total_steps = int(total_steps)
        self.disable = bool(disable)
        self.checkpoint_recompute_passes = 1
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

    def set_checkpoint_recompute_passes(self, passes: int):
        passes = int(passes)
        if passes < 0:
            raise ValueError("checkpoint recompute passes must be >= 0")
        self.checkpoint_recompute_passes = passes

    def start_step(self, *, step: int, train_index: int):
        self.close_work()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        # Work counted by the block wrapper:
        #   train_index rollout H3 calls
        #   + dense teacher + frozen student + train forward
        #   + checkpoint recomputation passes during backward.
        calls = int(train_index) + 3 + int(self.checkpoint_recompute_passes)
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
            + _memory_suffix()
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
            f"{type(exc).__name__}: {exc}" + _memory_suffix()
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


def _transformer_options(args, kwargs):
    options = kwargs.get("transformer_options")
    if options is None and len(args) >= 5 and isinstance(args[4], dict):
        options = args[4]
    return options if isinstance(options, dict) else {}


def _reject_block_replacements(args, kwargs):
    options = _transformer_options(args, kwargs)
    dit = options.get("patches_replace", {}).get("dit", {})
    if dit:
        raise RuntimeError(
            "segmented audio-fix checkpointing cannot run with dit block replacements; "
            "canonical training must remain exact all-actual H3 without Spectrum or "
            "Progressive-Handoff block replacement"
        )


def install_scope_safe_block_checkpointing(
    model,
    bank,
    progress=None,
    scope_getter=current_train_scope,
    group_size: int = DEFAULT_CHECKPOINT_GROUP_SIZE,
):
    """Install segmented two-level checkpointing without changing H3 block topology.

    H3's normal outer loop still sees the original 50 ``model.blocks`` entries.  The
    first wrapper in each group executes that whole group under one outer reentrant
    checkpoint; the remaining wrappers in that group become identities for that one
    checkpointed training pass because their blocks were already executed by the group
    start.  Outside the audio-fix training scope every block behaves exactly as before.

    During outer-group backward recomputation, each block is wrapped in an inner
    reentrant checkpoint.  Consequently the initial forward retains only one hidden
    tensor per group, and backward retains at most one group's per-block boundaries at
    a time instead of all 50.  Reentrant checkpointing is intentional here: its initial
    execution is under ``no_grad``, so the large per-block surrogate/intermediate graph
    is not retained.

    If the first group input does not require gradients, a detached view with
    ``requires_grad=True`` is used solely as the checkpoint carrier.  Nothing trainable
    exists before H3 block 0 in this audio-fix design, so severing that otherwise-dead
    upstream gradient cannot change trained parameters or forward values.

    ``bank`` is retained in the signature for API compatibility.  The returned
    ``(block, original_forward)`` list remains compatible with
    ``audio_fix_train.restore_block_checkpointing``.
    """
    del bank
    blocks = list(model.blocks)
    if not blocks:
        return []
    group_size = int(group_size)
    if group_size < 1:
        raise ValueError("checkpoint group_size must be >= 1")

    originals = [(block, block.forward) for block in blocks]
    original_by_index = [original for _block, original in originals]
    group_start_by_index = {
        index: (index // group_size) * group_size for index in range(len(blocks))
    }

    if progress is not None:
        # Outer group recomputation + inner per-block recomputation.
        progress.set_checkpoint_recompute_passes(2)

    for index, block in enumerate(blocks):
        original = original_by_index[index]
        group_start = group_start_by_index[index]
        group_stop = min(group_start + group_size, len(blocks))

        def wrapped(*args, _index=index, _original=original,
                    _group_start=group_start, _group_stop=group_stop, **kwargs):
            scope = scope_getter()
            checkpointed = (
                scope is not None and scope.enabled and scope.checkpoint_blocks
            )
            if not checkpointed:
                if progress is not None:
                    progress.block_done()
                return _original(*args, **kwargs)

            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("segmented H3 checkpoint expected Tensor block input")
            _reject_block_replacements(args, kwargs)

            # The group start already executes every original block in this segment.
            # The normal MiniMax-H3 loop still invokes later ModuleList entries, so
            # those wrappers must be exact identities for this checkpointed pass.
            if _index != _group_start:
                return args[0]

            group_context = contextvars.copy_context()
            tail_args = args[1:]
            group_originals = original_by_index[_group_start:_group_stop]

            def run_group(h):
                # Initial outer-checkpoint forward runs under no_grad.  During outer
                # recomputation grad mode is enabled; introduce inner per-block
                # checkpoints only then so their boundaries live for one group only.
                if not torch.is_grad_enabled():
                    for group_original in group_originals:
                        if progress is not None:
                            progress.block_done()
                        h = group_original(h, *tail_args, **kwargs)
                    return h

                for group_original in group_originals:
                    inner_context = contextvars.copy_context()

                    def run_one(block_input, _group_original=group_original,
                                _inner_context=inner_context):
                        if progress is not None:
                            progress.block_done()
                        return _inner_context.run(
                            _group_original, block_input, *tail_args, **kwargs
                        )

                    if not h.requires_grad:
                        h = h.detach().requires_grad_(True)
                    h = checkpoint(run_one, h, use_reentrant=True)
                return h

            checkpoint_input = args[0]
            if not checkpoint_input.requires_grad:
                checkpoint_input = checkpoint_input.detach().requires_grad_(True)

            def run_group_captured(h):
                return group_context.run(run_group, h)

            return checkpoint(
                run_group_captured,
                checkpoint_input,
                use_reentrant=True,
            )

        block.forward = wrapped

    return originals


__all__ = [
    "DEFAULT_CHECKPOINT_GROUP_SIZE",
    "TrainingProgress",
    "cuda_memory_snapshot",
    "install_scope_safe_block_checkpointing",
]
