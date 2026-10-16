"""Opt-in CUDA event timing for actual VDN component boundaries.

Enable VDN_H3_PERF_PROFILE=1 on a ComfyUI process and capture the ordinary
vdn_h3 logger. It measures GPU stream duration, not Python wall time. No
synchronization occurs during the block; one synchronization is used after
the outer sample completes. The profiler never retains activations, weights,
Q/K/V or model objects. Disabled by default.
"""
from __future__ import annotations

from collections import defaultdict
from contextvars import ContextVar
import logging
import os

import torch

_log = logging.getLogger("comfy.vdn")
_ACTIVE = ContextVar("vdn_h3_cuda_perf_scope", default=None)
ENABLED = os.environ.get("VDN_H3_PERF_PROFILE") == "1"


class Sample:
    def __init__(self):
        self.spans = []
        self.counts = defaultdict(int)
        self.device = None

    def start_forward(self, stage, block_index, device):
        if torch.device(device).type != "cuda":
            return None
        if self.device is None:
            self.device = torch.device(device)
        elif self.device != torch.device(device):
            raise RuntimeError("VDN CUDA profiling spans cannot cross devices")
        stage = stage if isinstance(stage, str) and stage else "standalone"
        self.counts[stage] += 1
        return Forward(self, stage, block_index)

    def report(self):
        if not self.spans:
            _log.info("[vdn] CUDA component audit no instrumented forward calls")
            return
        # This is an opt-in post-sample synchronization only. It MUST NOT be
        # invoked from an active model-forward or projected-QKV hot path.
        torch.cuda.synchronize(self.device)
        totals = defaultdict(lambda: defaultdict(float))
        for stage, phase, start, end in self.spans:
            totals[stage][phase] += start.elapsed_time(end)
        for stage in sorted(totals):
            times = totals[stage]
            active_ms = sum(times.values())
            _log.warning(
                "[vdn] CUDA component audit stage=%s forwards=%d "
                "qkv_projection_ms=%.3f linear_readout_ms=%.3f "
                "qk_norm_rope_ms=%.3f grouped_softmax_ms=%.3f "
                "output_projection_ms=%.3f component_cuda_total_ms=%.3f "
                "caveat=stream_duration_sum_not_wall_clock",
                stage, self.counts[stage],
                times["qkv_projection"], times["linear_readout"],
                times["qk_norm_rope"], times["grouped_softmax"],
                times["output_projection"], active_ms,
            )
        self.spans.clear()
        self.counts.clear()


class Forward:
    def __init__(self, sample, stage, block_index):
        self.sample = sample
        self.stage = stage
        self.block_index = block_index
        self.last = None

    def mark(self, completed_phase=None):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        if self.last is not None:
            phase, start = self.last
            self.sample.spans.append((self.stage, phase, start, event))
        # completed_phase is the label for the NEXT interval, not a tensor.
        self.last = (completed_phase, event) if completed_phase else None


def enter():
    """Create the outermost sample scope only; nested Flow stages reuse it."""
    if not ENABLED or _ACTIVE.get() is not None:
        return None
    sample = Sample()
    token = _ACTIVE.set(sample)
    return sample, token


def exit_scope(entered):
    if entered is None:
        return
    sample, token = entered
    try:
        try:
            sample.report()
        except Exception:
            # Profiling must never change sampling success or mask its failure.
            _log.exception("[vdn] CUDA component audit failed")
    finally:
        _ACTIVE.reset(token)


def forward(stage, block_index, device):
    sample = _ACTIVE.get()
    if sample is None:
        return None
    return sample.start_forward(stage, block_index, device)
