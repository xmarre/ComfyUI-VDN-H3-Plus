"""State-owned transient/retained runtime resources for VDN-H3.

The upstream v1.4 performance work showed that repeatedly allocating scan banks,
window gather buffers and activation copies costs meaningful time. Those buffers are
useful, but process-global CUDA caches make ownership and cancellation ambiguous.

This module keeps the optimization local to one Apply-VDN result. The primary pool is
leased for one diffusion-model execution at a time. A nested or concurrent execution
that cannot acquire that lease receives an isolated transient pool instead, so shared
scratch is never raced across ModelPatcher clones or threads.

Branch *weights* do not live here. They remain either bounded streamed checkpoint
descriptors or a ComfyUI-managed additional ModelPatcher. The only process-global
runtime object is one bounded worker executor; it owns no model tensors or cache.

Cross-stream ownership is explicit even with ``cudaMallocAsync``. PyTorch's async
allocator still uses ``record_stream`` to learn about streams other than an
allocation's creation stream before ``cudaFreeAsync`` can safely recycle storage.
VDN prefetch allocates on a dedicated producer stream and consumes on the caller's
stream, so this is exactly the cross-stream case that must be recorded. The warning
about redundant ``record_stream`` calls under ``cudaMallocAsync`` concerns recording
the allocation/creation stream itself; it does not make side-stream lifetime
tracking unnecessary.
"""
from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import contextvars
import logging
import threading

import torch


_log = logging.getLogger("comfy.vdn")
_MAX_SCAN_BANKS = 1
_MAX_DELTA_SCRATCH = 1
_MAX_WINDOW_PLANS = 8
_MAX_KV_SCRATCH = 1
_MAX_ACTIVATION_SCRATCH = 1
_ACTIVE_BUFFERS = contextvars.ContextVar("vdn_active_runtime_buffers", default=None)
# One worker is enough for one-block lookahead. It never stores branch weights itself;
# each RuntimeBuffers owns at most one Future/result and drops it on reset.
_PREFETCH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="vdn-branch-prefetch")


def current_runtime_buffers():
    """Return the pool leased by the current diffusion-model execution, if any."""
    return _ACTIVE_BUFFERS.get()


def _bounded_put(mapping, key, value, limit):
    if key in mapping:
        mapping.pop(key)
    mapping[key] = value
    while len(mapping) > limit:
        mapping.popitem(last=False)
    return value


class _StreamPrefetcher:
    """One cancellable in-flight block transfer; no per-state worker/thread leak."""

    def __init__(self):
        self._lock = threading.Lock()
        self._generation = 0
        self._future = None
        self._index = None

    @staticmethod
    def _record_stream(tensor, stream):
        """Record every CUDA storage consumed on a stream other than its producer.

        QuantizedTensor is a wrapper rather than the allocation that kernels read;
        record its qdata plus any scale/original-weight/bias tensors directly. A
        registration failure is a correctness failure: silently continuing would
        allow either allocator backend to recycle storage before the consumer ends.
        """
        inner = getattr(tensor, "_qdata", None)
        seen = [inner if isinstance(inner, torch.Tensor) else tensor]
        params = getattr(tensor, "_params", None)
        for name in ("scale", "orig_weight", "bias"):
            child = getattr(params, name, None)
            if isinstance(child, torch.Tensor):
                seen.append(child)
        for item in seen:
            item.record_stream(stream)

    @staticmethod
    def _fetch(generation, index, fetch):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            weights = fetch()
            event = torch.cuda.Event()
            event.record(stream)
        return generation, index, weights, event

    def request(self, index, fetch):
        with self._lock:
            # A completed result is consumed only by take(). Do not overwrite it with
            # a later block before the execution reaches its intended target.
            if self._future is not None:
                return
            generation = self._generation
            self._index = index
            self._future = _PREFETCH_EXECUTOR.submit(
                self._fetch, generation, index, fetch)

    def take(self, index):
        with self._lock:
            future = self._future
            target = self._index
            generation = self._generation
        if future is None or target != index:
            return None
        try:
            got_generation, got_index, weights, event = future.result()
        except Exception as exc:
            _log.warning(
                "[vdn] branch prefetch for block %s failed (%s); using synchronous "
                "streaming for this block", index, exc)
            with self._lock:
                if self._future is future:
                    self._future = None
                    self._index = None
            return None
        with self._lock:
            if (self._future is future and got_generation == self._generation
                    and generation == self._generation and got_index == index):
                self._future = None
                self._index = None
            else:
                return None
        current = torch.cuda.current_stream()
        current.wait_event(event)
        for tensor in weights.values():
            self._record_stream(tensor, current)
        return weights

    def reset(self):
        with self._lock:
            self._generation += 1
            future = self._future
            self._future = None
            self._index = None
        if future is not None:
            future.cancel()


class RuntimeBuffers:
    """Scratch owned by one execution pool; no model/checkpoint weights are cached."""

    def __init__(self, retain: bool):
        self.retain = bool(retain)
        self._scan = collections.OrderedDict()
        self._delta = collections.OrderedDict()
        self._plans = collections.OrderedDict()
        self._kv = collections.OrderedDict()
        self._activations = collections.OrderedDict()
        self._prefetcher = None

    def delta_scratch(self, shape, device):
        if not self.retain:
            return torch.empty(shape, dtype=torch.float32, device=device)
        key = (tuple(shape), str(device))
        hit = self._delta.get(key)
        if hit is None:
            hit = torch.empty(shape, dtype=torch.float32, device=device)
            _bounded_put(self._delta, key, hit, _MAX_DELTA_SCRATCH)
        else:
            self._delta.move_to_end(key)
        return hit

    def scan_banks(self, num_frames, state_shape, dtype, device):
        shape = (num_frames, *state_shape)
        if not self.retain:
            prefix = torch.empty(shape, dtype=dtype, device=device)
            return prefix, torch.empty_like(prefix)
        key = (num_frames, tuple(state_shape), str(device), dtype)
        hit = self._scan.get(key)
        if hit is None:
            prefix = torch.empty(shape, dtype=dtype, device=device)
            hit = (prefix, torch.empty_like(prefix))
            _bounded_put(self._scan, key, hit, _MAX_SCAN_BANKS)
        else:
            self._scan.move_to_end(key)
        return hit

    def activation_scratch(self, video_rows, text_rows, heads, head_dim, device, dtype):
        if not self.retain:
            return None
        key = (video_rows, text_rows, heads, head_dim, str(device), dtype)
        hit = self._activations.get(key)
        if hit is None:
            vshape = (video_rows, heads, head_dim)
            tshape = (text_rows, heads, head_dim)
            hit = {
                "q": torch.empty(vshape, device=device, dtype=dtype),
                "k": torch.empty(vshape, device=device, dtype=dtype),
                "v": torch.empty(vshape, device=device, dtype=dtype),
                "tk": torch.empty(tshape, device=device, dtype=dtype),
                "tv": torch.empty(tshape, device=device, dtype=dtype),
            }
            _bounded_put(self._activations, key, hit, _MAX_ACTIVATION_SCRATCH)
        else:
            self._activations.move_to_end(key)
        return hit

    def window_plan(self, key, builder):
        if not self.retain:
            return builder()
        hit = self._plans.get(key)
        if hit is None:
            hit = builder()
            _bounded_put(self._plans, key, hit, _MAX_WINDOW_PLANS)
        else:
            self._plans.move_to_end(key)
        return hit

    def kv_scratch(self, rows, heads, head_dim, device, dtype):
        shape = (rows, heads, head_dim)
        if not self.retain:
            return (
                torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype),
            )
        key = (str(device), dtype, heads, head_dim)
        pair = self._kv.get(key)
        need = rows * heads * head_dim
        if pair is None or pair[0].numel() < need:
            pair = (
                torch.empty(need, device=device, dtype=dtype),
                torch.empty(need, device=device, dtype=dtype),
            )
            _bounded_put(self._kv, key, pair, _MAX_KV_SCRATCH)
        else:
            self._kv.move_to_end(key)
        return pair[0][:need].view(shape), pair[1][:need].view(shape)

    def prefetch_take(self, index):
        if not self.retain or not torch.cuda.is_available():
            return None
        if self._prefetcher is None:
            return None
        return self._prefetcher.take(index)

    def prefetch_request(self, index, fetch):
        if not self.retain or not torch.cuda.is_available():
            return
        if self._prefetcher is None:
            self._prefetcher = _StreamPrefetcher()
        self._prefetcher.request(index, fetch)

    def clear(self):
        self._scan.clear()
        self._delta.clear()
        self._plans.clear()
        self._kv.clear()
        self._activations.clear()
        if self._prefetcher is not None:
            self._prefetcher.reset()

    def retained_counts(self):
        return {
            "scan": len(self._scan),
            "delta": len(self._delta),
            "plans": len(self._plans),
            "kv": len(self._kv),
            "activations": len(self._activations),
        }


class RuntimeBufferOwner:
    """Lease one retained pool; concurrent/nested executions get transient scratch."""

    def __init__(self, retain: bool):
        self.retain = bool(retain)
        self._primary = RuntimeBuffers(self.retain)
        self._lease = threading.Lock()
        self._active = contextvars.ContextVar(
            f"vdn_runtime_buffers_{id(self)}", default=None)

    @contextlib.contextmanager
    def execution(self):
        primary = self.retain and self._lease.acquire(blocking=False)
        buffers = self._primary if primary else RuntimeBuffers(False)
        token = self._active.set(buffers)
        global_token = _ACTIVE_BUFFERS.set(buffers)
        try:
            yield buffers
        finally:
            _ACTIVE_BUFFERS.reset(global_token)
            self._active.reset(token)
            if primary:
                self._lease.release()
            else:
                buffers.clear()

    def current(self):
        return self._active.get()

    def clear(self):
        self._primary.clear()
