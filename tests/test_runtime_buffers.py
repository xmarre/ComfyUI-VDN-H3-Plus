from __future__ import annotations

from types import SimpleNamespace
import gc
import weakref

import torch

from vdn_h3 import branch as B
from vdn_h3 import window as W
from vdn_h3.hybrid import VDNState
from vdn_h3.retained import (
    RuntimeLinearBranch,
    run_scans_runtime,
    window_softmax_grouped_runtime,
)
from vdn_h3.runtime import RuntimeBufferOwner, current_runtime_buffers


def test_runtime_buffer_owner_reuses_primary_but_isolates_nested_execution():
    owner = RuntimeBufferOwner(True)
    with owner.execution() as outer:
        assert outer.retain
        assert current_runtime_buffers() is outer
        first = outer.activation_scratch(8, 3, 2, 4, "cpu", torch.float32)
        with owner.execution() as inner:
            assert not inner.retain
            assert inner is not outer
            assert current_runtime_buffers() is inner
        assert current_runtime_buffers() is outer
        second = outer.activation_scratch(8, 3, 2, 4, "cpu", torch.float32)
        assert first["q"].data_ptr() == second["q"].data_ptr()
    assert current_runtime_buffers() is None


def test_activation_scratch_evicts_old_geometry_before_replacement_allocation(monkeypatch):
    owner = RuntimeBufferOwner(True)
    with owner.execution() as resources:
        first = resources.activation_scratch(8, 3, 2, 4, "cpu", torch.float32)
        first_ref = weakref.ref(first["q"])
        del first
        gc.collect()
        assert first_ref() is not None  # retained by the cache

        original_empty = torch.empty
        observed = {"checked": False}

        def checked_empty(*args, **kwargs):
            if not observed["checked"]:
                gc.collect()
                assert first_ref() is None
                observed["checked"] = True
            return original_empty(*args, **kwargs)

        monkeypatch.setattr(torch, "empty", checked_empty)
        second = resources.activation_scratch(16, 3, 2, 4, "cpu", torch.float32)
        assert observed["checked"]
        assert second["q"].shape[0] == 16
        assert resources.retained_counts()["activations"] == 1


def test_kv_scratch_evicts_smaller_capacity_before_growth_allocation(monkeypatch):
    owner = RuntimeBufferOwner(True)
    with owner.execution() as resources:
        first_k, first_v = resources.kv_scratch(8, 2, 4, "cpu", torch.float32)
        backing_ref = weakref.ref(resources._kv[("cpu", torch.float32, 2, 4)][0])
        del first_k, first_v
        gc.collect()
        assert backing_ref() is not None  # retained by the cache

        original_empty = torch.empty
        observed = {"checked": False}

        def checked_empty(*args, **kwargs):
            if not observed["checked"]:
                gc.collect()
                assert backing_ref() is None
                observed["checked"] = True
            return original_empty(*args, **kwargs)

        monkeypatch.setattr(torch, "empty", checked_empty)
        grown_k, grown_v = resources.kv_scratch(16, 2, 4, "cpu", torch.float32)
        assert observed["checked"]
        assert grown_k.shape == grown_v.shape == (16, 2, 4)
        assert resources.retained_counts()["kv"] == 1


def test_retained_scan_banks_match_reference_and_reuse_storage():
    torch.manual_seed(810)
    frames, heads, dim = 7, 2, 8
    alpha = torch.rand(frames, heads, dim) * 0.4 + 0.55
    a_raw = torch.randn(frames, heads, dim, dim) * 0.03
    a_raw = 0.5 * (a_raw + a_raw.transpose(-1, -2))
    b_raw = torch.randn(frames, heads, dim, dim) * 0.2
    text = torch.randn(heads, dim, dim) * 0.1
    backend = B.VdnDelta(None)

    want_f, want_r = B.run_scans(
        backend, alpha, a_raw, b_raw, text_state=text)
    want_f, want_r = want_f.clone(), want_r.clone()

    owner = RuntimeBufferOwner(True)
    with owner.execution() as resources:
        got_f, got_r = run_scans_runtime(
            backend, alpha, a_raw, b_raw, text_state=text)
        got_f, got_r = got_f.clone(), got_r.clone()
        counts = resources.retained_counts()
        assert counts["scan"] == 1
        bank1 = resources.scan_banks(
            frames, got_f.shape[1:], got_f.dtype, got_f.device)[0]
        again_f, again_r = run_scans_runtime(
            backend, alpha, a_raw, b_raw, text_state=text)
        bank2 = resources.scan_banks(
            frames, again_f.shape[1:], again_f.dtype, again_f.device)[0]
        assert bank1.data_ptr() == bank2.data_ptr()
        assert torch.equal(again_f, got_f)
        assert torch.equal(again_r, got_r)

    assert torch.equal(got_f, want_f)
    assert torch.equal(got_r, want_r)


def test_transient_scan_path_matches_reference_without_retained_bank():
    torch.manual_seed(811)
    frames, heads, dim = 6, 2, 6
    alpha = torch.rand(frames, heads, dim) * 0.4 + 0.55
    a_raw = torch.randn(frames, heads, dim, dim) * 0.02
    a_raw = 0.5 * (a_raw + a_raw.transpose(-1, -2))
    b_raw = torch.randn(frames, heads, dim, dim) * 0.15
    backend = B.VdnDelta(None)

    want_f, want_r = B.run_scans(backend, alpha, a_raw, b_raw)
    owner = RuntimeBufferOwner(False)
    with owner.execution() as resources:
        got_f, got_r = run_scans_runtime(backend, alpha, a_raw, b_raw)
        assert resources.retained_counts()["scan"] == 0
    assert torch.equal(got_f, want_f)
    assert torch.equal(got_r, want_r)


def test_retained_grouped_window_matches_reference_and_reuses_plan_scratch():
    torch.manual_seed(812)
    frames, per_frame, heads, dim = 9, 3, 2, 8
    text, tail = 4, 2
    video_start = text
    video_end = text + frames * per_frame
    seq = video_end + tail
    q = torch.randn(seq, heads, dim)
    k = torch.randn(seq, heads, dim)
    v = torch.randn(seq, heads, dim)
    bounds = W.window_bounds(frames, 1, 3)
    scale = dim ** -0.5

    want = W.window_softmax_grouped(
        q, k, v, video_start, video_end, frames, per_frame,
        bounds, scale, anchor_frames="both")

    owner = RuntimeBufferOwner(True)
    with owner.execution() as resources:
        got = window_softmax_grouped_runtime(
            q, k, v, video_start, video_end, frames, per_frame,
            bounds, scale, anchor_frames="both")
        counts = resources.retained_counts()
        assert counts["plans"] == 1
        assert counts["kv"] == 1
        got2 = window_softmax_grouped_runtime(
            q, k, v, video_start, video_end, frames, per_frame,
            bounds, scale, anchor_frames="both")
        assert resources.retained_counts()["plans"] == 1
        assert resources.retained_counts()["kv"] == 1

    assert torch.equal(got, want)
    assert torch.equal(got2, want)


def _branch_weights(hidden, heads, dim):
    return {
        "beta_proj.weight": torch.randn(heads, hidden) * 0.1,
        "norm.weight": torch.randn(dim) * 0.1 + 1.0,
        "alpha.A_log": torch.randn(heads) * 0.1,
        "alpha.dt_bias": torch.randn(heads * dim) * 0.1,
        "alpha.down.weight": torch.randn(dim, hidden) * 0.1,
        "alpha.up.weight": torch.randn(heads * dim, dim) * 0.1,
        "output_gate.down.weight": torch.randn(dim, hidden) * 0.1,
        "output_gate.up.weight": torch.randn(heads * dim, dim) * 0.1,
        "output_gate.up.bias": torch.randn(heads * dim) * 0.1,
    }


def test_complete_runtime_linear_branch_matches_reference_branch():
    torch.manual_seed(813)
    frames, per_frame, heads, dim, hidden = 7, 3, 2, 4, 8
    rows = frames * per_frame
    text_rows = 5
    weights = _branch_weights(hidden, heads, dim)
    xv = torch.randn(rows, hidden)
    q = torch.randn(rows, heads, dim)
    k = torch.randn(rows, heads, dim)
    v = torch.randn(rows, heads, dim)
    text_x = torch.randn(text_rows, hidden)
    text_k = torch.randn(text_rows, heads, dim)
    text_v = torch.randn(text_rows, heads, dim)
    bounds = W.window_bounds(frames, 1, 3)

    reference = B.LinearBranch(
        weights, heads, dim, delta_rule="vdn_solve", bridge="alpha",
        a_fp32=True, short_conv=(), enable_text_state=True)
    runtime = RuntimeLinearBranch(
        weights, heads, dim, delta_rule="vdn_solve", bridge="alpha",
        a_fp32=True, short_conv=(), enable_text_state=True)

    want = reference.readout(
        weights, xv, q, k, v, frames, per_frame, bounds,
        frame_size=(1, per_frame), text_x=text_x,
        text_k_raw=text_k, text_v_raw=text_v, skip_ends=False)

    owner = RuntimeBufferOwner(True)
    with owner.execution() as resources:
        got = runtime.readout(
            weights, xv, q, k, v, frames, per_frame, bounds,
            frame_size=(1, per_frame), text_x=text_x,
            text_k_raw=text_k, text_v_raw=text_v, skip_ends=False)
        assert resources.retained_counts()["scan"] >= 1

    assert torch.equal(got, want)


def test_stream_prefetch_identity_includes_block_device_and_dtype():
    class Resources:
        retain = True

        def __init__(self):
            self.taken = []
            self.requested = []

        def prefetch_take(self, key):
            self.taken.append(key)
            return None

        def prefetch_request(self, key, fetch):
            self.requested.append(key)

    resources = Resources()

    class Runtime:
        @staticmethod
        def current():
            return resources

    state = VDNState(
        "test", {}, [SimpleNamespace(w={}), SimpleNamespace(w={})], 1, 1)
    state.runtime = Runtime()
    state._stream_weights = lambda index, device, dtype: {"block": index}

    got = state.weights_on(0, "cuda:0", torch.bfloat16)
    assert got == {"block": 0}
    assert resources.taken == [(0, "cuda:0", torch.bfloat16)]
    assert resources.requested == [(1, "cuda:0", torch.bfloat16)]


def test_completed_prefetch_is_retained_for_its_target(monkeypatch):
    from concurrent.futures import Future
    from vdn_h3 import runtime

    prefetcher = runtime._StreamPrefetcher()
    completed = Future()
    completed.set_result((0, "original", {}, object()))
    prefetcher._future = completed
    prefetcher._index = "original"

    def unexpected_submit(*args, **kwargs):
        raise AssertionError("unconsumed transfer must not be replaced")

    monkeypatch.setattr(runtime._PREFETCH_EXECUTOR, "submit", unexpected_submit)
    prefetcher.request("later", lambda: {})
    assert prefetcher.take("later") is None
    assert prefetcher._future is completed
    assert prefetcher._index == "original"
    prefetcher.reset()
    assert prefetcher._future is None


def test_release_retained_clears_primary_pool_after_execution():
    owner = RuntimeBufferOwner(True)
    with owner.execution() as resources:
        resources.activation_scratch(8, 3, 2, 4, "cpu", torch.float32)
        resources.kv_scratch(8, 2, 4, "cpu", torch.float32)
    before = owner.release_retained()
    assert before["activations"] == 1
    assert before["kv"] == 1
    with owner.execution() as resources:
        counts = resources.retained_counts()
        assert counts["activations"] == 0
        assert counts["kv"] == 0
