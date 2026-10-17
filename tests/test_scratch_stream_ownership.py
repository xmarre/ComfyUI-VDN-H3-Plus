"""Retained scratch must never migrate between asynchronous CUDA streams."""
from types import SimpleNamespace

import pytest
import torch

from vdn_h3 import runtime


def test_cuda_cache_key_uses_resolved_device_and_stream(monkeypatch):
    seen = []
    stream = SimpleNamespace(device=torch.device('cuda:1'), cuda_stream=137)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: seen.append(device) or stream)
    assert runtime._scratch_device_key('cuda') == ('cuda:1', 137)
    assert seen == [torch.device('cuda')]
    assert runtime._scratch_device_key('cpu') == 'cpu'


@pytest.mark.parametrize('kind', ['delta', 'scan', 'activations', 'kv', 'plan'])
def test_retained_scratch_reuses_same_stream_and_isolates_other_stream(monkeypatch, kind):
    # Simulate asynchronous stream identities with CPU allocations so this
    # ownership regression executes on CI as well as on a CUDA host.
    active = ['A']
    monkeypatch.setattr(runtime, '_scratch_device_key', lambda device: ('cpu', active[0]))
    owner = runtime.RuntimeBufferOwner(True)

    def borrow(pool):
        if kind == 'delta':
            return pool.delta_scratch((4, 2), 'cpu')
        if kind == 'scan':
            return pool.scan_banks(4, (2,), torch.float32, 'cpu')[0]
        if kind == 'activations':
            return pool.activation_scratch(4, 2, 1, 2, 'cpu', torch.float32)['q']
        if kind == 'kv':
            return pool.kv_scratch(4, 1, 2, 'cpu', torch.float32)[0]
        return pool.window_plan('layout', lambda: torch.empty(4, 2), device='cpu')

    with owner.execution() as pool:
        previous = borrow(pool)
        previous.fill_(7)
        assert borrow(pool).data_ptr() == previous.data_ptr()
    active[0] = 'B'
    with owner.execution() as pool:
        current = borrow(pool)
        assert current.data_ptr() != previous.data_ptr()
        current.fill_(11)
        assert torch.all(previous == 7)
        assert borrow(pool).data_ptr() == current.data_ptr()
    assert sum(owner._primary.retained_counts()[k] for k in ('scan', 'delta', 'activations', 'kv', 'plans')) <= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA stream scheduling requires a GPU')
def test_pending_cuda_scan_writes_do_not_race_later_execution():
    owner = runtime.RuntimeBufferOwner(True)
    stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(stream_a), owner.execution() as pool:
        old = pool.scan_banks(4, (128,), torch.float32, 'cuda')[0]
        torch.cuda._sleep(10_000_000)
        old.fill_(7)
    with torch.cuda.stream(stream_b), owner.execution() as pool:
        new = pool.scan_banks(4, (128,), torch.float32, 'cuda')[0]
        assert new.data_ptr() != old.data_ptr()
        new.fill_(11)
        snapshot = new.clone()
    stream_a.synchronize()
    stream_b.synchronize()
    assert torch.all(old == 7)
    assert torch.all(new == 11)
    assert torch.equal(new, snapshot)
