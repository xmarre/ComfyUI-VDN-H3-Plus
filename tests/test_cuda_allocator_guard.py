from __future__ import annotations

import pytest
import torch

from vdn_h3 import runtime


class _Recorded:
    def __init__(self):
        self.streams = []

    def record_stream(self, stream):
        self.streams.append(stream)


def test_prefetch_records_consumer_stream_without_allocator_special_case():
    stream = object()
    tensor = _Recorded()

    runtime._StreamPrefetcher._record_stream(tensor, stream)

    assert tensor.streams == [stream]


def test_prefetch_records_consumer_stream_even_when_cuda_malloc_async(monkeypatch):
    """cudaMallocAsync still tracks non-creation usage streams in PyTorch."""
    monkeypatch.setattr(
        runtime.torch.cuda,
        "get_allocator_backend",
        lambda: "cudaMallocAsync",
        raising=False,
    )
    stream = object()
    tensor = _Recorded()

    runtime._StreamPrefetcher._record_stream(tensor, stream)

    assert tensor.streams == [stream]


def test_prefetch_records_quantized_backing_storages(monkeypatch):
    stream = object()
    calls = []

    def record_tensor(self, got_stream):
        calls.append((id(self), got_stream))

    monkeypatch.setattr(torch.Tensor, "record_stream", record_tensor, raising=True)

    class Params:
        pass

    wrapper = _Recorded()
    wrapper._qdata = torch.empty(1)
    wrapper._params = Params()
    wrapper._params.scale = torch.empty(1)
    wrapper._params.orig_weight = torch.empty(1)
    wrapper._params.bias = torch.empty(1)

    expected_children = {
        id(wrapper._qdata),
        id(wrapper._params.scale),
        id(wrapper._params.orig_weight),
        id(wrapper._params.bias),
    }

    runtime._StreamPrefetcher._record_stream(wrapper, stream)

    # The wrapper is not the allocation consumed by the quantized kernel. Record its
    # backing storages directly, matching PyTorch's lifetime contract.
    assert wrapper.streams == []
    assert {tensor_id for tensor_id, got_stream in calls if got_stream is stream} == expected_children


def test_prefetch_record_failure_is_not_hidden():
    class Broken:
        def record_stream(self, stream):
            raise RuntimeError("stream registration failed")

    with pytest.raises(RuntimeError, match="stream registration failed"):
        runtime._StreamPrefetcher._record_stream(Broken(), object())
