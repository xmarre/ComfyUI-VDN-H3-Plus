"""VDN factors keep their math/error gate without a host read at every layer."""
import weakref

import pytest
import torch

from vdn_h3 import branch as B
from vdn_h3 import runtime as R


def _inputs(seed=72):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn((3, 2, 4, 4), generator=generator) * 0.1
    a = features @ features.transpose(-1, -2)
    b = torch.randn(a.shape, generator=generator)
    alpha = torch.rand((3, 2, 4), generator=generator)
    return alpha, a, b


def _exercise_deferral_on_host(monkeypatch):
    # Real LAPACK factors and status codes exercise ownership/error handling on
    # CPU. The production eligibility guard is tested separately; CUDA parity
    # below is deliberately skipped on a host without a GPU.
    monkeypatch.setattr(R, "_can_defer_cholesky", lambda _matrix: not torch.is_grad_enabled(), raising=False)


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("backend", [B.VdnDelta, B.VdnScaledDelta])
def test_50_layers_and_text_factors_share_one_status_read_and_preserve_outputs(monkeypatch, retained, backend):
    alpha, a, b = _inputs()
    factor = backend(11)
    with torch.no_grad():
        expected = factor.factor_apply(alpha, a, b)
    _exercise_deferral_on_host(monkeypatch)
    reads = []
    original_item = torch.Tensor.item

    def read_status(tensor, *_args, **_kwargs):
        reads.append(tensor.shape)
        return original_item(tensor)

    monkeypatch.setattr(torch.Tensor, "item", read_status)
    monkeypatch.setattr(torch.linalg, "cholesky", lambda *_a, **_k: pytest.fail("Per-layer synchronizing Cholesky"))
    owner = R.RuntimeBufferOwner(retained)
    # A second differently conditioned execution must not consume prior statuses.
    for seed in (72, 73):
        alpha, a, b = _inputs(seed)
        with torch.no_grad(), owner.execution() as buffers:
            for _layer in range(50):
                video = factor.factor_apply(alpha, a, b)
                text = factor.factor_apply(alpha[:1], a[:1], b[:1])
            assert len(reads) == (0 if seed == 72 else 1)
            assert len(buffers._cholesky_statuses) == 100
            references = [weakref.ref(info) for info in buffers._cholesky_statuses]
        assert all(reference() is None for reference in references)
        assert buffers._cholesky_status_bytes == 0
        assert buffers._cholesky_calls == buffers._cholesky_status_reads == 0
        if seed == 72:
            for got, want in zip(video, expected, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            for got, want in zip(text, expected, strict=True):
                torch.testing.assert_close(got, want[:1], rtol=0, atol=0)
    assert reads == [torch.Size([]), torch.Size([])]


@pytest.mark.parametrize("retained", [False, True])
def test_failed_factorization_never_returns_prediction_and_next_execution_is_clean(monkeypatch, retained):
    _exercise_deferral_on_host(monkeypatch)
    owner = R.RuntimeBufferOwner(retained)
    returned = []

    def predict():
        with torch.no_grad(), owner.execution() as buffers:
            alpha, a, b = _inputs()
            a[1, 0] = -2 * torch.eye(4)
            result = B.VdnDelta().factor_apply(alpha, a, b)
            assert len(buffers._cholesky_statuses) == 1
            return result

    with pytest.raises(torch.linalg.LinAlgError, match="prediction.*rejected"):
        returned.append(predict())
    assert returned == []
    assert R.current_runtime_buffers() is None
    with torch.no_grad(), owner.execution() as buffers:
        assert not buffers._cholesky_statuses
        B.VdnDelta().factor_apply(*_inputs())


def test_cancellation_preserves_original_error_and_drops_status_tensors(monkeypatch):
    _exercise_deferral_on_host(monkeypatch)
    owner = R.RuntimeBufferOwner(True)
    with pytest.raises(RuntimeError, match="original interruption"):
        with torch.no_grad(), owner.execution() as buffers:
            R.checked_cholesky(torch.eye(4))
            reference = weakref.ref(buffers._cholesky_statuses[0])
            raise RuntimeError("original interruption")
    assert reference() is None
    assert not buffers._cholesky_statuses
    assert R.current_runtime_buffers() is None


def test_nested_execution_checks_and_releases_only_its_own_statuses(monkeypatch):
    _exercise_deferral_on_host(monkeypatch)
    owner = R.RuntimeBufferOwner(True)
    with torch.no_grad(), owner.execution() as outer:
        R.checked_cholesky(torch.eye(4))
        outer_ref = weakref.ref(outer._cholesky_statuses[0])
        with pytest.raises(torch.linalg.LinAlgError):
            with owner.execution() as inner:
                assert inner is not outer
                R.checked_cholesky(-torch.eye(4))
                inner_ref = weakref.ref(inner._cholesky_statuses[0])
        assert inner_ref() is None
        assert outer_ref() is not None
        assert R.current_runtime_buffers() is outer
    assert outer_ref() is None


@pytest.mark.parametrize("limit", ["calls", "bytes"])
def test_pending_status_storage_is_bounded_and_all_batches_are_checked(monkeypatch, limit):
    _exercise_deferral_on_host(monkeypatch)
    monkeypatch.setattr(R, "_MAX_CHOLESKY_STATUSES", 2 if limit == "calls" else 128)
    monkeypatch.setattr(R, "_MAX_CHOLESKY_STATUS_BYTES", 8 if limit == "bytes" else 1024)
    with torch.no_grad(), R.RuntimeBufferOwner(True).execution() as buffers:
        for _ in range(7):
            R.checked_cholesky(torch.eye(4))
            assert len(buffers._cholesky_statuses) <= 2
            assert buffers._cholesky_status_bytes <= 8
        assert buffers._cholesky_calls == 7
        assert buffers._cholesky_status_reads == 3


def test_cpu_and_autograd_keep_immediate_check_and_gradient(monkeypatch):
    calls = []
    original = torch.linalg.cholesky

    def immediate(matrix):
        calls.append(matrix.device.type)
        return original(matrix)

    monkeypatch.setattr(torch.linalg, "cholesky", immediate)
    owner = R.RuntimeBufferOwner(True)
    matrix = (2 * torch.eye(4)).requires_grad_()
    with owner.execution() as buffers:
        R.checked_cholesky(matrix).sum().backward()
        assert not buffers._cholesky_statuses
    assert matrix.grad is not None and torch.isfinite(matrix.grad).all()
    with torch.no_grad(), owner.execution(), pytest.raises(torch.linalg.LinAlgError):
        R.checked_cholesky(-torch.eye(4))
    assert calls == ["cpu", "cpu"]


@pytest.mark.parametrize("guard", ["autograd", "compile", "capture"])
def test_cuda_guard_preserves_training_compile_and_capture_fallback(monkeypatch, guard):
    class CudaMatrix:
        device = torch.device("cuda")

    monkeypatch.setattr(torch, "is_grad_enabled", lambda: guard == "autograd")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: guard == "compile")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: guard == "capture")
    assert not R._can_defer_cholesky(CudaMatrix())


def test_compiled_factorization_does_not_read_runtime_context():
    alpha, a, b = _inputs()
    factor = B.VdnDelta()
    compiled = torch.compile(factor.factor_apply, backend="eager", fullgraph=True)
    with torch.no_grad():
        expected = factor.factor_apply(alpha, a, b)
        actual = compiled(alpha, a, b)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_cuda_compile_guard_runs_before_reading_runtime_context(monkeypatch):
    class CudaMatrix:
        device = torch.device("cuda")

    sentinel = object()
    monkeypatch.setattr(torch, "is_grad_enabled", lambda: False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(R, "current_runtime_buffers", lambda: pytest.fail("Compiled call read a ContextVar"))
    monkeypatch.setattr(torch.linalg, "cholesky", lambda _matrix: sentinel)
    assert R.checked_cholesky(CudaMatrix()) is sentinel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA factorization parity requires a GPU")
def test_cuda_factorization_preserves_native_result_and_error_gate():
    alpha, a, b = [tensor.cuda() for tensor in _inputs()]
    backend = B.VdnDelta()
    with torch.no_grad():
        expected = backend.factor_apply(alpha, a, b)
        with R.RuntimeBufferOwner(True).execution() as buffers:
            actual = backend.factor_apply(alpha, a, b)
            assert len(buffers._cholesky_statuses) == 1
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        with pytest.raises(torch.linalg.LinAlgError):
            with R.RuntimeBufferOwner(True).execution():
                backend.factor_apply(alpha, -2 * torch.eye(4, device="cuda").expand_as(a), b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA status stream ownership requires a GPU")
def test_cuda_status_check_uses_its_producer_stream_and_handles_stream_switch():
    matrix = torch.eye(4, device="cuda")
    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    first.wait_stream(torch.cuda.current_stream())
    second.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), R.RuntimeBufferOwner(True).execution() as buffers:
        with torch.cuda.stream(first):
            R.checked_cholesky(matrix)
            assert buffers._cholesky_status_reads == 0
        with torch.cuda.stream(second):
            R.checked_cholesky(matrix)
            assert buffers._cholesky_status_reads == 1
        # The execution exits on the original stream; the second batch must
        # still be read on its producer, and the caller's stream restored.
        assert buffers._cholesky_status_stream == second
