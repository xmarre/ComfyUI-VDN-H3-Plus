"""CUDA component audit must remain opt-in and not change sampler behavior."""
import logging

from vdn_h3 import perf


def test_disabled_profiling_leaves_no_scope(monkeypatch):
    monkeypatch.setattr(perf, "ENABLED", False)
    assert perf.enter() is None
    assert perf.forward("high", 0, "cuda:0") is None


def test_event_profile_tracks_exact_stage_components_and_clears(monkeypatch, caplog):
    timeline = []

    class Event:
        def __init__(self, enable_timing=False):
            assert enable_timing
            self.index = len(timeline)

        def record(self):
            timeline.append(self.index)

        def elapsed_time(self, other):
            return float(other.index - self.index)

    sync = []
    monkeypatch.setattr(perf, "ENABLED", True)
    monkeypatch.setattr(perf.torch.cuda, "Event", Event)
    monkeypatch.setattr(perf.torch.cuda, "synchronize", lambda device: sync.append(str(device)))
    with caplog.at_level(logging.WARNING, logger="comfy.vdn"):
        entered = perf.enter()
        assert entered is not None
        assert perf.enter() is None   # Flow stage nesting reuses existing scope.
        sample, _ = entered
        for stage in ("low", "high"):
            forward = perf.forward(stage, 24, "cuda:0")
            for phase in ("qkv_projection", "linear_readout", "qk_norm_rope",
                          "grouped_softmax", "output_projection", None):
                forward.mark(phase)
        perf.exit_scope(entered)
    assert sync == ["cuda:0"]
    assert sample.spans == []
    assert sample.counts == {}
    assert perf.forward("high", 24, "cuda:0") is None
    assert sum("CUDA component audit stage=" in m.message for m in caplog.records) == 2
    assert any("stage=high" in m.message and "forwards=1" in m.message
               and "component_cuda_total_ms=5.000" in m.message for m in caplog.records)


def test_profiler_failure_does_not_mask_sampler_exception(monkeypatch, caplog):
    monkeypatch.setattr(perf, "ENABLED", True)

    def fail(_sample):
        raise RuntimeError("simulated profiler failure")

    monkeypatch.setattr(perf.Sample, "report", fail)
    scope = perf.enter()
    assert scope is not None
    with caplog.at_level(logging.ERROR, logger="comfy.vdn"):
        perf.exit_scope(scope)
    assert "CUDA component audit failed" in caplog.text
    assert perf.forward("high", 24, "cuda:0") is None
