"""Bounded observation of actual short-conv inputs and intermediate outputs.

Two complete heads are copied, so offline activation/L2 comparisons have all
channels of each selected head. The production operator is never recomputed.
"""

from __future__ import annotations

import torch

WITNESS_KEY = "h3_flow_partitioned_boundary_witness_v1"
WITNESS_API = 1


class FeatureWitness:
    def __init__(self, sink, context):
        self.sink = sink
        self.context = dict(context)
        self.frames = ()
        self.destinations = ()
        self.heads = ()
        self.cross_taps = []

    def raw(self, name, tokens, spatial, temporal, sizes, offsets, *, l2norm):
        boundary = next((i for i in range(1, len(sizes)) if sizes[i] != sizes[i - 1]), None)
        if boundary is None:
            return
        radius = temporal.shape[-1] // 2
        # Observe the complete temporal support of the receiving boundary frames.
        # Wider checkpoints remain bounded rather than silently truncating support.
        if radius > 4:
            raise RuntimeError("boundary witness supports temporal radius at most four")
        self.destinations = tuple(range(max(0, boundary - radius), min(len(sizes), boundary + radius)))
        self.frames = tuple(range(max(0, boundary - 2 * radius), min(len(sizes), boundary + 2 * radius)))
        self.heads = tuple(sorted({0, tokens.shape[1] - 1}))
        self.context.update(
            frame_sizes=tuple(sizes), frame_offsets=tuple(offsets), boundary_inner_frame=boundary,
            selected_heads=self.heads, selected_frames=self.frames, destinations=self.destinations,
        )
        dim = tokens.shape[-1]
        for frame in self.frames:
            start, stop = offsets[frame]
            self.sink.add(f"{name}/raw/{frame}", tokens[start:stop, self.heads, :])
        channels = [h * dim + d for h in self.heads for d in range(dim)]
        self.sink.add(f"{name}/spatial_weight", spatial[channels])
        self.sink.add(f"{name}/temporal_weight", temporal[channels])
        self.context.setdefault("features", {})[name] = {"l2norm": l2norm, "head_dim": dim}

    def native(self, name, maps, owners):
        for frame in self.frames:
            run, local = owners[frame]
            self._volume(f"{name}/native/{frame}", maps[run][local : local + 1])

    def _volume(self, name, volume):
        dim = volume.shape[1] // (max(self.heads) + 1)
        channels = [h * dim + d for h in self.heads for d in range(dim)]
        self.sink.add(name, volume[:, channels])

    def tap(self, name, frame, source_frame, tap, mapped, weighted):
        if frame not in self.destinations:
            return
        self._volume(f"{name}/mapped/{frame}/{source_frame}/{tap}", mapped)
        self._volume(f"{name}/weighted/{frame}/{source_frame}/{tap}", weighted)
        self.cross_taps.append((name, frame, source_frame, tap))

    def mixed(self, name, runs, mixed_runs, outputs, offsets):
        for (start, stop, _grid), mixed in zip(runs, mixed_runs, strict=True):
            for frame in self.destinations:
                if start <= frame < stop:
                    self._volume(f"{name}/preactivation/{frame}", mixed[frame - start : frame - start + 1])
        # Outputs are run-local activated tensors; use the existing output, not a replay.
        for (start, stop, grid), output in zip(runs, outputs, strict=True):
            rows = grid[0] * grid[1]
            for frame in self.destinations:
                if start <= frame < stop:
                    local = (frame - start) * rows
                    self.sink.add(f"{name}/activated/{frame}", output[local : local + rows, self.heads, :])

    def statistics(self, a_raw, b_raw, alpha):
        for frame in self.destinations:
            # Full-head frame norms refer to the actual measured production statistics.
            for name, tensor in (("A", a_raw), ("B", b_raw), ("alpha", alpha)):
                self.sink.add(f"statistics/{name}/{frame}", tensor[frame].float().norm().reshape(1))

    def finish(self):
        self.context["observed_cross_taps"] = self.cross_taps
        self.sink.finish(self.context)


def claim_feature_witness(options, *, block_index, plan_digest, mode):
    sink = options.get(WITNESS_KEY)
    if sink is None:
        return None
    if getattr(sink, "api", None) != WITNESS_API:
        raise RuntimeError("partitioned boundary witness API is unsupported")
    context = {
        "stage": options.get("h3_flow_stage"),
        "request_id": options.get("h3_flow_request_id_v1"),
        "stage_id": options.get("h3_flow_stage_id_v1"),
        "evaluation_id": options.get("h3_flow_evaluation_id_v1"),
        "block_index": block_index,
        "plan_digest": plan_digest,
        "diagnostic_mode": mode,
        "policy": "native_grid_then_map_v1",
    }
    return FeatureWitness(sink, context) if sink.claim(context) else None
