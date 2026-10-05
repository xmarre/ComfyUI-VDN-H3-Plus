"""Compare both stencil orderings offline on captured actual projected features.

This CPU component replay neither invokes H3 nor qualifies rendered output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def physical_map(source, target_hw):
    sh, sw = source.shape[-2:]
    th, tw = target_hw
    if (sh, sw) == (th, tw):
        return source
    sa, ta = math.sqrt(sh * sw), math.sqrt(th * tw)

    def axis(sl, tl):
        if sl == 1:
            return torch.zeros(tl)
        positions = 16 * (1 - tl / ta) + 32 * torch.arange(tl) / ta
        indices = (positions - 16 * (1 - sl / sa)) / (32 / sa)
        return 2 * indices / (sl - 1) - 1

    yy, xx = torch.meshgrid(axis(sh, th), axis(sw, tw), indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)[None]
    return F.grid_sample(source.float(), grid, padding_mode="border", align_corners=True).to(source.dtype)


def compare(tensors, context):
    sizes = context["frame_sizes"]
    results = []
    for name, spec in context["features"].items():
        spatial = tensors[f"{name}/spatial_weight"]
        temporal = tensors[f"{name}/temporal_weight"].squeeze(1)
        channels = spatial.shape[0]
        pad = temporal.shape[-1] // 2
        for frame in context["destinations"]:
            old = new = None
            for tap in range(temporal.shape[-1]):
                neighbor = frame + tap - pad
                if not 0 <= neighbor < len(sizes):
                    continue
                if context["diagnostic_mode"] == "suppress_cross_grid_temporal_taps" and sizes[neighbor] != sizes[frame]:
                    continue
                raw = tensors[f"{name}/raw/{neighbor}"]
                raw = raw.reshape(*sizes[neighbor], channels).permute(2, 0, 1)[None]
                native = F.conv2d(raw, spatial, padding=2, groups=channels)
                legacy = physical_map(native, sizes[frame])
                destination = F.conv2d(physical_map(raw, sizes[frame]), spatial, padding=2, groups=channels)
                weight = temporal[:, tap].to(raw.dtype).view(1, -1, 1, 1)
                op, np = legacy * weight, destination * weight
                old = op if old is None else old + op
                new = np if new is None else new + np
            delta = (old - new).float()
            observed = tensors[f"{name}/preactivation/{frame}"]
            # CPU convolution can differ from CUDA's low-precision accumulation.
            # Report that replay discrepancy separately from ordering differences.
            regional = []
            for row in torch.tensor_split(delta, 4, dim=-2):
                for roi in torch.tensor_split(row, 4, dim=-1):
                    regional.append(float(roi.square().mean().sqrt()) if roi.numel() else None)
            shape = (sizes[frame][0] * sizes[frame][1], len(context["selected_heads"]), spec["head_dim"])
            activated = F.silu(new[0].permute(1, 2, 0).reshape(shape))
            if spec["l2norm"]:
                activated = F.normalize(activated, dim=-1)
            results.append({
                "feature": name, "inner_frame": frame,
                "physical_frame": frame + context.get("frame_index_origin", 0),
                "ordering_max_abs": float(delta.abs().max()),
                "ordering_rms": float(delta.square().mean().sqrt()),
                "ordering_roi4x4_rms": regional,
                "cpu_legacy_vs_observed_max_abs": float((old - observed).float().abs().max()),
                "destination_activated_vs_observed_rms": float(
                    (activated - tensors[f"{name}/activated/{frame}"]).float().square().mean().sqrt()
                ),
                "finite": bool(torch.isfinite(delta).all()),
            })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    source = args.receipt.parent / receipt["tensor_file"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != receipt["tensor_sha256"]:
        raise RuntimeError("boundary witness tensor file hash does not match its receipt")
    tensors = torch.load(source, map_location="cpu", weights_only=True)
    result = {
        "receipt_sha256": hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
        "diagnostic_only": True, "rendered_causality_established": False,
        "extra_cpu_component_replay": True, "extra_h3_nfe": 0,
        "results": compare(tensors, receipt["context"]),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
