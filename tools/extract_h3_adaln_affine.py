#!/usr/bin/env python3
"""Extract the tiny AdaLN pruning affine needed for released full-width H3 LoRAs.

This tool is only for source checkpoints that actually retain ``adaln_basis`` and
``adaln_mean``. The standard Comfy-Org MiniMax-H3 ``*_pruned_*`` single-file
checkpoints contain the collapsed curve table but intentionally omit those two
auxiliary tensors, so they are not valid extraction sources. For those public
checkpoints, install the matching published ~97 KB ``adaln_affine.safetensors``
sidecar instead; see the repository README.

Quantized derivatives may also omit the auxiliaries. This tool writes only the two
small tensors to ``adaln_affine.safetensors`` and records the matching curve-table
SHA-256 when the source checkpoint also contains ``adaln_t_table`` or
``time_embedder.table``.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def tensor_hash(t: torch.Tensor) -> str:
    x = t.detach().to(device="cpu", dtype=torch.float32).contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(x.shape)).encode())
    h.update(x.numpy().tobytes())
    return h.hexdigest()


def missing_affine_message(source: Path, missing: list[str]) -> str:
    name = source.name.lower()
    if "ref2va" in name:
        sidecar = "transformer_ref/adaln_affine.safetensors"
    elif "fl2va" in name:
        sidecar = "transformer/adaln_affine.safetensors"
    else:
        sidecar = "transformer/adaln_affine.safetensors (T2VA/FL2VA) or transformer_ref/adaln_affine.safetensors (Ref2VA)"

    return (
        f"{source}: missing {missing}\n\n"
        "This file cannot be used as an AdaLN-affine extraction source. "
        "The standard Comfy-Org MiniMax-H3 *_pruned_* checkpoints contain the "
        "collapsed curve table but do not contain adaln_basis/adaln_mean.\n\n"
        "For those public pruned checkpoints, download the matching ~97 KB sidecar "
        "from multimodalart/MiniMax-H3-Pruned instead:\n"
        f"  {sidecar}\n"
        "and place it as <ComfyUI>/models/vdn/<stage>/adaln_affine.safetensors.\n\n"
        "Use this extractor only with a matching source checkpoint that actually "
        "retained adaln_basis and adaln_mean."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract adaln_basis/adaln_mean from a source checkpoint that retained "
            "the pruning auxiliaries. Standard Comfy-Org *_pruned_* checkpoints are "
            "not extraction sources; use the published affine sidecar for them."
        )
    )
    parser.add_argument("source", type=Path, help="matching source safetensors containing adaln_basis/adaln_mean")
    parser.add_argument(
        "output", type=Path, nargs="?", default=Path("adaln_affine.safetensors"))
    args = parser.parse_args()

    metadata = {}
    with safe_open(str(args.source), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        missing = [key for key in ("adaln_basis", "adaln_mean") if key not in keys]
        if missing:
            raise SystemExit(missing_affine_message(args.source, missing))
        basis = handle.get_tensor("adaln_basis").to(torch.float32).clone().contiguous()
        mean = handle.get_tensor("adaln_mean").to(torch.float32).clone().contiguous()
        table = None
        for key in ("adaln_t_table", "time_embedder.table"):
            if key in keys:
                table = handle.get_tensor(key).to(torch.float32).clone().contiguous()
                break

    if basis.ndim != 2 or mean.ndim != 1 or basis.shape[1] != mean.shape[0]:
        raise SystemExit(
            f"invalid affine shapes: basis={tuple(basis.shape)} mean={tuple(mean.shape)}")
    if table is not None:
        if table.ndim != 2 or table.shape[1] != basis.shape[0]:
            raise SystemExit(
                f"curve/affine mismatch: table={tuple(table.shape)} basis={tuple(basis.shape)}")
        metadata["adaln_table_sha256"] = tensor_hash(table)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"adaln_basis": basis, "adaln_mean": mean}, str(args.output), metadata=metadata)
    size = args.output.stat().st_size
    print(f"wrote {args.output} ({size} bytes)")
    if "adaln_table_sha256" in metadata:
        print(f"adaln_table_sha256={metadata['adaln_table_sha256']}")
    else:
        print("warning: source had no curve table; output has no automatic table identity")


if __name__ == "__main__":
    main()
