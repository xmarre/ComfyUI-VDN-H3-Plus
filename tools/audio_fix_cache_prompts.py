#!/usr/bin/env python3
"""Cache MiniMax-H3 prompt conditioning with the text encoder already in ComfyUI.

No Hugging Face download is performed. ``--text-encoder`` may be an absolute path or a
filename already present under ``ComfyUI/models/text_encoders``. The cached tensors are
all the production INT8 audio-fix trainer needs from Qwen3-VL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _bootstrap():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--comfy-root", default=os.environ.get("COMFYUI_ROOT", "/home/toor/ComfyUI"))
    known, _ = parser.parse_known_args()
    root = os.path.abspath(os.path.expanduser(known.comfy_root))
    if not os.path.isfile(os.path.join(root, "comfy", "sd.py")):
        raise SystemExit(f"Not a ComfyUI checkout: {root}")
    sys.path.insert(0, root)
    return root


COMFY_ROOT = _bootstrap()

import torch  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.sd  # noqa: E402


def _resolve_encoder(path):
    path = os.path.expanduser(path)
    if os.path.isabs(path) and os.path.isfile(path):
        return os.path.realpath(path)
    candidate = os.path.join(COMFY_ROOT, "models", "text_encoders", path)
    if os.path.isfile(candidate):
        return os.path.realpath(candidate)
    raise FileNotFoundError(
        f"Text encoder {path!r} not found as an absolute file or under "
        f"{os.path.join(COMFY_ROOT, 'models', 'text_encoders')}")


def _read_prompts(path):
    prompts = []
    with open(path, encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            if path.lower().endswith(".jsonl"):
                row = json.loads(raw)
                prompt = row.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"{path}:{line_number}: missing non-empty 'prompt'")
                prompts.append(prompt)
            else:
                prompts.append(raw)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", default=COMFY_ROOT)
    parser.add_argument("--text-encoder", required=True,
                        help="Existing MiniMax-H3 text encoder safetensors")
    parser.add_argument("--prompts", required=True, help="JSONL with prompt field, or one prompt per line")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--take", type=int, default=0, help="0 = all prompts")
    parser.add_argument("--prefix", default="sample")
    args = parser.parse_args()

    encoder_path = _resolve_encoder(args.text_encoder)
    prompts = _read_prompts(os.path.abspath(args.prompts))
    if args.take > 0:
        prompts = prompts[:args.take]
    output = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output, exist_ok=True)

    print(f"loading existing MiniMax-H3 conditioner: {encoder_path}", flush=True)
    clip = comfy.sd.load_clip(
        ckpt_paths=[encoder_path],
        embedding_directory=None,
        clip_type=comfy.sd.CLIPType.MINIMAX,
    )

    for index, prompt in enumerate(prompts):
        tokens = clip.tokenize(prompt)
        encoded = clip.encode_from_tokens(tokens, return_dict=True)
        context = encoded.get("cond")
        tags = encoded.get("minimax_token_tags")
        if not isinstance(context, torch.Tensor) or context.ndim != 3 or context.shape[0] != 1:
            raise RuntimeError(
                f"MiniMax-H3 encoder returned invalid context for prompt {index}: "
                f"{None if context is None else tuple(context.shape)}")
        if not isinstance(tags, torch.Tensor) or tags.numel() != context.shape[1]:
            raise RuntimeError(
                f"MiniMax-H3 encoder did not return matching minimax_token_tags for prompt {index}")
        payload = {
            "format": 1,
            "prompt": prompt,
            "context": context.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
            "text_token_tags": tags.detach().to(device="cpu", dtype=torch.long).reshape(-1).contiguous(),
            "source_text_encoder": encoder_path,
        }
        path = os.path.join(output, f"{args.prefix}_{index:05d}.pt")
        torch.save(payload, path)
        print(f"[{index + 1}/{len(prompts)}] {path}", flush=True)

    del clip
    comfy.model_management.unload_all_models()
    print(f"cached {len(prompts)} prompts under {output}", flush=True)


if __name__ == "__main__":
    main()
