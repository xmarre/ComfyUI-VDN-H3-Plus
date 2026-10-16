"""Bitwise parity of the single-pass temporal short-conv with the eager arithmetic."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from vdn_h3 import branch as B
from vdn_h3 import temporal_shift as T

_ROOT = Path(__file__).resolve().parents[1]

_INTERPRETER_SCRIPT = r"""
import json, os, sys
os.environ["TRITON_INTERPRET"] = "1"
import torch
try:
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
except ImportError:
    pass
from vdn_h3.temporal_shift import eager_temporal_shift, fused_temporal_shift

def bits(t):
    return t.view(torch.int16)

results = []
generator = torch.Generator().manual_seed(1234)
for frames, rows, channels in [(1, 3, 7), (2, 5, 130), (3, 17, 128), (4, 9, 129), (6, 33, 257), (9, 16, 64)]:
    x = (torch.randn(frames, rows, channels, generator=generator) * 3).to(torch.bfloat16)
    w = (torch.randn(channels, 5, generator=generator) * 0.3).to(torch.bfloat16)
    specials = torch.tensor([0.0, -0.0, 3.0e38, -3.0e38, float("inf"), 1.5, -2.25], dtype=torch.bfloat16)
    count = min(channels, specials.numel())
    x[frames // 2, 0, :count] = specials[:count]
    fused = fused_temporal_shift(x, w)
    eager = eager_temporal_shift(x, w, 5)
    nan_match = bool(torch.equal(torch.isnan(fused), torch.isnan(eager)))
    finite = ~torch.isnan(eager)
    results.append({
        "shape": [frames, rows, channels],
        "bitwise": bool(torch.equal(bits(fused)[finite], bits(eager)[finite])),
        "nan_match": nan_match,
        "contiguous": fused.is_contiguous(),
    })
print(json.dumps(results))
"""


def test_fused_kernel_matches_eager_bits_in_triton_interpreter():
    pytest.importorskip("triton")
    environment = dict(os.environ)
    paths = [str(_ROOT)]
    comfy_root = os.environ.get("COMFYUI_ROOT")
    if comfy_root:
        paths.append(comfy_root)
    environment["PYTHONPATH"] = os.pathsep.join(paths + [environment.get("PYTHONPATH", "")])
    completed = subprocess.run(
        [sys.executable, "-c", _INTERPRETER_SCRIPT], cwd=str(_ROOT), env=environment,
        capture_output=True, text=True, timeout=300,
    )
    if completed.returncode != 0 and "No module named 'comfy'" in completed.stderr:
        pytest.skip("ComfyUI is not importable for the package import")
    assert completed.returncode == 0, completed.stderr[-4000:]
    results = json.loads(completed.stdout.strip().splitlines()[-1])
    assert len(results) == 6
    for result in results:
        assert result == {"shape": result["shape"], "bitwise": True, "nan_match": True, "contiguous": True}


def test_cpu_and_unsupported_inputs_use_the_eager_arithmetic(monkeypatch):
    calls = []
    monkeypatch.setattr(T, "fused_temporal_shift", lambda *args: calls.append(args))
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(4, 6, 8, generator=generator).to(torch.bfloat16)
    w = torch.randn(8, 5, generator=generator).to(torch.bfloat16)
    assert torch.equal(T.temporal_shift(x, w, 5), T.eager_temporal_shift(x, w, 5))
    assert torch.equal(T.temporal_shift(x.float(), w.float(), 5), T.eager_temporal_shift(x.float(), w.float(), 5))
    assert calls == []


def test_conv_features_keeps_released_arithmetic_on_cpu():
    generator = torch.Generator().manual_seed(7)
    frames, grid, heads, head_dim = 5, (3, 4), 2, 4
    channels = heads * head_dim
    tokens = torch.randn(frames * 12, heads, head_dim, generator=generator).to(torch.bfloat16)
    sp = (torch.randn(channels, 1, 5, 5, generator=generator) * 0.1).to(torch.bfloat16)
    tm = (torch.randn(channels, 1, 5, generator=generator) * 0.2).to(torch.bfloat16)
    volume = tokens.reshape(frames, *grid, channels).permute(0, 3, 1, 2)
    volume = torch.nn.functional.conv2d(volume, sp, padding=2, groups=channels)
    x = volume.permute(0, 2, 3, 1).reshape(frames, 12, channels)
    expected = B._activate(T.eager_temporal_shift(x, tm.squeeze(1), 5).reshape(-1, heads, head_dim), True)
    actual = B.conv_features(tokens, sp, tm, frames, grid, l2norm=True)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_dispatch_is_bitwise_identical():
    pytest.importorskip("triton")
    generator = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn(13, 391, 7168, device="cuda", generator=generator).to(torch.bfloat16)
    w = (torch.randn(7168, 5, device="cuda", generator=generator) * 0.2).to(torch.bfloat16)
    fused = T.temporal_shift(x, w, 5)
    assert T.fused_temporal_status() == {"verified": True, "disabled_reason": None}
    assert torch.equal(fused.view(torch.int16), T.eager_temporal_shift(x, w, 5).view(torch.int16))


def test_component_benchmark_smoke_runs_on_cpu(tmp_path):
    pytest.importorskip("triton")
    environment = dict(os.environ, TRITON_INTERPRET="1")
    comfy_root = os.environ.get("COMFYUI_ROOT")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(_ROOT)] + ([comfy_root] if comfy_root else []) + [environment.get("PYTHONPATH", "")]
    )
    output = tmp_path / "components.json"
    script = (
        "import sys, runpy\n"
        "try:\n"
        "    import comfy.cli_args\n"
        "    comfy.cli_args.args.cpu = True\n"
        "except ImportError:\n"
        "    pass\n"
        "sys.argv = ['benchmark', '--device', 'cpu', '--frames', '4', '--grid', '3', '5', '--heads', '2',\n"
        "            '--head-dim', '8', '--rank', '4', '--rounds', '1', '--warmup', '0', '--json', sys.argv[1]]\n"
        "runpy.run_path('tools/benchmark_vdn_linear_components.py', run_name='__main__')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(output)], cwd=str(_ROOT), env=environment,
        capture_output=True, text=True, timeout=300,
    )
    if completed.returncode != 0 and "No module named 'comfy'" in completed.stderr:
        pytest.skip("ComfyUI is not importable for the package import")
    assert completed.returncode == 0, completed.stderr[-4000:]
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["contract"] == "vdn-linear-components-benchmark-v1"
    assert report["temporal"]["bitwise_equal"] and report["readout"]["outputs_bitwise_equal"]
    assert "k_temporal_fused" in report["components"]
