from pathlib import Path
from types import SimpleNamespace
import importlib.util


def _load_entrypoint():
    path = Path(__file__).resolve().parents[1] / "tools" / "audio_fix_int8_train_gpu.py"
    spec = importlib.util.spec_from_file_location("audio_fix_int8_train_gpu_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_branch_policy_prefers_int8_stream(tmp_path):
    module = _load_entrypoint()
    plain = tmp_path / "model.safetensors"
    quant = tmp_path / "model_int8_convrot_comfyui.safetensors"
    plain.write_bytes(b"plain")
    quant.write_bytes(b"quant")
    policy = SimpleNamespace(branch_candidates=lambda _path: (str(plain), str(quant)))

    mode, prefer_int8 = module._training_branch_policy(policy, str(tmp_path), 96 << 30)

    assert mode == "stream"
    assert prefer_int8 is True


def test_training_branch_policy_never_promotes_bf16_to_resident(tmp_path):
    module = _load_entrypoint()
    plain = tmp_path / "model.safetensors"
    quant = tmp_path / "missing_int8.safetensors"
    plain.write_bytes(b"plain")
    policy = SimpleNamespace(branch_candidates=lambda _path: (str(plain), str(quant)))

    mode, prefer_int8 = module._training_branch_policy(policy, str(tmp_path), 96 << 30)

    assert mode == "stream"
    assert prefer_int8 is False
