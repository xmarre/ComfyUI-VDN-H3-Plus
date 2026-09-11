import inspect

from vdn_h3.direct_gpu_load import _must_stay_cpu, load_diffusion_model_direct_gpu


def test_quantization_metadata_stays_on_cpu():
    assert _must_stay_cpu("model.layers.0.self_attn.q_proj.comfy_quant")
    assert _must_stay_cpu("blocks.0.mlp.fc1.comfy_quant")


def test_tokenizer_payloads_stay_on_cpu():
    for key in (
        "tokenizer_json",
        "spiece_model",
        "tekken_model",
        "gemma_spiece_model",
        "nested.tokenizer_json",
    ):
        assert _must_stay_cpu(key)


def test_model_weights_and_quant_scales_remain_gpu_candidates():
    for key in (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight_scale",
        "model.layers.0.mlp.down_proj.weight",
        "visual.blocks.0.attn.qkv.weight",
    ):
        assert not _must_stay_cpu(key)


def test_diffusion_loader_can_replace_comfy_public_loader_call_shape():
    signature = inspect.signature(load_diffusion_model_direct_gpu)
    assert "path" in signature.parameters
    assert "model_options" in signature.parameters
