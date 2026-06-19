"""Round-trip tests for GPTQ + LoRA safetensors save/load.

The tests use a tiny CPU-only ``LlamaForCausalLM`` assembled from config so
they do not need CUDA, network access, or downloaded weights.  The real
``Runner.save_quantized_model`` and ``QuantizedModelLoader.load_quantized_model``
paths are exercised end-to-end.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

from logging import getLogger
from types import SimpleNamespace
from unittest.mock import patch

import torch

from onecomp.post_process.post_process_lora_sft import LoRAGPTQLinear, PostProcessLoraSFT
from onecomp.quantized_model_loader import QuantizedModelLoader
from onecomp.quantizer.gptq.gptq_layer import GPTQLinear
from onecomp.runner import Runner

QUANTIZED_LAYER_NAME = "model.layers.0.mlp.down_proj"


class _FakeTokenizer:
    """Small tokenizer stub with just enough API for save/load tests."""

    def save_pretrained(self, save_directory):
        from pathlib import Path

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        tokenizer_config = save_path / "tokenizer_config.json"
        tokenizer_config.write_text("{}\n", encoding="utf-8")
        return (str(tokenizer_config),)


class _TinyTokenizer(_FakeTokenizer):
    """Tokenizer stub used by PostProcessLoraSFT on the tiny local dataset."""

    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self,
        texts,
        max_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
    ):
        del return_attention_mask
        input_ids = []
        attention_mask = []
        for text in texts:
            ids = [1] + [3 + (ord(ch) % 61) for ch in str(text)] + [self.eos_token_id]
            if truncation:
                ids = ids[:max_length]
            mask = [1] * len(ids)
            if padding == "max_length":
                pad_len = max_length - len(ids)
                ids = ids + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len
            input_ids.append(ids)
            attention_mask.append(mask)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _build_tiny_llama():
    """Build a tiny Llama model whose MLP down projection can be GPTQ-wrapped."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        vocab_size=64,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config.torch_dtype = torch.float32
    model = LlamaForCausalLM(config).eval()
    return model


def _make_gptq_linear(in_features, out_features, *, packed=True):
    """Build a deterministic GPTQLinear compatible with the tiny Llama layer."""
    wbits = 4
    groupsize = 32
    num_groups = in_features // groupsize
    qweight = torch.arange(out_features * in_features, dtype=torch.int32).reshape(
        out_features,
        in_features,
    ) % (1 << wbits)
    scales = torch.full((num_groups, out_features), 0.01, dtype=torch.float16)
    zeros = torch.full((num_groups, out_features), 7.0, dtype=torch.float16)

    return GPTQLinear(
        in_features=in_features,
        out_features=out_features,
        wbits=wbits,
        groupsize=groupsize,
        actorder=False,
        quantized_weight=qweight,
        scale=scales,
        zero=zeros,
        bias=None,
        device="cpu",
        pack_weights=packed,
        use_gemlite=False,
    )


def _install_quantized_down_proj(model):
    """Replace the tiny model's down projection with a packed GPTQLinear."""
    original = model.model.layers[0].mlp.down_proj
    replacement = _make_gptq_linear(
        original.in_features,
        original.out_features,
        packed=True,
    )

    model.model.layers[0].mlp.down_proj = replacement
    model.config.quantization_config = {
        "quant_method": "gptq",
        "bits": 4,
        "wbits": 4,
        "group_size": 32,
        "groupsize": 32,
        "sym": True,
        "desc_act": False,
        "actorder": False,
        "checkpoint_format": "gptq",
        "modules_in_block_to_quantize": [QUANTIZED_LAYER_NAME],
        "quantized_layer_names": [QUANTIZED_LAYER_NAME],
        "rotated": False,
        "fp32_had": False,
    }
    return replacement


def _make_post_process_model_config():
    """Build the ModelConfig subset consumed by PostProcessLoraSFT."""
    return SimpleNamespace(
        device="cpu",
        dtype="float32",
        model_id=None,
        path=None,
        load_tokenizer=lambda: _TinyTokenizer(),
    )


def _make_runner_stub(model):
    """Build a Runner instance with only the fields save_quantized_model uses."""
    runner = Runner.__new__(Runner)
    runner.logger = getLogger("test_lora_save_load_roundtrip")
    runner.quantized_model = model
    runner.model_config = SimpleNamespace(
        dtype="float32",
        load_tokenizer=lambda: _FakeTokenizer(),
        get_model_id_or_path=lambda: None,
    )
    return runner


def _load_saved_model(save_dir):
    with patch(
        "onecomp.quantized_model_loader.AutoTokenizer.from_pretrained",
        return_value=_FakeTokenizer(),
    ):
        loaded_model, _ = QuantizedModelLoader.load_quantized_model(
            str(save_dir),
            torch_dtype=torch.float32,
            device_map="",
            local_files_only=True,
        )
    loaded_model.eval()
    return loaded_model


def _collect_saved_state(save_dir):
    return QuantizedModelLoader._load_state_dict_from_dir(str(save_dir))


def _assert_gptq_tensors_are_packed(state_dict, layer_name, layer):
    assert state_dict[f"{layer_name}.qweight"].shape == (
        layer.in_features * layer.wbits // 32,
        layer.out_features,
    )
    assert state_dict[f"{layer_name}.qzeros"].shape == (
        layer.in_features // layer.groupsize,
        layer.out_features * layer.wbits // 32,
    )


def test_gptq_lora_post_process_save_load_roundtrip_preserves_logits(tmp_path):
    """GPTQ + PostProcessLoraSFT saves sidecar and reloads as LoRAGPTQLinear."""
    model = _build_tiny_llama()
    _install_quantized_down_proj(model)
    data_file = tmp_path / "sft.jsonl"
    data_file.write_text('{"text": "round trip sample"}\n', encoding="utf-8")

    post_process = PostProcessLoraSFT(
        data_files=str(data_file),
        epochs=1,
        max_train_samples=1,
        max_length=8,
        batch_size=1,
        gradient_accumulation_steps=1,
        lr=1e-2,
        logging_steps=0,
        warmup_ratio=0.0,
        target_modules=("down_proj",),
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        use_bf16=False,
    )
    post_process.run(model, _make_post_process_model_config())

    lora_layer = model.model.layers[0].mlp.down_proj
    assert isinstance(lora_layer, LoRAGPTQLinear)
    assert lora_layer.base_layer._weight_is_packed is True
    assert lora_layer.lora_B.weight.abs().max().item() > 0.0

    runner = _make_runner_stub(model)
    input_ids = torch.tensor([[1, 7, 11, 13]], dtype=torch.long)

    with torch.no_grad():
        reference_logits = model(input_ids).logits.detach().clone()

    save_dir = tmp_path / "gptq_lora_roundtrip"
    runner.save_quantized_model(str(save_dir))

    state_dict = _collect_saved_state(save_dir)
    assert not any(".base_layer." in key for key in state_dict)
    assert not any(".lora_A." in key or ".lora_B." in key for key in state_dict)
    _assert_gptq_tensors_are_packed(
        state_dict,
        QUANTIZED_LAYER_NAME,
        lora_layer.base_layer,
    )
    assert (save_dir / Runner.LORA_ADAPTER_SUBDIR / "adapter_model.safetensors").is_file()
    assert (save_dir / Runner.LORA_ADAPTER_SUBDIR / "adapter_config.json").is_file()

    loaded_model = _load_saved_model(save_dir)
    loaded_layer = loaded_model.model.layers[0].mlp.down_proj
    assert isinstance(loaded_layer, LoRAGPTQLinear)
    assert isinstance(loaded_layer.base_layer, GPTQLinear)
    assert loaded_layer.base_layer._weight_is_packed is True

    with torch.no_grad():
        loaded_logits = loaded_model(input_ids).logits

    torch.testing.assert_close(loaded_logits, reference_logits, rtol=0.0, atol=1e-4)


def test_gptq_non_lora_save_load_roundtrip_preserves_logits(tmp_path):
    """Non-LoRA GPTQ save/load still reloads packed GPTQLinear and same logits."""
    model = _build_tiny_llama()
    gptq_layer = _install_quantized_down_proj(model)
    runner = _make_runner_stub(model)
    input_ids = torch.tensor([[1, 3, 5, 7]], dtype=torch.long)

    with torch.no_grad():
        reference_logits = model(input_ids).logits.detach().clone()

    save_dir = tmp_path / "gptq_roundtrip"
    runner.save_quantized_model(str(save_dir))

    state_dict = _collect_saved_state(save_dir)
    _assert_gptq_tensors_are_packed(state_dict, QUANTIZED_LAYER_NAME, gptq_layer)
    assert not (save_dir / Runner.LORA_ADAPTER_SUBDIR).exists()

    loaded_model = _load_saved_model(save_dir)
    loaded_layer = loaded_model.model.layers[0].mlp.down_proj
    assert isinstance(loaded_layer, GPTQLinear)
    assert not isinstance(loaded_layer, LoRAGPTQLinear)
    assert loaded_layer._weight_is_packed is True

    with torch.no_grad():
        loaded_logits = loaded_model(input_ids).logits

    torch.testing.assert_close(loaded_logits, reference_logits, rtol=0.0, atol=1e-4)
