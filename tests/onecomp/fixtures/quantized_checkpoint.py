"""Tiny quantized checkpoints in the layout read by ``load_quantized_model``.

The models are built from local configs and saved with a local tokenizer, so
loading them needs neither CUDA nor network access.

Copyright 2025-2026 Fujitsu Ltd.
"""

import contextlib
import json
from itertools import chain
from pathlib import Path
from typing import Callable, Optional

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from onecomp import quantized_model_loader_streaming
from onecomp.quantizer.dbf.dbf_layer import DoubleBinaryLinear
from onecomp.quantizer.gptq.gptq_layer import GPTQLinear

QUANTIZED_SUFFIXES = ("self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj")
VOCAB_SIZE = 64
_GROUP_SIZE = 32
_MODEL_CLASSES = {
    "qwen3": (Qwen3Config, Qwen3ForCausalLM),
    "llama": (LlamaConfig, LlamaForCausalLM),
}


def _random_gptq_linear(
    linear: torch.nn.Linear, wbits: int, generator: torch.Generator
) -> GPTQLinear:
    """Return a packed actorder ``GPTQLinear`` with random codes, scales and zeros."""
    in_features, out_features = linear.in_features, linear.out_features
    num_groups = in_features // _GROUP_SIZE
    return GPTQLinear(
        in_features=in_features,
        out_features=out_features,
        wbits=wbits,
        groupsize=_GROUP_SIZE,
        actorder=True,
        quantized_weight=torch.randint(
            0, 2**wbits, (out_features, in_features), generator=generator, dtype=torch.int32
        ),
        scale=(torch.rand(num_groups, out_features, generator=generator) * 0.02 + 0.01).half(),
        zero=torch.randint(1, 2**wbits, (num_groups, out_features), generator=generator).half(),
        perm=torch.randperm(in_features, generator=generator),
        device="cpu",
        pack_weights=True,
        use_gemlite=False,
    )


def _random_dbf_linear(linear: torch.nn.Linear, generator: torch.Generator) -> DoubleBinaryLinear:
    """Return a ``DoubleBinaryLinear`` with random binary factors and scales."""
    in_features, out_features = linear.in_features, linear.out_features
    mid_features = min(in_features, out_features)

    def signs(*shape: int) -> torch.Tensor:
        return torch.randint(0, 2, shape, generator=generator).float() * 2 - 1

    return DoubleBinaryLinear(
        dbf_Da=torch.rand(out_features, generator=generator),
        dbf_A=signs(out_features, mid_features),
        dbf_mid=torch.rand(mid_features, generator=generator),
        dbf_B=signs(mid_features, in_features),
        dbf_Db=torch.rand(in_features, generator=generator),
        use_gemlite=False,
    )


def _quantization_config(
    quant_method: str, layer_bits: tuple[int, ...], checkpoint_format: str
) -> dict:
    """Return a ``quantization_config`` like the one saved for these layers."""
    names = [
        f"model.layers.{index}.{suffix}"
        for index in range(len(layer_bits))
        for suffix in QUANTIZED_SUFFIXES
    ]
    if quant_method == "dbf":
        return {"quant_method": "dbf", "bits": 1.5, "modules_in_block_to_quantize": names}
    config = {
        "quant_method": "gptq",
        "bits": layer_bits[0],
        "group_size": _GROUP_SIZE,
        "desc_act": True,
        "sym": False,
        "checkpoint_format": checkpoint_format,
        "modules_in_block_to_quantize": names,
    }
    if len(set(layer_bits)) > 1:
        config["quant_method"] = "mixed_gptq"
        config["quantization_bits"] = [
            {
                suffix: {"bits": bits, "method": "gptq", "params": {"group_size": _GROUP_SIZE}}
                for suffix in QUANTIZED_SUFFIXES
            }
            for bits in layer_bits
        ]
    return config


def _write_tokenizer(save_dir: Path) -> None:
    """Save a word-level tokenizer whose ids cover the model vocabulary."""
    vocab = {f"w{index}": index for index in range(VOCAB_SIZE)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="w0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="w0").save_pretrained(save_dir)


def write_quantized_checkpoint(
    save_dir: Path,
    *,
    quant_method: str = "gptq",
    model_type: str = "qwen3",
    tie_word_embeddings: bool = True,
    layer_bits: tuple[int, ...] = (4, 4),
    config_dtype: str = "float16",
    fp32_norm: bool = False,
    checkpoint_format: str = "gptq",
    num_shards: int = 1,
    edit_state_dict: Optional[Callable[[dict[str, torch.Tensor]], object]] = None,
    edit_quantization_config: Optional[Callable[[dict], object]] = None,
) -> None:
    """Save a model whose decoder layers each have quantized projections.

    Non-quantized weights are stored in float16 whatever ``config_dtype`` says.
    As in Hugging Face checkpoints, a tied ``lm_head.weight`` is not stored and
    split checkpoints come with ``model.safetensors.index.json``.

    Args:
        save_dir: Directory to create.
        quant_method: ``"gptq"`` or ``"dbf"``.
        model_type: ``"qwen3"`` or ``"llama"``.
        tie_word_embeddings: Whether ``lm_head`` shares the input embedding.
        layer_bits: GPTQ bit width of each decoder layer. Differing widths are
            saved as ``mixed_gptq`` with a per-layer ``quantization_bits`` table.
            DBF uses only the number of layers.
        config_dtype: ``dtype`` recorded in ``config.json``.
        fp32_norm: Store the RMSNorm weights in float32.
        checkpoint_format: GPTQ ``"gptq"`` or ``"gptq_v2"``.
        num_shards: Number of safetensors files.
        edit_state_dict: Called on the tensors to save, to alter the checkpoint.
        edit_quantization_config: Called on the ``quantization_config`` to save.
    """
    config_class, model_class = _MODEL_CLASSES[model_type]
    config = config_class(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(layer_bits),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
    )
    generator = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    model = model_class(config).half()
    for layer, bits in zip(model.model.layers, layer_bits):
        for suffix in QUANTIZED_SUFFIXES:
            parent_name, name = suffix.split(".")
            parent = getattr(layer, parent_name)
            linear = getattr(parent, name)
            if quant_method == "dbf":
                setattr(parent, name, _random_dbf_linear(linear, generator))
            else:
                setattr(parent, name, _random_gptq_linear(linear, bits, generator))

    state_dict = {
        key: tensor.float() if fp32_norm and key.endswith("norm.weight") else tensor
        for key, tensor in model.state_dict().items()
        if not (tie_word_embeddings and key == "lm_head.weight")
    }
    if edit_state_dict is not None:
        edit_state_dict(state_dict)
    save_dir.mkdir(parents=True)
    keys = list(state_dict)
    if num_shards == 1:
        save_file(state_dict, str(save_dir / "model.safetensors"))
    else:
        weight_map = {}
        for index in range(num_shards):
            file_name = f"model-{index + 1:05d}-of-{num_shards:05d}.safetensors"
            shard_keys = keys[index::num_shards]
            save_file({key: state_dict[key] for key in shard_keys}, str(save_dir / file_name))
            weight_map.update(dict.fromkeys(shard_keys, file_name))
        index_file = save_dir / "model.safetensors.index.json"
        index_file.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))

    config_dict = config.to_dict()
    config_dict["dtype"] = config_dtype
    config_dict["quantization_config"] = _quantization_config(
        quant_method, layer_bits, checkpoint_format
    )
    if edit_quantization_config is not None:
        edit_quantization_config(config_dict["quantization_config"])
    (save_dir / "config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    _write_tokenizer(save_dir)


def assert_same_loaded_model(actual: torch.nn.Module, expected: torch.nn.Module) -> None:
    """Assert equal modules, tensors (values, dtypes, devices), weight tying and placement."""
    assert [type(module) for module in actual.modules()] == [
        type(module) for module in expected.modules()
    ]
    assert [module.training for module in actual.modules()] == [
        module.training for module in expected.modules()
    ]
    actual_tensors = dict(chain(actual.named_parameters(), actual.named_buffers()))
    expected_tensors = dict(chain(expected.named_parameters(), expected.named_buffers()))
    assert actual_tensors.keys() == expected_tensors.keys()
    for name, tensor in expected_tensors.items():
        assert actual_tensors[name].dtype == tensor.dtype, name
        assert actual_tensors[name].device == tensor.device, name
        assert actual_tensors[name].requires_grad == tensor.requires_grad, name
        assert torch.equal(actual_tensors[name], tensor), name
    assert (actual.lm_head.weight is actual.get_input_embeddings().weight) == (
        expected.lm_head.weight is expected.get_input_embeddings().weight
    )
    assert getattr(actual, "hf_device_map", None) == getattr(expected, "hf_device_map", None)
    assert hasattr(actual, "_hf_hook") == hasattr(expected, "_hf_hook")
    assert actual.config.quantization_config == expected.config.quantization_config


def intercept_tensor_reads(
    monkeypatch: pytest.MonkeyPatch, on_read: Callable[[str], None]
) -> None:
    """Call ``on_read(key)`` before the streaming loader reads the value of each tensor."""
    real_safe_open = quantized_model_loader_streaming.safe_open

    class InterceptingReader:
        def __init__(self, reader):
            self._reader = reader

        def __getattr__(self, name):
            return getattr(self._reader, name)

        def get_tensor(self, key):
            on_read(key)
            return self._reader.get_tensor(key)

    @contextlib.contextmanager
    def intercepting_safe_open(*args, **kwargs):
        with real_safe_open(*args, **kwargs) as reader:
            yield InterceptingReader(reader)

    monkeypatch.setattr(quantized_model_loader_streaming, "safe_open", intercepting_safe_open)
