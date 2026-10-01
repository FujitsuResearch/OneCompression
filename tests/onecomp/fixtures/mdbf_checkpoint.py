"""Shared builders for tiny MDBF checkpoint tests.

Copyright 2025-2026 Fujitsu Ltd.
"""

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from onecomp.quantizer.mdbf.initialize import MDBFParams
from onecomp.quantizer.mdbf.mdbf_layer import MultipathMDBFLinear

TARGET_SUFFIXES = ("self_attn.q_proj", "mlp.down_proj")
MDBF_RANK = 8
MDBF_PATHS = 2


def make_mdbf_params(n: int, m: int, r: int, l: int, seed: int) -> MDBFParams:
    """Build deterministic, non-degenerate parameters for one MDBF path.

    Args:
        n: Output features.
        m: Input features.
        r: Decomposition rank.
        l: Multi-scale amplitude rank.
        seed: RNG seed.

    Returns:
        MDBF parameters with sign matrices and positive amplitudes.
    """
    generator = torch.Generator().manual_seed(seed)

    def _sign(*shape: int) -> torch.Tensor:
        """Build a deterministic sign tensor."""
        return torch.where(torch.randn(*shape, generator=generator) > 0, 1.0, -1.0)

    def _amplitude(*shape: int) -> torch.Tensor:
        """Build a deterministic positive amplitude tensor."""
        return torch.rand(*shape, generator=generator) + 0.5

    return MDBFParams(
        A_sign=_sign(n, r),
        B_sign=_sign(r, m),
        A_amp=_amplitude(n, l),
        B_amp=_amplitude(m, l),
        Q_U_amp=_amplitude(r, l),
        Q_V_amp=_amplitude(r, l),
    )


def build_mdbf_model(*, with_bias: bool) -> tuple[torch.nn.Module, Any, list[str]]:
    """Build a tiny Llama with selected linears replaced by MDBF layers.

    Args:
        with_bias: Whether replaced linears carry bias.

    Returns:
        Model, config, and quantized layer names.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
        vocab_size=32,
        tie_word_embeddings=False,
        attention_bias=with_bias,
        mlp_bias=with_bias,
    )
    config.torch_dtype = torch.float16
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = LlamaForCausalLM(config).to(torch.float16).eval()

    name_to_module = dict(model.named_modules())
    quantized_names: list[str] = []
    for layer_index in range(config.num_hidden_layers):
        for suffix in TARGET_SUFFIXES:
            name = f"model.layers.{layer_index}.{suffix}"
            quantized_names.append(name)
            parent_name, _, child_name = name.rpartition(".")
            parent = name_to_module[parent_name]
            linear = getattr(parent, child_name)
            bias = linear.bias.detach().clone() if linear.bias is not None else None
            params_list = [
                make_mdbf_params(
                    linear.out_features,
                    linear.in_features,
                    MDBF_RANK,
                    1,
                    seed=1000 * layer_index + 7 * path_index + len(suffix),
                )
                for path_index in range(MDBF_PATHS)
            ]
            setattr(
                parent,
                child_name,
                MultipathMDBFLinear(params_list, bias=bias, use_gemlite=False),
            )

    return model, config, quantized_names


def write_mdbf_save_dir(
    save_dir: Path,
    config: Any,
    state_dict: dict[str, torch.Tensor],
    quantized_names: list[str],
    *,
    record_paths: bool = True,
    rotated: bool = False,
) -> None:
    """Persist a tiny MDBF checkpoint.

    Args:
        save_dir: Destination directory.
        config: Model configuration.
        state_dict: Tensors to save.
        quantized_names: MDBF layer names.
        record_paths: Whether to record the configured path count.
        rotated: Whether to mark the checkpoint as rotated.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    config_dict = config.to_dict()
    config_dict["torch_dtype"] = "float16"
    config_dict["quantization_config"] = {
        "quant_method": "mdbf",
        "bits": 2.0,
        "l": 1,
        "modules_in_block_to_quantize": quantized_names,
        "rotated": rotated,
    }
    if record_paths:
        config_dict["quantization_config"]["P"] = MDBF_PATHS
    (save_dir / "config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    save_file(
        {key: tensor.contiguous() for key, tensor in state_dict.items()},
        str(save_dir / "model.safetensors"),
    )
