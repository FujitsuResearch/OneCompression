"""GGUF export for OneComp GPTQ checkpoints.

Modules:
    blocks      -- lossless packing of GPTQ codes into GGUF legacy blocks
    checkpoint  -- read an OneComp GPTQ checkpoint into ``GPTQLayer`` objects
    dequantize  -- reconstruct a dense fp16 HF model from a GPTQ checkpoint
    skeleton    -- build a metadata/tokenizer skeleton GGUF and stitch tensors
    direct      -- direct, lossless GPTQ -> GGUF export (preferred)
    fallback    -- dequantize -> llama-quantize export (re-quantizes; universal)
    rotation    -- fold a rotated model's online down_proj Hadamard into weights
    auto        -- single entry point that routes by quant_method / rotation

Copyright 2025-2026 Fujitsu Ltd.
"""

from onecomp.cpu.export.auto import export_to_gguf, plan_export
from onecomp.cpu.export.blocks import (
    UnsupportedGPTQLayout,
    pack_gptq_linear,
    select_gguf_type,
)
from onecomp.cpu.export.checkpoint import (
    GPTQLayer,
    QuantMeta,
    dequantize_layer,
    iter_gptq_layers,
    load_quant_config,
    read_quant_meta,
)
from onecomp.cpu.export.dequantize import dequantize_to_hf
from onecomp.cpu.export.direct import build_replacements, convert_gptq_to_gguf
from onecomp.cpu.export.fallback import export_via_dequantize
from onecomp.cpu.export.rotation import defold_down_proj_hadamard
from onecomp.cpu.export.skeleton import (
    arch_name_map,
    build_skeleton_gguf,
    gguf_weight_name,
    stitch_gguf,
)

__all__ = [
    "UnsupportedGPTQLayout",
    "pack_gptq_linear",
    "select_gguf_type",
    "GPTQLayer",
    "QuantMeta",
    "iter_gptq_layers",
    "dequantize_layer",
    "load_quant_config",
    "read_quant_meta",
    "dequantize_to_hf",
    "convert_gptq_to_gguf",
    "build_replacements",
    "export_via_dequantize",
    "defold_down_proj_hadamard",
    "export_to_gguf",
    "plan_export",
    "build_skeleton_gguf",
    "stitch_gguf",
    "arch_name_map",
    "gguf_weight_name",
]
