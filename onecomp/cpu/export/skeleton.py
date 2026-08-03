"""GGUF skeleton building and tensor stitching shared by the export paths.

A "skeleton" GGUF carries the correct metadata + tokenizer + tensor names for a
model (produced by llama.cpp's ``convert_hf_to_gguf.py``); the export paths then
*stitch* their own packed quantized tensors onto it. Centralizing the skeleton /
stitch / tensor-name-mapping helpers here keeps the direct (lossless) exporter
and the mixed-precision plugin from duplicating this fiddly ``gguf`` plumbing.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from __future__ import annotations

import os
from logging import getLogger
from typing import Dict, Optional, Tuple

import numpy as np

from onecomp.cpu.export.checkpoint import load_quant_config, needs_mixed_export
from onecomp.cpu.export.dequantize import dequantize_to_hf
from onecomp.cpu.llama_tooling import run_convert_hf_to_gguf

logger = getLogger(__name__)


def build_skeleton_gguf(
    quantized_dir: str,
    original_model: Optional[str],
    work_dir: str,
) -> str:
    """Produce an f16 skeleton GGUF (correct metadata + tokenizer + tensor names).

    If ``original_model`` points at a local full-precision HF model it is used
    directly; otherwise the GPTQ checkpoint is dequantized to fp16 first so that
    the skeleton (and any layers later re-quantized from it) reflects the
    GPTQ-optimized values.
    """
    skeleton = os.path.join(work_dir, "skeleton.f16.gguf")
    qcfg = load_quant_config(quantized_dir)
    # Skipped / K-quant layers are re-quantized from the skeleton. Using the
    # original FP model would bake pre-GPTQ weights into those tensors.
    if original_model and os.path.isdir(original_model) and needs_mixed_export(qcfg):
        logger.warning(
            "Ignoring original_model=%s for skeleton: checkpoint needs mixed/K-quant "
            "export; building from GPTQ-dequantized weights instead.",
            original_model,
        )
        original_model = None
    if original_model and os.path.isdir(original_model):
        logger.info("Building skeleton GGUF from original model %s", original_model)
        run_convert_hf_to_gguf(original_model, skeleton, outtype="f16")
    else:
        dense_dir = os.path.join(work_dir, "dense_fp16")
        logger.info("Dequantizing GPTQ -> dense fp16 to build skeleton")
        dequantize_to_hf(quantized_dir, dense_dir)
        run_convert_hf_to_gguf(dense_dir, skeleton, outtype="f16")
    return skeleton


def rope_permutation_heads(reader) -> Optional[Tuple[int, int]]:
    """Return ``(n_head, n_head_kv)`` when stitched Q/K rows must be permuted.

    llama.cpp's ``convert_hf_to_gguf.py`` permutes ``attn_q`` / ``attn_k`` rows
    of ``llama``-architecture models from the Hugging Face half-split RoPE
    layout to the interleaved ("NORM") layout, and llama.cpp applies
    interleaved RoPE at inference. Tensors stitched onto such a skeleton from
    an HF-layout GPTQ checkpoint must therefore be permuted the same way.
    Returns ``None`` for architectures that keep the HF layout (qwen2, gemma,
    ... -- NEOX-style RoPE).
    """
    arch = reader.get_field("general.architecture").contents()
    if arch != "llama":
        return None
    n_head = int(reader.get_field(f"{arch}.attention.head_count").contents())
    kv_field = reader.get_field(f"{arch}.attention.head_count_kv")
    if kv_field is None:
        n_head_kv = n_head
    else:
        kv = kv_field.contents()
        if isinstance(kv, (list, tuple)):
            if len(set(int(x) for x in kv)) != 1:
                raise ValueError(
                    "Per-layer head_count_kv is not supported for the stitched "
                    f"Q/K RoPE permutation (got {kv})."
                )
            kv = kv[0]
        n_head_kv = int(kv)
    return n_head, n_head_kv


def rope_heads_for_tensor(gguf_tensor_name: str, rope_heads: Optional[Tuple[int, int]]):
    """Return the head count to permute ``gguf_tensor_name`` with, or ``None``."""
    if rope_heads is None:
        return None
    if ".attn_q." in gguf_tensor_name:
        return rope_heads[0]
    if ".attn_k." in gguf_tensor_name:
        return rope_heads[1]
    return None


def permute_gptq_rows(q_int, scales, zeros, n_head: int):
    """Permute a GPTQ layer's output rows to the interleaved RoPE layout.

    The permutation acts on output rows only, so the GPTQ grouping (over input
    columns) is untouched and the repacking stays lossless: ``q_int`` is
    ``(out, in)`` (permute rows) and ``scales`` / ``zeros`` are
    ``(num_groups, out)`` (permute columns with the same row order).
    """
    from onecomp.export.gguf_export import permute_rope_rows

    q = permute_rope_rows(q_int, n_head)
    s = permute_rope_rows(scales.transpose(0, 1), n_head).transpose(0, 1).contiguous()
    z = permute_rope_rows(zeros.transpose(0, 1), n_head).transpose(0, 1).contiguous()
    return q, s, z


def arch_name_map(reader) -> Tuple[str, "object"]:
    """Return ``(architecture, TensorNameMap)`` for a GGUF reader."""
    import gguf

    arch = reader.get_field("general.architecture").contents()
    block_count_field = reader.get_field(f"{arch}.block_count")
    n_blocks = int(block_count_field.contents()) if block_count_field is not None else 0
    name_to_arch = {v: k for k, v in gguf.MODEL_ARCH_NAMES.items()}
    if arch not in name_to_arch:
        raise ValueError(f"Unknown GGUF architecture '{arch}' for tensor name mapping.")
    return arch, gguf.get_tensor_name_map(name_to_arch[arch], n_blocks)


def _name_candidates(hf_module_name: str):
    """Yield equivalent HF module names, normalizing multimodal text-model nesting.

    Multimodal checkpoints (e.g. Gemma4ForConditionalGeneration) nest the text
    decoder under ``language_model`` (``model.language_model.layers.N.*``), but
    ``gguf``'s text TensorNameMap keys are the un-nested ``model.layers.N.*``.
    Try the raw name first, then the de-nested variants.
    """
    seen = set()
    cands = [
        hf_module_name,
        hf_module_name.replace(".language_model.", "."),
        hf_module_name.replace("language_model.", "", 1),
    ]
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            yield c


def gguf_weight_name(tnm, hf_module_name: str) -> Optional[str]:
    """Map an HF module name (no suffix) to a GGUF '*.weight' tensor name."""
    for cand in _name_candidates(hf_module_name):
        base = tnm.get_name(cand)
        if base is not None:
            return f"{base}.weight"
    return None


def skeleton_logical_shapes(reader) -> Dict[str, Tuple[int, ...]]:
    """Return ``{tensor_name: (out, in, ...)}`` logical shapes for a GGUF reader.

    ``GGUFReader`` reports shapes reversed (in, out); this flips them back to the
    PyTorch ``(out_features, in_features)`` convention used by the packer.
    """
    return {t.name: tuple(int(x) for x in reversed(t.shape)) for t in reader.tensors}


def filter_replacements_by_skeleton(
    skeleton_shapes: Dict[str, Tuple[int, ...]],
    replacements: Dict[str, Tuple[np.ndarray, object]],
) -> Tuple[Dict[str, Tuple[np.ndarray, object]], Dict[str, str]]:
    """Drop packed tensors that are absent from / shape-incompatible with the skeleton.

    Heterogeneous architectures (e.g. Gemma4 per-layer-embedding projections)
    can expose linears that either have no GGUF tensor or whose GGUF tensor has a
    different shape. Rather than abort the whole export, such layers are skipped
    (left as the skeleton's fp16 weight) and reported.

    Returns ``(kept_replacements, skipped_reasons)``.
    """
    from gguf.quants import quant_shape_from_byte_shape

    kept: Dict[str, Tuple[np.ndarray, object]] = {}
    skipped: Dict[str, str] = {}
    for name, (packed, qtype) in replacements.items():
        skel = skeleton_shapes.get(name)
        if skel is None:
            skipped[name] = "GGUF tensor not present in skeleton"
            continue
        logical = tuple(int(x) for x in quant_shape_from_byte_shape(packed.shape, qtype))
        if logical != skel:
            skipped[name] = f"shape mismatch packed{logical} vs skeleton{skel}"
            continue
        kept[name] = (packed, qtype)
    return kept, skipped


def stitch_gguf(
    skeleton: str,
    out_gguf: str,
    replacements: Dict[str, Tuple[np.ndarray, object]],
) -> int:
    """Copy skeleton GGUF to out_gguf, substituting packed tensors for replacements.

    Tensors not in ``replacements`` are copied verbatim; dense (F16/F32) tensors
    let the writer infer their dtype, while already-quantized tensors (e.g.
    K-quant layers produced by ``llama-quantize``) keep their raw block bytes and
    quant type.
    """
    import gguf
    from gguf.constants import GGMLQuantizationType, GGUFValueType
    from gguf.quants import quant_shape_from_byte_shape

    reader = gguf.GGUFReader(skeleton)
    arch = reader.get_field("general.architecture").contents()
    writer = gguf.GGUFWriter(out_gguf, arch)

    for field in reader.fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vtype, sub_type=sub_type)

    n_replaced = 0
    for tensor in reader.tensors:
        if tensor.name in replacements:
            packed, qtype = replacements[tensor.name]
            # packed byte shape -> logical (out, in); GGUFReader.shape is reversed (in, out).
            logical = tuple(int(x) for x in quant_shape_from_byte_shape(packed.shape, qtype))
            skeleton_logical = tuple(int(x) for x in reversed(tensor.shape))
            if logical != skeleton_logical:
                raise ValueError(
                    f"Shape mismatch for {tensor.name}: packed->{logical} "
                    f"vs skeleton {skeleton_logical}"
                )
            writer.add_tensor(tensor.name, packed, raw_dtype=qtype)
            n_replaced += 1
        else:
            data = np.ascontiguousarray(tensor.data)
            if tensor.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
                # Dense tensor: let the writer infer the dtype from the numpy array.
                writer.add_tensor(tensor.name, data)
            else:
                # Already-quantized tensor (e.g. a K-quant fallback layer produced
                # by llama-quantize): copy the raw block bytes verbatim, preserving
                # both the quant type and the logical shape.
                writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    logger.info("Stitched GGUF written to %s (%d tensors replaced)", out_gguf, n_replaced)
    return n_replaced
