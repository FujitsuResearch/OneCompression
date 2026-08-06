"""Direct, re-quantization-free export of OneComp GPTQ weights to GGUF.

Pipeline ("GGUF stitch"):
  1. Build a metadata + tokenizer skeleton GGUF (f16) from a dense HF model
     via llama.cpp's ``convert_hf_to_gguf.py`` (we do not re-implement the
     per-architecture / tokenizer serialization).
  2. Read the skeleton, and for every GPTQ linear that maps losslessly to a
     GGUF legacy block type (Q4_0/Q4_1/Q8_0), replace its f16 tensor with the
     directly packed integer codes + GPTQ scales/zeros. All other tensors
     (embeddings, norms, output, biases) are copied verbatim.
  3. Write the stitched GGUF.

This preserves the exact GPTQ (and therefore QEP-corrected) integer codes, so
the GGUF model has the same accuracy as the OneComp PyTorch inference path,
unlike re-quantizing dequantized weights with llama-quantize.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from __future__ import annotations

import os
import tempfile
from logging import getLogger
from typing import Dict, Optional, Tuple

import numpy as np

from onecomp.cpu.export.blocks import UnsupportedGPTQLayout, pack_gptq_linear
from onecomp.cpu.export.checkpoint import iter_gptq_layers
from onecomp.cpu.export.skeleton import (
    arch_name_map,
    build_skeleton_gguf,
    filter_replacements_by_skeleton,
    gguf_weight_name,
    permute_gptq_rows,
    rope_heads_for_tensor,
    rope_permutation_heads,
    skeleton_logical_shapes,
    stitch_gguf,
)

logger = getLogger(__name__)


def build_replacements(
    quantized_dir: str, tnm, rope_heads: Optional[Tuple[int, int]] = None
) -> Tuple[Dict[str, Tuple[np.ndarray, "object"]], Dict[str, str]]:
    """Pack all losslessly-convertible GPTQ layers; return {gguf_name: (bytes, qtype)}.

    ``rope_heads`` (``(n_head, n_head_kv)`` from :func:`rope_permutation_heads`)
    triggers the interleaved-RoPE row permutation of ``attn_q`` / ``attn_k``
    tensors so the packed codes match the llama.cpp skeleton layout.
    """
    replacements: Dict[str, Tuple[np.ndarray, object]] = {}
    skipped: Dict[str, str] = {}
    for layer in iter_gptq_layers(quantized_dir):
        name = gguf_weight_name(tnm, layer.name)
        if name is None:
            skipped[layer.name] = "no GGUF tensor-name mapping"
            continue
        if layer.actorder:
            skipped[layer.name] = "actorder=True not representable in GGUF blocks"
            continue
        q_int, scales, zeros = layer.q_int, layer.scales, layer.zeros
        n_perm = rope_heads_for_tensor(name, rope_heads)
        if n_perm is not None:
            q_int, scales, zeros = permute_gptq_rows(q_int, scales, zeros, n_perm)
        try:
            packed, qtype = pack_gptq_linear(
                q_int=q_int.cpu().numpy(),
                scales=scales.cpu().numpy(),
                zeros=zeros.cpu().numpy(),
                wbits=layer.wbits,
                sym=layer.sym,
                groupsize=layer.groupsize,
            )
        except UnsupportedGPTQLayout as exc:
            skipped[layer.name] = str(exc)
            continue
        replacements[name] = (packed, qtype)
    return replacements, skipped


def convert_gptq_to_gguf(
    quantized_dir: str,
    out_gguf: str,
    original_model: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Convert an OneComp GPTQ checkpoint to GGUF without re-quantization.

    Args:
        quantized_dir: OneComp quantized model directory (gptq / mixed_gptq).
        out_gguf: Output ``.gguf`` path.
        original_model: Optional path to the original full-precision HF model
            (used only to build the metadata/tokenizer skeleton; if omitted the
            model is dequantized to fp16 to build the skeleton).
        work_dir: Scratch directory (a temp dir is created if omitted).

    Returns:
        A summary dict: number of replaced/skipped tensors and skip reasons.
    """
    owns_workdir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="onecomp_gguf_")
    os.makedirs(work_dir, exist_ok=True)
    try:
        skeleton = build_skeleton_gguf(quantized_dir, original_model, work_dir)
        import gguf

        reader = gguf.GGUFReader(skeleton)
        _arch, tnm = arch_name_map(reader)
        skel_shapes = skeleton_logical_shapes(reader)
        rope_heads = rope_permutation_heads(reader)
        del reader
        replacements, skipped = build_replacements(quantized_dir, tnm, rope_heads=rope_heads)
        # Drop any packed tensor that the skeleton can't accept (missing or a
        # different shape, e.g. exotic per-layer-embedding projections); those
        # layers stay fp16 instead of aborting the export.
        replacements, shape_skipped = filter_replacements_by_skeleton(skel_shapes, replacements)
        skipped.update(shape_skipped)
        n_replaced = stitch_gguf(skeleton, out_gguf, replacements)
        if n_replaced == 0 and skipped:
            raise ValueError(
                "Direct GGUF export replaced 0 weight tensors; every GPTQ layer was "
                f"skipped ({len(skipped)} layer(s)). Use export mode 'mixed' or "
                "'fallback' for this checkpoint. First skips: "
                f"{list(skipped.items())[:4]}"
            )
        if skipped and original_model:
            raise ValueError(
                f"Direct export skipped {len(skipped)} layer(s) while using "
                f"original_model={original_model!r}; skipped layers would retain "
                "pre-quantization FP weights. Omit original_model or use mode "
                "'mixed'/'fallback'. First skips: "
                f"{list(skipped.items())[:4]}"
            )
        return {
            "out_gguf": out_gguf,
            "replaced": n_replaced,
            "skipped": skipped,
        }
    finally:
        if owns_workdir:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)
