"""llama.cpp mixed-bit GPTQ "plugin": export per-module mixed-precision GGUF.

This is the llama.cpp counterpart of ``vllm_plugins/gptq/vllm_plugin.py``.

llama.cpp has no run-time plugin mechanism for new quantization types (they are
compiled into ggml).  Instead, the unit of extensibility is the *GGUF file*:
every tensor carries its own quantization type and llama.cpp dispatches the
matching (already compiled) kernel per tensor.  So the llama.cpp equivalent of
the vLLM ``mixed_gptq`` plugin is an *exporter* that reads the very same
``quantization_bits`` config and writes each module with the GGUF type that
matches its bit-width:

    4-bit sym -> Q4_0   8-bit sym -> Q8_0   4-bit asym -> Q4_1   (direct, lossless)
    2-bit     -> Q2_K   3-bit     -> Q3_K                        (K-quant fallback)

Direct layers reuse the exact GPTQ (QEP-corrected) integer codes with no
re-quantization; fallback layers are re-quantized from their *dequantized*
weights via ``llama-quantize`` because no lossless legacy GGUF type exists for
those bit-widths.  The resulting single GGUF runs natively on llama.cpp with
genuinely mixed per-layer precision.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import numpy as np

from llamacpp_plugins.gptq.constants import (
    ROUTE_DENSE,
    ROUTE_DIRECT,
    ROUTE_KQUANT,
    select_gguf_route,
)
from onecomp.cpu.export.blocks import UnsupportedGPTQLayout, pack_gptq_linear
from onecomp.cpu.export.checkpoint import GPTQLayer, iter_gptq_layers
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
from onecomp.cpu.llama_tooling import run_llama_quantize_per_tensor

logger = getLogger(__name__)

# K-quant chosen when an otherwise-direct layer cannot use a legacy block type
# (e.g. act-order reordering breaks the per-block scale alignment).  The layer
# is still re-quantized from its dequantized weights, so this is correct, only
# lossy.
_ACTORDER_KQUANT = {2: "Q2_K", 3: "Q3_K", 4: "Q4_K", 5: "Q5_K", 6: "Q6_K", 8: "Q6_K"}


@dataclass
class ModulePlan:
    """How one GPTQ module will be written to GGUF."""

    name: str  # HF module name
    bits: int
    sym: bool
    groupsize: int
    actorder: bool
    route: str  # direct / kquant / dense
    ggml_type: str  # GGUF type name (e.g. "Q4_0", "Q3_K", "F16")
    reason: str = ""


def _direct_pack_feasible(layer: GPTQLayer) -> bool:
    """Whether a DIRECT-routed layer can actually be packed losslessly.

    Mirrors the guards in :func:`onecomp.cpu.export.blocks.pack_gptq_linear`
    (32-aligned input dim, group a multiple of 32, and the symmetric zero point
    being exactly 8 for Q4_0 / 128 for Q8_0) so the *plan* matches what the
    export will really do. This keeps ``has_kquant`` accurate, which in turn
    forces the GPTQ-dequantized skeleton when a demote is unavoidable.
    """
    if layer.in_features % 32 != 0:
        return False
    if layer.groupsize != -1 and layer.groupsize % 32 != 0:
        return False
    z = layer.zeros
    if layer.wbits == 4 and not layer.sym:
        return True  # Q4_1 (value = d*q + m) represents any zero point
    if layer.wbits == 4 and layer.sym:
        return bool((z == 8).all())
    if layer.wbits == 8 and layer.sym:
        return bool((z == 128).all())
    return False


def _route_layer(layer: GPTQLayer) -> Tuple[str, str, str]:
    """Decide (route, ggml_type_name, reason) for one unpacked GPTQ layer."""
    route, gtype = select_gguf_route(layer.wbits, layer.sym)

    if layer.actorder and route == ROUTE_DIRECT:
        # Act-order permutes input columns, so a GGUF block of 32 no longer maps
        # to a single GPTQ group -> fall back to a K-quant of matching width.
        kq = _ACTORDER_KQUANT.get(layer.wbits)
        if kq is not None:
            return ROUTE_KQUANT, kq, "actorder=True: re-quantized to K-quant"
        return ROUTE_DENSE, "F16", "actorder=True with no K-quant fallback"

    if route == ROUTE_DIRECT and not _direct_pack_feasible(layer):
        # Zero point / shape prevents a lossless legacy pack; re-quantize from the
        # GPTQ-dequantized weights (decided here, not at runtime, so the skeleton
        # is built from dequantized GPTQ values rather than the original FP model).
        kq = _ACTORDER_KQUANT.get(layer.wbits, "Q4_K")
        return ROUTE_KQUANT, kq, "direct pack infeasible (zero-point/shape): K-quant fallback"

    return route, gtype.name, ""


def plan_mixed_export(quantized_dir: str) -> List[ModulePlan]:
    """Return the per-module GGUF routing plan for a (mixed_)gptq checkpoint.

    Does not build a skeleton or pack anything; useful for previewing how a
    checkpoint will be laid out before running the (slower) export.
    """
    plans: List[ModulePlan] = []
    for layer in iter_gptq_layers(quantized_dir):
        route, gtype_name, reason = _route_layer(layer)
        plans.append(
            ModulePlan(
                name=layer.name,
                bits=layer.wbits,
                sym=layer.sym,
                groupsize=layer.groupsize,
                actorder=layer.actorder,
                route=route,
                ggml_type=gtype_name,
                reason=reason,
            )
        )
    return plans


def summarize_plan(plans: List[ModulePlan]) -> Dict[str, object]:
    """Aggregate a plan into counts by route and by GGUF type."""
    by_route: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    for p in plans:
        by_route[p.route] = by_route.get(p.route, 0) + 1
        by_type[p.ggml_type] = by_type.get(p.ggml_type, 0) + 1
    return {"modules": len(plans), "by_route": by_route, "by_type": by_type}


def export_mixed_gptq_gguf(
    quantized_dir: str,
    out_gguf: str,
    original_model: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Export a (mixed_)gptq checkpoint to a single mixed-precision GGUF.

    Direct (4-bit sym/asym, 8-bit sym) layers are packed losslessly from the
    GPTQ codes; 2/3-bit (and act-order) layers are K-quantized from their
    dequantized weights via ``llama-quantize``.

    Args:
        quantized_dir: OneComp ``gptq`` / ``mixed_gptq`` checkpoint directory.
        out_gguf: Output ``.gguf`` path.
        original_model: Optional original FP model dir for skeleton metadata.
            Ignored (forced to the dequantize path) when any K-quant fallback
            layer is present, so those layers reflect the GPTQ-optimized values.
        work_dir: Scratch directory (a temp dir is created if omitted).

    Returns:
        Summary dict with the routing breakdown and replacement counts.
    """
    plans = plan_mixed_export(quantized_dir)
    has_kquant = any(p.route == ROUTE_KQUANT for p in plans)

    owns_workdir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="onecomp_mixed_gguf_")
    os.makedirs(work_dir, exist_ok=True)
    try:
        # When K-quant fallback layers exist, the skeleton must carry the GPTQ
        # *dequantized* values (not the original FP weights) so llama-quantize
        # re-quantizes the GPTQ-optimized weights. Force the dequantize path.
        skeleton_src = None if has_kquant else original_model
        skeleton = build_skeleton_gguf(quantized_dir, skeleton_src, work_dir)

        import gguf

        reader = gguf.GGUFReader(skeleton)
        _arch, tnm = arch_name_map(reader)
        skel_shapes = skeleton_logical_shapes(reader)
        rope_heads = rope_permutation_heads(reader)
        del reader

        direct_replacements: Dict[str, Tuple[np.ndarray, object]] = {}
        kquant_targets: Dict[str, str] = {}
        dense: Dict[str, str] = {}
        unmapped: Dict[str, str] = {}

        for layer in iter_gptq_layers(quantized_dir):
            gguf_name = gguf_weight_name(tnm, layer.name)
            if gguf_name is None:
                unmapped[layer.name] = "no GGUF tensor-name mapping"
                continue
            route, gtype_name, _reason = _route_layer(layer)

            if route == ROUTE_DIRECT:
                q_int, scales, zeros = layer.q_int, layer.scales, layer.zeros
                n_perm = rope_heads_for_tensor(gguf_name, rope_heads)
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
                    direct_replacements[gguf_name] = (packed, qtype)
                except UnsupportedGPTQLayout as exc:
                    # Demote to K-quant fallback rather than failing the export.
                    kq = _ACTORDER_KQUANT.get(layer.wbits, "Q4_K")
                    kquant_targets[gguf_name] = kq
                    logger.warning("Direct pack failed for %s (%s); using %s", layer.name, exc, kq)
            elif route == ROUTE_KQUANT:
                kquant_targets[gguf_name] = gtype_name
            else:
                dense[gguf_name] = gtype_name

        # Drop packed tensors the skeleton can't accept (missing / shape-mismatch,
        # e.g. exotic per-layer projections) so the export never aborts.
        direct_replacements, shape_skipped = filter_replacements_by_skeleton(
            skel_shapes, direct_replacements
        )
        unmapped.update(shape_skipped)

        # Stage 1: re-quantize the fallback layers to their K-quant types.
        # llama-quantize needs a *quantized* default ftype to honour per-tensor
        # overrides, so direct layers are quantized to Q8_0 here and overwritten
        # losslessly in stage 2; dense layers are pinned to F16; token_embd /
        # output stay F16 via the dedicated flags.
        if kquant_targets:
            stage1_types: Dict[str, str] = dict(kquant_targets)
            for name in dense:
                stage1_types[name] = "F16"
            intermediate = os.path.join(work_dir, "mixed.kquant.gguf")
            run_llama_quantize_per_tensor(
                skeleton, intermediate, stage1_types, default_type="Q8_0"
            )
        else:
            intermediate = skeleton

        # Stage 2: stitch the lossless direct packs onto the intermediate GGUF
        # (already-quantized K-quant tensors are copied through verbatim).
        n_replaced = stitch_gguf(intermediate, out_gguf, direct_replacements)

        summary = {
            "out_gguf": out_gguf,
            "direct_lossless": n_replaced,
            "kquant_fallback": len(kquant_targets),
            "dense_fp16": len(dense),
            "unmapped": unmapped,
            "plan": summarize_plan(plans),
        }
        logger.info("Mixed GGUF export summary: %s", summary)
        return summary
    finally:
        if owns_workdir:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)
