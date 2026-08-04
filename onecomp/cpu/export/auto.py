"""Single entry point that picks the right GGUF export path for a checkpoint.

OneComp can save several quantization families; each maps to CPU/GGUF differently:

============  ====================================  =========================
quant_method  layout                                export path
============  ====================================  =========================
gptq          AutoGPTQ qweight/qzeros/scales        direct (lossless)
mixed_gptq    same, per-layer bitwidths             mixed (lossless + K-quant)
jointq / rtn  same AutoGPTQ layout                  direct (lossless)
dbf           DoubleBinaryLinear (binary factors)   fallback (dequantize)
autobit       mix of gptq/dbf children              fallback (dequantize)
============  ====================================  =========================

QEP only changes the *integer codes* of a GPTQ checkpoint, so QEP-corrected
models export through the very same lossless direct/mixed paths.

Rotation pre-processing (``rotated=true``) keeps an online Hadamard on
``down_proj`` that ``llama.cpp`` cannot reproduce; such models are routed through
the dequantize fallback, which folds the Hadamard back into the weight
(see :mod:`onecomp.cpu.export.rotation`) so the GGUF runs correctly with no
online operation.

OneBit (``quant_method == "onebit"``) is intentionally unsupported.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from __future__ import annotations

from logging import getLogger
from typing import Dict, Optional

from onecomp.cpu.export.checkpoint import (
    configured_bit_widths,
    load_quant_config,
    needs_mixed_export,
    read_quant_meta,
)

logger = getLogger(__name__)

# Methods we will not export (no faithful GGUF representation / out of scope).
_UNSUPPORTED = {"onebit"}


def plan_export(quantized_dir: str) -> Dict[str, object]:
    """Decide which export path to use for ``quantized_dir`` (no side effects).

    Returns a dict with ``method`` (direct / mixed / fallback / unsupported),
    plus the parsed :class:`QuantMeta` fields, and a human-readable ``reason``.
    """
    meta = read_quant_meta(quantized_dir)
    if meta.quant_method in _UNSUPPORTED:
        return {
            "path": "unsupported",
            "meta": meta,
            "reason": f"{meta.quant_method} is not supported",
        }
    if meta.rotated:
        return {
            "path": "fallback",
            "meta": meta,
            "reason": "rotated model: down_proj online Hadamard folded into weights via dequantize",
        }
    if not meta.is_gptq_family:
        return {
            "path": "fallback",
            "meta": meta,
            "reason": f"{meta.quant_method} has no lossless GGUF block layout; dequantize + requantize",
        }
    if meta.quant_method == "mixed_gptq":
        return {
            "path": "mixed",
            "meta": meta,
            "reason": "mixed-bit GPTQ: lossless packing + K-quant fallback",
        }
    qcfg = load_quant_config(quantized_dir)
    if needs_mixed_export(qcfg):
        bits = sorted(configured_bit_widths(qcfg))
        reason = (
            "act-order GPTQ: per-layer K-quant (direct packing not block-aligned)"
            if meta.actorder
            else f"bit-width(s) {bits} need K-quant fallback (no lossless GGUF block type)"
        )
        return {"path": "mixed", "meta": meta, "reason": reason}
    return {
        "path": "direct",
        "meta": meta,
        "reason": "uniform GPTQ family: lossless direct packing",
    }


def export_to_gguf(
    quantized_dir: str,
    out_gguf: str,
    mode: str = "auto",
    qtype: str = "Q4_K_M",
    original_model: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Export any (supported) OneComp checkpoint to a GGUF, choosing the best path.

    Args:
        quantized_dir: OneComp quantized checkpoint directory.
        out_gguf: Output ``.gguf`` path.
        mode: ``auto`` (route by quant_method/rotation) or force a path with
            ``direct`` / ``mixed`` / ``fallback``.
        qtype: target type for the fallback (dequantize) path, e.g. ``Q4_K_M``.
        original_model: optional original FP model dir for skeleton metadata.
        work_dir: scratch directory.

    Returns:
        Summary dict including the chosen ``path`` and per-path details.
    """
    plan = plan_export(quantized_dir)
    chosen = mode if mode != "auto" else plan["path"]
    meta = plan["meta"]

    if chosen == "unsupported":
        raise ValueError(
            f"quant_method={meta.quant_method!r} is not supported for CPU/GGUF export. "
            "Supported: gptq, mixed_gptq, jointq, rtn, dbf, autobit (and rotated variants)."
        )

    logger.info(
        "export_to_gguf: %s -> %s | method=%s rotated=%s | path=%s (%s)",
        quantized_dir,
        out_gguf,
        meta.quant_method,
        meta.rotated,
        chosen,
        plan["reason"],
    )

    if chosen == "direct":
        from onecomp.cpu.export.direct import convert_gptq_to_gguf

        result = convert_gptq_to_gguf(
            quantized_dir, out_gguf, original_model=original_model, work_dir=work_dir
        )
    elif chosen == "mixed":
        from llamacpp_plugins.gptq import export_mixed_gptq_gguf

        result = export_mixed_gptq_gguf(quantized_dir, out_gguf, original_model=original_model)
        if not isinstance(result, dict):
            result = {"out_gguf": out_gguf, "detail": result}
    elif chosen == "fallback":
        from onecomp.cpu.export.fallback import export_via_dequantize

        export_via_dequantize(quantized_dir, out_gguf, qtype=qtype, work_dir=work_dir)
        result = {"out_gguf": out_gguf, "qtype": qtype}
    else:
        raise ValueError(f"Unknown export mode {mode!r} (expected auto/direct/mixed/fallback).")

    result["path"] = chosen
    result["quant_method"] = meta.quant_method
    result["rotated"] = meta.rotated
    return result
