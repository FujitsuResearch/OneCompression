"""Fallback GGUF export via dequantization + llama-quantize.

This path works for any OneComp GPTQ checkpoint (including 2/3-bit, actorder,
and mixed bitwidths) but re-quantizes the weights, so it does not preserve the
GPTQ/QEP error correction. Prefer ``onecomp.cpu.export.direct.convert_gptq_to_gguf``
when its constraints are met.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from __future__ import annotations

import os
import tempfile
from logging import getLogger
from typing import Optional

from onecomp.cpu.export.dequantize import dequantize_to_hf
from onecomp.cpu.llama_tooling import run_convert_hf_to_gguf, run_llama_quantize

logger = getLogger(__name__)


def export_via_dequantize(
    quantized_dir: str,
    out_gguf: str,
    qtype: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> str:
    """Dequantize -> f16 GGUF, then optionally quantize to ``qtype`` (e.g. Q4_K_M).

    Args:
        quantized_dir: OneComp GPTQ checkpoint.
        out_gguf: Output GGUF path.
        qtype: If given, run llama-quantize to this type; otherwise keep f16.
        work_dir: Scratch dir.

    Returns:
        ``out_gguf``.
    """
    owns_workdir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="onecomp_gguf_dq_")
    os.makedirs(work_dir, exist_ok=True)
    try:
        dense_dir = os.path.join(work_dir, "dense_fp16")
        dequantize_to_hf(quantized_dir, dense_dir)

        if qtype is None:
            run_convert_hf_to_gguf(dense_dir, out_gguf, outtype="f16")
            return out_gguf

        f16_gguf = os.path.join(work_dir, "model.f16.gguf")
        run_convert_hf_to_gguf(dense_dir, f16_gguf, outtype="f16")
        run_llama_quantize(f16_gguf, out_gguf, qtype)
        return out_gguf
    finally:
        if owns_workdir:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)
