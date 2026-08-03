"""GGUF type-dispatch constants for the llama.cpp mixed-bit GPTQ plugin.

This is the llama.cpp counterpart of ``vllm_plugins/gptq/constants.py``.  Where
the vLLM plugin selects a *runtime kernel* per module (Marlin / Exllama), the
llama.cpp side selects a *GGUF tensor type* per module, because llama.cpp stores
and dispatches each tensor independently by its on-disk quantization type.

Two routing classes exist:

  * ``direct`` -- the GPTQ integer codes map losslessly onto a GGUF *legacy*
    block type (block size 32, one scale[/min] per block).  No re-quantization,
    so QEP/GPTQ error correction is preserved exactly.

  * ``kquant`` -- no lossless legacy block type exists for the bit-width (2/3-bit,
    and asymmetric 5/6-bit), so the layer's dequantized fp16 weights are
    re-quantized to a GGUF *K-quant* (super-block of 256) via ``llama-quantize``.
    This is lossy but keeps the layer at (roughly) the intended bit-width.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from typing import Tuple

from gguf.constants import GGMLQuantizationType as T

# (wbits, sym) -> GGUF legacy block type that losslessly represents GPTQ codes.
DIRECT_LOSSLESS_TYPES = {
    (4, True): T.Q4_0,  # value = d * (q - 8)
    (4, False): T.Q4_1,  # value = d * q + m
    (8, True): T.Q8_0,  # value = d * q8   (q8 = q - 128, int8)
}

# Bit-width -> GGUF K-quant used as a re-quantization fallback (lossy).
KQUANT_FALLBACK_TYPES = {
    2: T.Q2_K,
    3: T.Q3_K,
    5: T.Q5_K,
    6: T.Q6_K,
}

# Routing labels.
ROUTE_DIRECT = "direct"
ROUTE_KQUANT = "kquant"
ROUTE_DENSE = "dense"  # keep the layer as fp16 (always correct, just larger)


def select_gguf_route(bits: int, sym: bool) -> Tuple[str, T]:
    """Pick the GGUF type and routing class for one GPTQ module.

    Args:
        bits: GPTQ bit-width of the module.
        sym: Whether the module was quantized symmetrically.

    Returns:
        ``(route, ggml_type)`` where ``route`` is one of ``direct`` / ``kquant``
        / ``dense``.
    """
    direct = DIRECT_LOSSLESS_TYPES.get((bits, sym))
    if direct is not None:
        return ROUTE_DIRECT, direct
    kq = KQUANT_FALLBACK_TYPES.get(bits)
    if kq is not None:
        return ROUTE_KQUANT, kq
    return ROUTE_DENSE, T.F16
