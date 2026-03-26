"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Akihiro Yoshida

"""

from dataclasses import dataclass

import torch

_BYTES_PER_GB = 1e9


def effective_bits_per_param(
    wbits: int,
    group_size: int = 128,
    scale_bits: int = 16,
    zero_bits: int = None,
    in_features: int = None,
) -> float:
    """Actual bits per parameter including per-group scale and zero_point.

    In GPTQ's packed inference format (AutoGPTQ convention):

    - scale is stored as FP16 → ``scale_bits = 16``.
    - zero_point is packed at the same bit-width as the weights
      via ``pack_zeros(zero_int, wbits)`` in GPTQLinear, which stores
      ``32 // wbits`` values per INT32 element.  Actual memory per
      zero is ``wbits`` bits (not 32 despite INT32 container dtype)
      → ``zero_bits = wbits`` by default.

    When ``group_size <= 0`` and ``in_features`` is ``None``, the
    per-channel overhead is omitted (treated as 0).
    """
    if zero_bits is None:
        zero_bits = wbits
    meta = scale_bits + zero_bits
    if group_size > 0:
        return wbits + meta / group_size
    if in_features is not None and in_features > 0:
        return wbits + meta / in_features
    return float(wbits)


def raw_bits_for_quantizer(q):
    """Extract the raw (nominal) bit-width from a single quantizer.

    Looks up ``wbits``, ``bits``, or ``target_bits`` in that order.
    Returns ``None`` if no attribute is found.
    """
    for attr in ("wbits", "bits", "target_bits"):
        val = getattr(q, attr, None)
        if val is not None:
            return float(val)
    return None


def effective_bits_for_quantizer(q, in_features=None):
    """Effective bits per param for one quantizer, including scale/zero metadata.

    Extracts ``wbits`` and ``groupsize`` from the quantizer object and
    delegates to :func:`effective_bits_per_param`.

    """
    raw = raw_bits_for_quantizer(q)
    if raw is None:
        return 16.0

    gs = getattr(q, "groupsize", None)
    if gs is None:
        gs = getattr(q, "group_size", -1)

    return effective_bits_per_param(
        wbits=raw,
        group_size=gs if gs is not None else -1,
        in_features=in_features,
    )


def weight_memory_gb(
    num_params: int,
    wbits: int,
    group_size: int = 128,
    scale_bits: int = 16,
    zero_bits: int = 16,
) -> float:
    """Total memory (GB) for quantised weights including scale/zero metadata.

    Args:
        num_params: Number of weight parameters.
        wbits: Quantisation bit-width for the weights.
        group_size: Parameters per quantisation group (``-1`` = per-channel).
        scale_bits: Bits for the scale factor (16 = FP16).
        zero_bits: Bits for the zero-point (16 = FP16).
    """
    eff = effective_bits_per_param(wbits, group_size, scale_bits, zero_bits)
    return (num_params * eff / 8) / _BYTES_PER_GB


def _per_channel_meta(
    model: "torch.nn.Module",
    quantizable_ratio: float,
    scale_bits: int = 16,
    zero_bits: int = None,
    wbits: int = 4,
) -> float:
    """Weighted-average per-channel metadata overhead (bpw).

    For ``groupsize=-1``, each output row stores one scale (FP16) and
    one zero_point.  GPTQLinear packs zero_points via
    ``pack_zeros(zero_int, wbits)`` (``32 // wbits`` values per INT32),
    so actual memory per zero is ``wbits`` bits — not 32.  The overhead
    per weight element is ``(scale_bits + zero_bits) / in_features``,
    which varies across layers.  This function returns the
    parameter-weighted average across all ``nn.Linear`` layers.

    Args:
        wbits: Representative quantisation bit-width, used as the
            default for ``zero_bits`` (packed at ``wbits`` in the
            AutoGPTQ format).
    """
    if zero_bits is None:
        zero_bits = wbits
    meta_per_param = scale_bits + zero_bits
    total_meta_bits = 0
    total_weight_params = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            out_f, in_f = module.weight.shape
            total_meta_bits += out_f * meta_per_param
            total_weight_params += out_f * in_f
    if total_weight_params == 0:
        return 0.0
    quantizable_params = int(total_weight_params * quantizable_ratio)
    if quantizable_params == 0:
        return 0.0
    return total_meta_bits / quantizable_params


@dataclass
class VRAMBitwidthEstimation:
    """Result of VRAM-based bitwidth estimation."""

    target_bitwidth: float
    total_vram_gb: float
    budget_gb: float
    non_quant_weight_gb: float
    available_for_quant_gb: float
    total_params: int
    quantizable_params: int
    meta_bits_per_param: float


def estimate_target_bitwidth(
    model: torch.nn.Module,
    vram_ratio: float = 0.70,
    *,
    total_vram_gb: float = None,
    group_size: int = 128,
    scale_bits: int = 16,
    zero_bits: int = None,
    wbits: int = 4,
    quantizable_ratio: float = 0.95,
    logger=None,
) -> VRAMBitwidthEstimation:
    """Estimate the quantisation target bitwidth that fits in GPU VRAM.

    Reads total VRAM from ``torch.cuda`` (the same value shown by
    ``nvidia-smi``), multiplies by ``vram_ratio``, and solves for the
    largest bitwidth whose total memory (weights **+** scale/zero
    metadata) stays within that budget.

    In GPTQ's packed inference format (AutoGPTQ convention):

    - **scale** is stored as FP16 → ``scale_bits = 16``.
    - **zero_point** is packed at the weight bit-width
      → ``zero_bits = wbits`` by default.

    .. code-block:: text

        budget = total_vram × vram_ratio

        available = budget − FP16_non_quant_weights

        meta  = (scale_bits + zero_bits) / group_size   [grouped]
              = weighted_avg((scale_bits + zero_bits) / in_features) [per-channel]

        target_bit = available × 8 × 10⁹ / quantizable_params − meta

    Args:
        model: Model whose ``.parameters()`` to count.
        vram_ratio: Fraction of ``nvidia-smi`` total VRAM to allocate.
            For example, ``0.60`` means "use at most 60 %".
        total_vram_gb: Override GPU VRAM size in GB.  When ``None``
            (default), the value is read from ``torch.cuda``.  Useful
            for simulating resource-constrained environments, e.g.
            ``total_vram_gb=8.0`` to plan for an 8 GB card.
        group_size: Quantisation group size (for metadata calculation).
        scale_bits: Bits per scale factor (16 = FP16).
        zero_bits: Bits per zero-point.  Defaults to ``wbits``
            (packed at the weight bit-width in AutoGPTQ format).
        wbits: Representative quantisation bit-width, used as the
            default for ``zero_bits``.
        quantizable_ratio: Fraction of parameters that will be
            quantised (the rest stay in FP16).

    Returns:
        :class:`VRAMBitwidthEstimation` with the breakdown.

    Raises:
        RuntimeError: If no CUDA device is available and
            ``total_vram_gb`` is not provided.

    Examples::

        >>> result = estimate_target_bitwidth(model, vram_ratio=0.60)
        >>> print(f"{result.target_bitwidth:.2f} bits/param")
        >>> # Simulate 8 GB card
        >>> result = estimate_target_bitwidth(model, total_vram_gb=8.0)
    """
    if total_vram_gb is not None:
        if logger is not None:
            logger.info("Using user-specified VRAM: %.2f GB", total_vram_gb)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA device detected and total_vram_gb not specified. "
                "Pass total_vram_gb explicitly to simulate a target GPU."
            )
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        total_vram_gb = props.total_memory / _BYTES_PER_GB
        if logger is not None:
            logger.info("GPU: %s (%.2f GB)", props.name, total_vram_gb)

    budget_gb = total_vram_gb * vram_ratio

    total_params = sum(p.numel() for p in model.parameters())
    quantizable_params = int(total_params * quantizable_ratio)
    unquantizable_params = total_params - quantizable_params

    non_quant_gb = (unquantizable_params * 2) / _BYTES_PER_GB  # FP16
    available_gb = budget_gb - non_quant_gb

    if zero_bits is None:
        zero_bits = wbits
    if group_size > 0:
        meta_bits = (scale_bits + zero_bits) / group_size
    else:
        meta_bits = _per_channel_meta(
            model,
            quantizable_ratio,
            scale_bits,
            zero_bits,
            wbits,
        )

    if quantizable_params == 0 or available_gb <= 0:
        raise ValueError(
            f"Cannot fit model: budget={budget_gb:.2f} GB, non_quant={non_quant_gb:.2f} GB."
        )
    else:
        target = (available_gb * _BYTES_PER_GB * 8) / quantizable_params - meta_bits

    return VRAMBitwidthEstimation(
        target_bitwidth=target,
        total_vram_gb=total_vram_gb,
        budget_gb=budget_gb,
        non_quant_weight_gb=non_quant_gb,
        available_for_quant_gb=available_gb,
        total_params=total_params,
        quantizable_params=quantizable_params,
        meta_bits_per_param=meta_bits,
    )


def estimate_wbits_from_vram(
    model_id: str,
    vram_ratio: float = 0.8,
    *,
    total_vram_gb: float = None,
    group_size: int = 128,
    wbits: int = 4,
    logger=None,
) -> VRAMBitwidthEstimation:
    """Lightweight VRAM-based bitwidth estimation from a model identifier.

    Instantiates the model architecture on a ``meta`` device (no weight
    data, no GPU/CPU memory) to obtain accurate parameter counts, then
    delegates to :func:`estimate_target_bitwidth`.

    This is designed to be called **before** the full model is loaded,
    e.g. in :meth:`Runner.auto_run`, so that the estimated bitwidth
    can be used for output directory naming and passed directly to
    ``AutoBitQuantizer(target_bit=...)``.

    Args:
        model_id: Hugging Face model ID or local path.
        vram_ratio: Fraction of total VRAM to use (0.0–1.0).
        total_vram_gb: Override GPU VRAM in GB (reads from CUDA if ``None``).
        group_size: Quantisation group size for metadata calculation.
        wbits: Representative bit-width for zero-point metadata estimation.
        logger: Optional logger for diagnostics.

    Returns:
        :class:`VRAMBitwidthEstimation` — use ``result.target_bitwidth``
        as the raw bpw value (suitable for display and for passing as
        ``target_bit`` to ``AutoBitQuantizer``).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

    return estimate_target_bitwidth(
        model,
        vram_ratio=vram_ratio,
        total_vram_gb=total_vram_gb,
        group_size=group_size,
        wbits=wbits,
        logger=logger,
    )
