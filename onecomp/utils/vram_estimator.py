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


@dataclass
class EnvironmentSnapshot:
    """Physical hardware readings at check-env time."""

    gpu_count: int
    gpu_name: str | None
    gpu_total_vram_gb: float | None
    gpu_free_vram_gb: float | None
    ram_total_gb: float | None
    ram_available_gb: float | None
    disk_available_gb: float | None
    disk_path: str


@dataclass
class ModelMemoryProfile:
    """Derived memory footprint for the target model."""

    total_params: int
    fp16_gb: float
    quantized_gb: dict
    calibration_overhead_gb: float


@dataclass
class EnvCheckResult:
    """Composite result returned by check_environment()."""

    model_id: str
    env: EnvironmentSnapshot
    model: ModelMemoryProfile
    estimation: VRAMBitwidthEstimation | None
    risk: str
    risk_detail: str


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


def check_environment(
    model_id: str,
    *,
    total_vram_gb: float | None = None,
    group_size: int = 128,
    save_dir: str | None = None,
    vram_ratio: float = 0.80,
    calibration_overhead_ratio: float = 0.15,
) -> EnvCheckResult:
    """Collect hardware info and estimate OOM risk before quantization.

    Loads the model architecture on a ``meta`` device (no GPU/CPU memory)
    to count parameters, then compares available VRAM against estimated
    memory requirements at 2/4/8-bit quantization.

    Args:
        model_id: Hugging Face model ID or local path.
        total_vram_gb: Override GPU VRAM in GB for estimation math only.
            Physical GPU readings are always from the real device.
        group_size: GPTQ group size for metadata calculation.
        save_dir: Path used for disk-space check. Defaults to cwd.
        vram_ratio: Fraction of VRAM allocated for the estimation budget.
        calibration_overhead_ratio: Calibration activation buffer as a
            fraction of the FP16 model footprint (default 15 %).

    Returns:
        :class:`EnvCheckResult` with hardware snapshot, memory profile,
        VRAM estimation, and risk level (``"safe"``, ``"warning"``,
        ``"danger"``, or ``"unknown"``).
    """
    import os
    import pathlib
    import shutil

    from transformers import AutoConfig, AutoModelForCausalLM

    # --- GPU snapshot --------------------------------------------------------
    gpu_count = torch.cuda.device_count()
    if gpu_count > 0:
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        gpu_name = props.name
        gpu_total_vram_gb = props.total_memory / _BYTES_PER_GB
        try:
            free_bytes, _ = torch.cuda.mem_get_info(dev)
            gpu_free_vram_gb = free_bytes / _BYTES_PER_GB
        except Exception:
            gpu_free_vram_gb = None
    else:
        gpu_name = None
        gpu_total_vram_gb = None
        gpu_free_vram_gb = None

    # --- CPU RAM (psutil optional) -------------------------------------------
    try:
        import psutil

        vm = psutil.virtual_memory()
        ram_total_gb = vm.total / _BYTES_PER_GB
        ram_available_gb = vm.available / _BYTES_PER_GB
    except ImportError:
        ram_total_gb = None
        ram_available_gb = None

    # --- Disk space (stdlib) -------------------------------------------------
    check_path = save_dir if save_dir else os.getcwd()
    p = pathlib.Path(check_path)
    while not p.exists():
        p = p.parent
    disk_available_gb = shutil.disk_usage(p).free / _BYTES_PER_GB

    # --- Model memory profile ------------------------------------------------
    config = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

    total_params = sum(p.numel() for p in model.parameters())
    fp16_gb = (total_params * 2) / _BYTES_PER_GB
    quantized_gb = {b: weight_memory_gb(total_params, b, group_size) for b in (2, 4, 8)}
    calibration_overhead_gb = fp16_gb * calibration_overhead_ratio

    # --- VRAM bitwidth estimation (reuse existing) ---------------------------
    try:
        estimation = estimate_target_bitwidth(
            model,
            vram_ratio=vram_ratio,
            total_vram_gb=total_vram_gb,
            group_size=group_size,
        )
    except (RuntimeError, ValueError):
        estimation = None

    # --- OOM risk assessment -------------------------------------------------
    # Use free VRAM (runtime reality) when available; fall back to override.
    effective_vram = gpu_free_vram_gb if gpu_free_vram_gb is not None else total_vram_gb

    if effective_vram is None:
        risk = "unknown"
        risk_detail = "No GPU detected and no --total-vram-gb provided."
    else:
        need_4bit = quantized_gb[4] + calibration_overhead_gb
        if effective_vram >= fp16_gb * 1.2:
            risk = "safe"
            risk_detail = (
                f"Free VRAM ({effective_vram:.1f} GB) comfortably fits "
                f"even FP16 weights ({fp16_gb:.1f} GB × 1.2)."
            )
        elif effective_vram >= need_4bit:
            risk = "warning"
            risk_detail = (
                f"Free VRAM ({effective_vram:.1f} GB) fits 4-bit quantized "
                f"weights but is tight (calibration overhead included)."
            )
        else:
            risk = "danger"
            risk_detail = (
                f"Free VRAM ({effective_vram:.1f} GB) is insufficient for "
                f"4-bit + calibration ({need_4bit:.1f} GB needed)."
            )

    return EnvCheckResult(
        model_id=model_id,
        env=EnvironmentSnapshot(
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            gpu_total_vram_gb=gpu_total_vram_gb,
            gpu_free_vram_gb=gpu_free_vram_gb,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            disk_available_gb=disk_available_gb,
            disk_path=str(p),
        ),
        model=ModelMemoryProfile(
            total_params=total_params,
            fp16_gb=fp16_gb,
            quantized_gb=quantized_gb,
            calibration_overhead_gb=calibration_overhead_gb,
        ),
        estimation=estimation,
        risk=risk,
        risk_detail=risk_detail,
    )


def print_env_report(result: EnvCheckResult, *, total_vram_gb_override: float | None = None) -> None:
    """Print a human-readable environment and OOM risk report to stdout.

    Args:
        result: The :class:`EnvCheckResult` from :func:`check_environment`.
        total_vram_gb_override: When not ``None``, annotates the VRAM budget
            line with ``[--total-vram-gb override]``.
    """
    _W = 60
    _SEP = "=" * _W
    _COL = 22

    def _row(label: str, value: str) -> str:
        return f"  {label:<{_COL}}: {value}"

    risk_labels = {
        "safe": "SAFE",
        "warning": "WARNING",
        "danger": "DANGER !!",
        "unknown": "UNKNOWN",
    }
    risk_label = risk_labels.get(result.risk, result.risk.upper())

    e = result.env
    m = result.model

    print(_SEP)
    print("  OneComp Environment Check")
    print(_SEP)
    print()

    # Hardware
    print("Hardware")
    print(_row("GPU count", str(e.gpu_count)))
    if e.gpu_name is not None:
        print(_row("GPU name", e.gpu_name))
    if e.gpu_total_vram_gb is not None:
        label = "GPU VRAM (total)"
        value = f"{e.gpu_total_vram_gb:.1f} GB"
        if total_vram_gb_override is not None:
            value += "  [physical]"
        print(_row(label, value))
    if total_vram_gb_override is not None:
        print(_row("VRAM budget used", f"{total_vram_gb_override:.1f} GB  [--total-vram-gb override]"))
    if e.gpu_free_vram_gb is not None:
        print(_row("GPU VRAM (free)", f"{e.gpu_free_vram_gb:.1f} GB"))
    if e.ram_total_gb is not None:
        print(_row("CPU RAM (total)", f"{e.ram_total_gb:.1f} GB"))
        print(_row("CPU RAM (avail)", f"{e.ram_available_gb:.1f} GB"))
    else:
        print(_row("CPU RAM", "n/a (install psutil for RAM info)"))
    print(_row("Disk (avail)", f"{e.disk_available_gb:.1f} GB  [{e.disk_path}]"))
    print()

    # Model
    print(f"Model: {result.model_id}")
    print(_row("Parameters", f"{m.total_params:,}"))
    print(_row("FP16 footprint", f"{m.fp16_gb:.2f} GB"))
    print()

    # Memory estimates
    gs = "(group_size varies)"
    print(f"Memory Estimates")
    for bits in (2, 4, 8):
        print(_row(f"{bits}-bit quantized", f"{m.quantized_gb[bits]:.2f} GB"))
    print(_row("Calib. overhead", f"{m.calibration_overhead_gb:.2f} GB  (15% of FP16)"))
    print(_row("4-bit + overhead", f"{m.quantized_gb[4] + m.calibration_overhead_gb:.2f} GB"))
    print()

    # OOM risk
    print("OOM Risk Assessment")
    print(_row("Risk level", risk_label))
    detail_words = result.risk_detail.split()
    detail_line = ""
    detail_lines = []
    for word in detail_words:
        if len(detail_line) + len(word) + 1 > 34:
            detail_lines.append(detail_line)
            detail_line = word
        else:
            detail_line = (detail_line + " " + word).lstrip()
    if detail_line:
        detail_lines.append(detail_line)
    for i, dl in enumerate(detail_lines):
        if i == 0:
            print(_row("Detail", dl))
        else:
            print(f"  {'':<{_COL}}  {dl}")
    print()
    if result.estimation is not None:
        print(_row("Recommended wbits", f"{result.estimation.target_bitwidth:.2f}  (VRAM-estimated)"))
    print(_SEP)
