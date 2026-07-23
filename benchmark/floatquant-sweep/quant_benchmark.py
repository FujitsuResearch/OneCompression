"""FloatQuant scale-sweep benchmark grid.

Runs direct, static-sweep, in-loop-sweep, and conditional-Hessian
FloatQuant configurations from one Hydra config.  QEP runs are executed
one configuration at a time because Runner does not support
``quantizers`` mode together with QEP.
"""

import copy
import json
import random
import statistics
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from onecomp import CalibrationConfig, ModelConfig, QEPConfig, Runner
from onecomp.quantizer.floatquant import FloatQuant


def _is_qep_compatible(mode: DictConfig) -> bool:
    """Only Hessian/GPTQ-style modes are meaningful under QEP."""
    return bool(mode.use_hessian)


def _make_quantizer(fmt: str, mode: DictConfig, cfg: DictConfig, qep_enabled: bool):
    """Create one named FloatQuant instance from the benchmark config."""
    qep_label = "qep" if qep_enabled else "noqep"
    candidate_strategy = str(
        mode.get("scale_candidate_strategy", cfg.floatquant.scale_candidate_strategy)
    )
    return FloatQuant(
        fmt=fmt,
        block_size=cfg.floatquant.block_size,
        use_hessian=bool(mode.use_hessian),
        scale_timing=str(mode.scale_timing),
        scale_objective=str(mode.scale_objective),
        scale_candidate_strategy=candidate_strategy,
        blocksize=int(cfg.floatquant.blocksize),
        percdamp=float(cfg.floatquant.percdamp),
        num_layers=cfg.floatquant.num_layers,
        calc_quant_error=bool(cfg.floatquant.calc_quant_error),
        name=f"FloatQuant_{fmt}_{mode.name}_{candidate_strategy}_{qep_label}",
    )


def _calibration_config(cfg: DictConfig, qep_enabled: bool):
    """Build calibration config, respecting Runner's QEP constraints."""
    batch_size = None if qep_enabled else cfg.calibration.batch_size
    return CalibrationConfig(
        max_length=cfg.calibration.max_length,
        num_calibration_samples=cfg.calibration.num_calibration_samples,
        strategy=cfg.calibration.strategy,
        seed=cfg.calibration.seed,
        batch_size=batch_size,
    )


def _qep_config(cfg: DictConfig):
    """Build QEPConfig from Hydra config."""
    return QEPConfig(
        general=bool(cfg.qep.general),
        percdamp=float(cfg.qep.percdamp),
        perccorr=float(cfg.qep.perccorr),
        device=str(cfg.qep.device),
        exclude_layer_keywords=list(cfg.qep.exclude_layer_keywords),
    )


def _cuda_synchronize_if_requested(cfg: DictConfig):
    """Synchronize CUDA kernels when timing if requested and available."""
    if bool(cfg.timing.synchronize_cuda) and torch.cuda.is_available():
        torch.cuda.synchronize()


def _summarize_seconds(values: list[float]) -> dict:
    """Return robust summary statistics for timing repeats."""
    if not values:
        return {}
    return {
        "values": values,
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _run_runner(model_config, quantizer, cfg: DictConfig, qep_enabled: bool):
    """Run one quantizer and return a compact result record."""
    quantizer_template = copy.deepcopy(quantizer)
    warmups = int(cfg.timing.warmup_runs)
    repeats = int(cfg.timing.repeats)
    if repeats < 1:
        raise ValueError("timing.repeats must be >= 1.")

    runner = None
    elapsed_values: list[float] = []
    for repeat_idx in range(warmups + repeats):
        is_warmup = repeat_idx < warmups
        quantizer = copy.deepcopy(quantizer_template)
        runner = Runner(
            model_config=model_config,
            quantizer=quantizer,
            calibration_config=_calibration_config(cfg, qep_enabled),
            qep=qep_enabled,
            qep_config=_qep_config(cfg) if qep_enabled else None,
        )

        _cuda_synchronize_if_requested(cfg)
        started = time.perf_counter()
        runner.run()
        _cuda_synchronize_if_requested(cfg)
        elapsed = time.perf_counter() - started
        if not is_warmup:
            elapsed_values.append(elapsed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runner.save_quantization_statistics(
        str(output_dir / f"quantization_statistics_{quantizer.name}.json"),
        quantizer=quantizer,
    )

    ppl = {}
    if cfg.evaluation.calc_ppl:
        for dataset in cfg.evaluation.ppl_datasets:
            ppl[dataset.name] = runner.benchmark_perplexity(
                original_model=bool(cfg.evaluation.calc_original_ppl),
                dequantized_model=bool(cfg.evaluation.dequantized_model),
                quantized_model=bool(cfg.evaluation.quantized_model),
                dataset_name=dataset.dataset_name,
                dataset_config=dataset.dataset_config,
                split=dataset.split,
                max_samples=dataset.max_samples,
                max_length=dataset.max_length,
                stride=dataset.stride,
            )

    return {
        "name": quantizer.name,
        "fmt": quantizer.fmt,
        "use_hessian": quantizer.use_hessian,
        "scale_timing": quantizer.scale_timing,
        "scale_objective": quantizer.scale_objective,
        "scale_candidate_strategy": quantizer.scale_candidate_strategy,
        "qep": qep_enabled,
        "elapsed_seconds": statistics.median(elapsed_values),
        "timing": {
            "warmup_runs": warmups,
            "repeats": repeats,
            "synchronize_cuda": bool(cfg.timing.synchronize_cuda),
            "quantization_elapsed_seconds": _summarize_seconds(elapsed_values),
        },
        "ppl": ppl,
    }


@hydra.main(version_base=None, config_path="conf", config_name="benchmark_floatquant")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if cfg.model_path is None:
        raise ValueError("model_path must be set to a Hugging Face model path or id.")

    model_config = ModelConfig(path=cfg.model_path, device=cfg.model_device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_filter = set(cfg.filters.modes) if cfg.filters.modes is not None else None
    format_filter = set(cfg.filters.formats) if cfg.filters.formats is not None else None

    run_specs = []
    for qep_enabled in cfg.qep.enabled:
        for fmt in cfg.formats:
            if format_filter is not None and fmt not in format_filter:
                continue
            for mode in cfg.modes:
                if mode_filter is not None and mode.name not in mode_filter:
                    continue
                if qep_enabled and not _is_qep_compatible(mode):
                    continue
                run_specs.append((bool(qep_enabled), fmt, mode))
    if bool(cfg.timing.randomize_order):
        random.Random(int(cfg.timing.random_seed)).shuffle(run_specs)

    records = []
    for qep_enabled, fmt, mode in run_specs:
        quantizer = _make_quantizer(fmt, mode, cfg, qep_enabled)
        print(f"Running {quantizer.name}")
        records.append(_run_runner(model_config, quantizer, cfg, qep_enabled))
        with open(output_dir / "floatquant_sweep_results.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
