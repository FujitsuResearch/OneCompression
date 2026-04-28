"""QEP + GPTQ Benchmark

Run GPTQ with QEP for a single bits × group_size combination
selected by ``task_id``.  Designed to be launched as a SLURM array
job (--array=0-3) where each task handles one combination.

Task mapping (bits × group_size, product order):
    0: 4-bit, gs128
    1: 4-bit, per-channel
    2: 3-bit, gs128
    3: 3-bit, per-channel

Copyright 2025-2026 Fujitsu Ltd.

Usage:
    python quant_benchmark.py task_id=0
"""

import itertools

import hydra
from omegaconf import DictConfig, OmegaConf

from onecomp import CalibrationConfig, GPTQ, ModelConfig, QEPConfig, Runner


def create_quantizer(cfg: DictConfig, task_id: int):
    """Create a single GPTQ quantizer for the given task_id."""
    combinations = list(itertools.product(cfg.gptq.bits, cfg.gptq.group_size))

    if task_id < 0 or task_id >= len(combinations):
        raise ValueError(
            f"task_id={task_id} is out of range "
            f"(0..{len(combinations) - 1} for {len(combinations)} combinations)"
        )

    bits, gs = combinations[task_id]
    sym = cfg.gptq.symmetric
    sym_label = "sym" if sym else "asym"
    gs_label = "pc" if gs is None else f"gs{gs}"
    gptq_groupsize = -1 if gs is None else gs

    return GPTQ(
        num_layers=cfg.gptq.num_layers,
        wbits=bits,
        sym=sym,
        groupsize=gptq_groupsize,
        blocksize=cfg.gptq.blocksize,
        percdamp=cfg.gptq.percdamp,
        actorder=cfg.gptq.actorder,
        mse=cfg.gptq.mse,
        q_grid=cfg.gptq.q_grid,
        q_norm=cfg.gptq.q_norm,
        calc_quant_error=True,
        name=f"GPTQ_{bits}bit_{gs_label}_{sym_label}",
    )


@hydra.main(version_base=None, config_path="conf", config_name="benchmark_llama3-8b")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    quantizer = create_quantizer(cfg, cfg.task_id)

    print(f"task_id={cfg.task_id} -> {quantizer.name}")

    model_config = ModelConfig(path=cfg.model_path, device=cfg.model_device)

    qep_config = QEPConfig(
        percdamp=cfg.qep.percdamp,
        perccorr=cfg.qep.perccorr,
    )

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=CalibrationConfig(
            max_length=cfg.max_length,
            num_calibration_samples=cfg.num_calibration_samples,
            strategy=cfg.calibration_strategy,
            seed=cfg.calibration_seed,
        ),
        qep=True,
        qep_config=qep_config,
    )

    runner.run()

    runner.save_quantization_statistics(
        f"quantization_statistics_{quantizer.name}.json"
    )

    # Perplexity evaluation
    if cfg.calc_ppl:
        runner.benchmark_perplexity(
            original_model=cfg.calc_original_ppl,
            dequantized_model=True,
            quantized_model=False,
        )

    # Accuracy evaluation
    if cfg.calc_acc:
        runner.benchmark_accuracy(
            original_model=cfg.calc_original_acc,
            dequantized_model=True,
            quantized_model=False,
        )


if __name__ == "__main__":
    main()
