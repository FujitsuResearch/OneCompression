"""JointQ Benchmark

Run JointQ for all combinations of bits x group_size in a single pass.
Shares calibration data accumulation across quantizers for efficiency.
Results are saved under output_dir.

Copyright 2025-2026 Fujitsu Ltd.

Usage:
    python quant_benchmark.py
"""

import itertools

import hydra
from omegaconf import DictConfig, OmegaConf

from onecomp import CalibrationConfig, GPTQ, JointQ, ModelConfig, Runner


def create_quantizers(cfg: DictConfig):
    """Create a list of JointQ quantizers for all combinations of bits x group_size.

    All lambda/regularization parameters are passed explicitly so that the
    benchmark remains reproducible regardless of future JointQ default changes.
    """
    quantizers = []
    sym = cfg.jointq.symmetric
    sym_label = "sym" if sym else "asym"

    lambda_mode = cfg.jointq.lambda_mode
    regularization_lambda = cfg.jointq.regularization_lambda
    regularization_mode = cfg.jointq.regularization_mode
    regularization_gamma = cfg.jointq.regularization_gamma
    lambda_list = list(cfg.jointq.lambda_list)

    for bits, gs in itertools.product(cfg.jointq.bits, cfg.jointq.group_size):
        gs_label = "pc" if gs is None else f"gs{gs}"
        jointq_groupsize = None if gs is None else gs

        gptq = None
        if cfg.jointq.gptq_mse:
            gptq_groupsize = -1 if gs is None else gs
            gptq = GPTQ(wbits=bits, groupsize=gptq_groupsize, sym=sym, mse=True)

        quantizers.append(
            JointQ(
                num_layers=cfg.jointq.num_layers,
                bits=bits,
                symmetric=sym,
                group_size=jointq_groupsize,
                log_level=cfg.jointq.log_level,
                device=cfg.jointq.device,
                lambda_mode=lambda_mode,
                regularization_lambda=regularization_lambda,
                regularization_mode=regularization_mode,
                regularization_gamma=regularization_gamma,
                lambda_list=lambda_list,
                actorder=cfg.jointq.actorder,
                gptq=gptq,
                calc_quant_error=True,
                name=f"JointQ_{bits}bit_{gs_label}_{sym_label}",
            )
        )

    return quantizers


@hydra.main(version_base=None, config_path="conf", config_name="benchmark_llama3-8b")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    model_config = ModelConfig(path=cfg.model_path, device=cfg.model_device)

    quantizers = create_quantizers(cfg)

    print(f"Number of quantizers: {len(quantizers)}")
    for q in quantizers:
        print(f"  - {q.name}")

    runner = Runner(
        model_config=model_config,
        quantizers=quantizers,
        calibration_config=CalibrationConfig(
            max_length=cfg.max_length,
            num_calibration_samples=cfg.num_calibration_samples,
            strategy=cfg.calibration_strategy,
            seed=cfg.calibration_seed,
            batch_size=cfg.calibration_batch_size,
        ),
    )

    runner.run()

    for q in quantizers:
        runner.save_quantization_statistics(
            f"quantization_statistics_{q.name}.json", quantizer=q
        )

    if cfg.calc_ppl:
        runner.benchmark_perplexity(
            original_model=cfg.calc_original_ppl,
            dequantized_model=True,
            quantized_model=False,
        )

    if cfg.calc_acc:
        runner.benchmark_accuracy(
            original_model=cfg.calc_original_acc,
            dequantized_model=True,
            quantized_model=False,
        )


if __name__ == "__main__":
    main()
