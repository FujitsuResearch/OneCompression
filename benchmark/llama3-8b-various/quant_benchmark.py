"""Various Quantizers Benchmark

Run various quantizers with their default parameters (no QEP).
All quantizers share calibration data accumulation (X^T X) for efficiency.

Copyright 2025-2026 Fujitsu Ltd.

Usage:
    python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from onecomp import GPTQ, JointQ, CalibrationConfig, ModelConfig, Runner, ARB, CQ, DBF, QBB, QUIP, RTN, Onebit


def create_quantizers():
    """Create all quantizers with default parameters."""
    return [
        GPTQ(calc_quant_error=True),
        JointQ(calc_quant_error=True),
        DBF(calc_quant_error=True),
        QUIP(calc_quant_error=True),
        Onebit(calc_quant_error=True),
        RTN(calc_quant_error=True),
        CQ(calc_quant_error=True),
        ARB(calc_quant_error=True),
        QBB(calc_quant_error=True),
    ]


@hydra.main(version_base=None, config_path="conf", config_name="benchmark_llama3-8b")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    model_config = ModelConfig(path=cfg.model_path, device=cfg.model_device)

    quantizers = create_quantizers()

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
