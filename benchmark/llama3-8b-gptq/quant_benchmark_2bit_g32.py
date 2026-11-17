"""GPTQ 2bit-g32 Benchmark

Run GPTQ with bits=2, group_size=32 for the four configurations
(default / actorder / mse / mse+actorder) in a single pass.
Shares calibration data accumulation across quantizers for efficiency.
Results are saved under output_dir.

Copyright 2025-2026 Fujitsu Ltd.

Usage:
    python quant_benchmark_2bit_g32.py model_path=/path/to/Meta-Llama-3-8B
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from onecomp import GPTQ, CalibrationConfig, ModelConfig, Runner

ACCURACY_TASKS = [
    "arc_easy",
    "arc_challenge",
    "piqa",
    "winogrande",
    "hellaswag",
    "boolq",
]

# The four GPTQ configurations (2x2 grid of actorder x mse).
CONFIGS = [
    ("default", False, False),
    ("actorder", True, False),
    ("mse", False, True),
    ("mse_actorder", True, True),
]


def create_quantizers(cfg: DictConfig):
    """Create GPTQ quantizers for the four configurations at bits=2, group_size=32."""
    quantizers = []
    sym = cfg.gptq.symmetric
    sym_label = "sym" if sym else "asym"

    bits = cfg.gptq.bits
    gs = cfg.gptq.group_size

    # group_size=None means per-channel (groupsize=-1)
    gs_label = "pc" if gs is None else f"gs{gs}"
    gptq_groupsize = -1 if gs is None else gs

    for config_label, actorder, mse in CONFIGS:
        quantizers.append(
            GPTQ(
                num_layers=cfg.gptq.num_layers,
                wbits=bits,
                sym=sym,
                groupsize=gptq_groupsize,
                blocksize=cfg.gptq.blocksize,
                percdamp=cfg.gptq.percdamp,
                actorder=actorder,
                mse=mse,
                q_grid=cfg.gptq.q_grid,
                q_norm=cfg.gptq.q_norm,
                calc_quant_error=True,
                name=f"GPTQ_{bits}bit_{gs_label}_{sym_label}_{config_label}",
            )
        )

    return quantizers


@hydra.main(version_base=None, config_path="conf", config_name="benchmark_llama3-8b-2bit-g32")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    model_config = ModelConfig(path=cfg.model_path, device=cfg.model_device, dtype="bfloat16")

    quantizers = create_quantizers(cfg)

    print(f"Number of quantizers: {len(quantizers)}")
    for q in quantizers:
        print(f"  - {q.name}")

    # Build Runner
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

    # Run quantization
    runner.run()

    # Save results
    for q in quantizers:
        runner.save_quantization_statistics(f"quantization_statistics_{q.name}.json", quantizer=q)

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
            tasks=ACCURACY_TASKS,
            original_model=cfg.calc_original_acc,
            dequantized_model=True,
            quantized_model=False,
        )


if __name__ == "__main__":
    main()
