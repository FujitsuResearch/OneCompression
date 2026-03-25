"""GPTQ Benchmark

Run GPTQ for all combinations of bits × group_size in a single pass.
Shares calibration data accumulation across quantizers for efficiency.
Results are saved under output_dir.

Copyright 2025-2026 Fujitsu Ltd.

Usage:
    python quant_benchmark.py
"""

import itertools

import hydra
from omegaconf import DictConfig, OmegaConf

from onecomp import GPTQ, ModelConfig, Runner


def create_quantizers(cfg: DictConfig):
    """Create a list of GPTQ quantizers for all combinations of bits × group_size."""
    quantizers = []
    sym = cfg.gptq.symmetric
    sym_label = "sym" if sym else "asym"

    for bits, gs in itertools.product(cfg.gptq.bits, cfg.gptq.group_size):
        # Label strings
        gs_label = "pc" if gs is None else f"gs{gs}"

        # GPTQ: groupsize=-1 means per-channel
        gptq_groupsize = -1 if gs is None else gs
        quantizers.append(
            GPTQ(
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

    # Build Runner
    runner = Runner(
        model_config=model_config,
        quantizers=quantizers,
        max_length=cfg.max_length,
        num_calibration_samples=cfg.num_calibration_samples,
        calibration_strategy=cfg.calibration_strategy,
        calibration_seed=cfg.calibration_seed,
        calibration_batch_size=cfg.calibration_batch_size,
    )

    # Run quantization
    runner.run()

    # Save results
    for q in quantizers:
        runner.save_quantization_statistics(
            f"quantization_statistics_{q.name}.json", quantizer=q
        )

    # Perplexity evaluation
    if cfg.calc_ppl:
        runner.benchmark_perplexity(original_model=cfg.calc_original_ppl)

    # Accuracy evaluation
    if cfg.calc_acc:
        runner.benchmark_accuracy(original_model=cfg.calc_original_acc)


if __name__ == "__main__":
    main()
