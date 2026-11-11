"""Run the JointQ 2-bit/group-size-32 benchmark configuration.

Usage:
    python quant_benchmark.py model_path=path/to/model output_dir=path/to/output_dir
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from onecomp import CalibrationConfig, JointQ, ModelConfig, Runner

ACCURACY_TASKS = [
    "arc_easy",
    "arc_challenge",
    "piqa",
    "winogrande",
    "hellaswag",
    "boolq",
]


def create_quantizers(cfg: DictConfig):
    """Create the selected JointQ_2bit_g32 configuration."""
    regularization_lambda_by_layer = {
        "self_attn.q_proj": 0.1,
        "self_attn.k_proj": 0.1,
        "self_attn.v_proj": 0.1,
        "self_attn.o_proj": 0.0,
        "mlp.gate_proj": 0.0,
        "mlp.up_proj": 0.0,
        "mlp.down_proj": 0.0,
    }

    return [
        JointQ(
            num_layers=cfg.jointq.num_layers,
            bits=2,
            symmetric=cfg.jointq.symmetric,
            group_size=32,
            regularization_lambda_by_layer=regularization_lambda_by_layer,
            regularization_retry_lambda=None,
            calc_quant_error=True,
            name="JointQ_2bit_g32",
        )
    ]


@hydra.main(version_base=None, config_path="conf", config_name="benchmark")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    quantizers = create_quantizers(cfg)
    runner = Runner(
        model_config=ModelConfig(
            path=cfg.model_path,
            device=cfg.model_device,
            dtype="bfloat16",
        ),
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

    for quantizer in quantizers:
        runner.save_quantization_statistics(
            f"quantization_statistics_{quantizer.name}.json",
            quantizer=quantizer,
        )

    if cfg.calc_ppl:
        runner.benchmark_perplexity(
            original_model=cfg.calc_original_ppl,
            dequantized_model=True,
            quantized_model=False,
        )

    if cfg.calc_acc:
        runner.benchmark_accuracy(
            tasks=ACCURACY_TASKS,
            original_model=cfg.calc_original_acc,
            dequantized_model=True,
            quantized_model=False,
        )


if __name__ == "__main__":
    main()
