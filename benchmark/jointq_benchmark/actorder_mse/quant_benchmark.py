"""Run the JointQ activation-ordering and GPTQ MSE ablation.

Usage:
    python quant_benchmark.py model_path=path/to/model output_dir=path/to/output_dir
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from onecomp import GPTQ, CalibrationConfig, JointQ, ModelConfig, Runner

ACCURACY_TASKS = [
    "arc_easy",
    "arc_challenge",
    "piqa",
    "winogrande",
    "hellaswag",
    "boolq",
]


def create_quantizers(cfg: DictConfig):
    """Create the selected 4-bit and 3-bit actorder/MSE configurations."""
    settings = [
        (
            "JointQ_4bit_g128_actorder_mse",
            4,
            [0.0, 0.0, 0.01, 0.0, 0.01, 0.01, 0.0],
        ),
        (
            "JointQ_3bit_g128_actorder_mse",
            3,
            [0.01, 0.01, 0.01, 0.0, 0.01, 0.01, 0.0],
        ),
    ]
    layer_names = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]

    return [
        JointQ(
            num_layers=cfg.jointq.num_layers,
            bits=bits,
            symmetric=cfg.jointq.symmetric,
            group_size=128,
            regularization_lambda_by_layer=dict(zip(layer_names, lambdas)),
            regularization_retry_lambda=0.01,
            regularization_retry_threshold=0.2,
            actorder=True,
            gptq=GPTQ(
                wbits=bits,
                groupsize=128,
                sym=cfg.jointq.symmetric,
                mse=True,
            ),
            enable_group_swap=False,
            calc_quant_error=True,
            name=name,
        )
        for name, bits, lambdas in settings
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
