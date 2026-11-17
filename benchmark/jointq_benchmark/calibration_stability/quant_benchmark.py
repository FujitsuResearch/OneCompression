"""Benchmark quantization stability with small calibration datasets.

Usage:
    python quant_benchmark.py model_path=path/to/model output_dir=path/to/output_dir
"""

import json
from pathlib import Path

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

ACCURACY_METRICS = {
    "arc_easy": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
    "hellaswag": "acc_norm,none",
    "boolq": "acc,none",
}

REGULARIZED_LAMBDAS = {
    3: [0.01, 0.01, 0.01, 0.0, 0.01, 0.01, 0.0],
    4: [0.0, 0.0, 0.01, 0.0, 0.01, 0.01, 0.0],
}


def create_quantizers(cfg: DictConfig):
    """Create the five group-size-128 benchmark configurations."""
    bits = int(cfg.bits)
    if bits not in REGULARIZED_LAMBDAS:
        raise ValueError(
            f"Unsupported bits: {bits}. Expected one of {sorted(REGULARIZED_LAMBDAS)}"
        )

    layer_names = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    regularized_lambdas = REGULARIZED_LAMBDAS[bits]
    unregularized_lambdas = [0.0] * len(layer_names)

    quantizers = [
        JointQ(
            num_layers=cfg.jointq.num_layers,
            bits=bits,
            symmetric=cfg.jointq.symmetric,
            group_size=128,
            regularization_lambda_by_layer=dict(zip(layer_names, regularized_lambdas)),
            regularization_retry_lambda=0.01,
            regularization_retry_threshold=0.2,
            calc_quant_error=True,
            name=f"JointQ_{bits}bit_g128",
        ),
        JointQ(
            num_layers=cfg.jointq.num_layers,
            bits=bits,
            symmetric=cfg.jointq.symmetric,
            group_size=128,
            regularization_lambda_by_layer=dict(zip(layer_names, unregularized_lambdas)),
            regularization_retry_lambda=None,
            calc_quant_error=True,
            name=f"JointQ_{bits}bit_g128_unregularized",
        ),
        JointQ(
            num_layers=cfg.jointq.num_layers,
            bits=bits,
            symmetric=cfg.jointq.symmetric,
            group_size=128,
            regularization_lambda_by_layer=dict(zip(layer_names, regularized_lambdas)),
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
            name=f"JointQ_{bits}bit_g128_actorder_mse",
        ),
    ]

    for name, mse, actorder in [
        (f"GPTQ_{bits}bit_g128_default", False, False),
        (f"GPTQ_{bits}bit_g128_mse_actorder", True, True),
    ]:
        quantizer = GPTQ(
            num_layers=cfg.gptq.num_layers,
            wbits=bits,
            sym=cfg.gptq.symmetric,
            groupsize=128,
            blocksize=cfg.gptq.blocksize,
            percdamp=cfg.gptq.percdamp,
            actorder=actorder,
            mse=mse,
            q_grid=cfg.gptq.q_grid,
            q_norm=cfg.gptq.q_norm,
            calc_quant_error=True,
            name=name,
        )
        quantizer.name = name
        quantizers.append(quantizer)

    return quantizers


def calculate_mean_accuracy(task_results: dict) -> float:
    """Calculate the arithmetic mean using the benchmark's task metrics."""
    values = [task_results[task][metric] for task, metric in ACCURACY_METRICS.items()]
    return float(sum(values) / len(values))


def build_results(quantizers, ppl_results: dict, accuracy_results: dict) -> dict:
    """Build the per-quantizer PPL and mean accuracy result mapping."""
    results = {}
    for quantizer in quantizers:
        result_key = f"{quantizer.name}_dequantized"
        results[quantizer.name] = {
            "ppl": float(ppl_results[result_key]),
            "mean_accuracy": calculate_mean_accuracy(accuracy_results[result_key]),
        }
    return results


def result_filename(num_calibration_samples: int, calibration_seed: int) -> str:
    """Return the benchmark result filename for one calibration run."""
    return f"numsamples{num_calibration_samples}_seed{calibration_seed}.json"


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
            shuffle_documents=cfg.shuffle_documents,
            batch_size=cfg.calibration_batch_size,
        ),
    )

    runner.run()

    for quantizer in quantizers:
        runner.save_quantization_statistics(
            f"quantization_statistics_{quantizer.name}.json",
            quantizer=quantizer,
        )

    ppl_results = runner.benchmark_perplexity(
        original_model=False,
        dequantized_model=True,
        quantized_model=False,
    )
    accuracy_results = runner.benchmark_accuracy(
        tasks=ACCURACY_TASKS,
        original_model=False,
        dequantized_model=True,
        quantized_model=False,
    )

    result = {
        "bits": int(cfg.bits),
        "max_length": int(cfg.max_length),
        "num_calibration_samples": int(cfg.num_calibration_samples),
        "calibration_seed": int(cfg.calibration_seed),
        "shuffle_documents": bool(cfg.shuffle_documents),
        "results": build_results(quantizers, ppl_results, accuracy_results),
    }
    output_path = Path(result_filename(cfg.num_calibration_samples, cfg.calibration_seed))
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
