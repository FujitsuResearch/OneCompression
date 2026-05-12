"""LPCD + GPTQ Benchmark

Run GPTQ with LPCD for a single bits × submodule combination
selected by ``task_id``.  Designed to be launched as a SLURM array
job (--array=0-11) where each task handles one combination.

Task mapping (bits × submodule, product order):
    0: 4-bit, QEP
    1: 4-bit, LPCD(residual)
    2: 4-bit, LPCD(Key/Query)
    3: 4-bit, LPCD(Value/Output)
    4: 4-bit, LPCD(Up/Down)
    5: 4-bit, LPCD(All)
    6: 3-bit, QEP
    7: 3-bit, LPCD(residual)
    8: 3-bit, LPCD(Key/Query)
    9: 3-bit, LPCD(Value/Output)
    10: 3-bit, LPCD(Up/Down)
    11: 3-bit, LPCD(All)

Usage:
    python quant_benchmark.py task_id=0
"""

import itertools

import hydra
from omegaconf import DictConfig, OmegaConf

from onecomp import GPTQ, ModelConfig, QEPConfig, LPCDConfig, Runner


def create_quantizer(cfg: DictConfig, task_id: int):
    """Create a single GPTQ quantizer for the given task_id."""
    combinations = list(itertools.product(cfg.gptq.bits, ['none', 'res', 'qk', 'vo', 'ud', 'all']))

    if task_id < 0 or task_id >= len(combinations):
        raise ValueError(
            f"task_id={task_id} is out of range "
            f"(0..{len(combinations) - 1} for {len(combinations)} combinations)"
        )

    bits, submodule = combinations[task_id]

    return GPTQ(
        wbits=bits, 
        groupsize=128,
        name=f"lpcd_{submodule}_{bits}bit",
    )


@hydra.main(version_base=None, config_path="conf", config_name="benchmark_llama3-8b")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    quantizer = create_quantizer(cfg, cfg.task_id)

    print(f"task_id={cfg.task_id} -> {quantizer.name}")

    model_config = ModelConfig(
        path=cfg.model_path, 
        dtype="bfloat16",
        device=cfg.model_device
    )

    qep_config = QEPConfig(
        percdamp=cfg.qep.percdamp,
        perccorr=cfg.qep.perccorr,
    )

    is_res = not quantizer.name.startswith("lpcd_none")
    is_qk = quantizer.name.startswith("lpcd_qk")
    is_vo = quantizer.name.startswith("lpcd_vo")
    is_ud = quantizer.name.startswith("lpcd_ud")
    is_all = quantizer.name.startswith("lpcd_all")

    lpcd_config = LPCDConfig(
        enable_residual=is_res,
        enable_qk=is_qk or is_all,
        enable_vo=is_vo or is_all,
        enable_ud=is_ud or is_all,
        alt_steps=cfg.lpcd.alt_steps,
        percdamp=cfg.lpcd.percdamp,
        perccorr=cfg.lpcd.perccorr,
        use_closed_form=cfg.lpcd.use_closed_form,
        gd_steps=cfg.lpcd.gd_steps,
        gd_batch_size=cfg.lpcd.gd_batch_size,
        gd_base_lr=cfg.lpcd.gd_base_lr,
    )

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        qep=True,
        qep_config=qep_config,
        lpcd=True,
        lpcd_config=lpcd_config,
        max_length=cfg.max_length,
        num_calibration_samples=cfg.num_calibration_samples,
        calibration_strategy=cfg.calibration_strategy,
        calibration_seed=cfg.calibration_seed,
    )

    runner.run()

    runner.save_quantization_statistics(
        f"quantization_statistics_{quantizer.name}.json"
    )

    # Perplexity evaluation
    if cfg.calc_ppl:
        runner.benchmark_perplexity(original_model=cfg.calc_original_ppl)

    # Accuracy evaluation
    if cfg.calc_acc:
        runner.benchmark_accuracy(original_model=cfg.calc_original_acc)


if __name__ == "__main__":
    main()
