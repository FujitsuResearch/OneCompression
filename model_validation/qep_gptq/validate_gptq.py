"""

Model validation: GPTQ quantization (wbits=4, groupsize=128, qep=True)

Hydra entry point for validating OneComp's GPTQ quantizer with QEP across
multiple models. The model is selected via either ``model_id`` (Hugging
Face Hub) or ``model_path`` (local). Exactly one of the two must be
provided; otherwise the script exits with ``ValueError``.

Copyright 2025-2026 Fujitsu Ltd.

Usage:
    python validate_gptq.py model_path=/path/to/model
    python validate_gptq.py model_id=TinyLlama/TinyLlama-1.1B-...

"""

import hydra
from omegaconf import DictConfig, OmegaConf

from onecomp import (
    CalibrationConfig,
    GPTQ,
    ModelConfig,
    Runner,
    setup_logger,
)


@hydra.main(version_base=None, config_path="conf", config_name="validate")
def main(cfg: DictConfig):
    setup_logger()
    print(OmegaConf.to_yaml(cfg))

    if cfg.model_id is None and cfg.model_path is None:
        raise ValueError(
            "Either model_id or model_path must be provided "
            "(e.g. model_path=/path/to/model)"
        )
    if cfg.model_id is not None and cfg.model_path is not None:
        raise ValueError("Specify only one of model_id or model_path")

    quantizer = GPTQ(wbits=4, groupsize=128)

    runner = Runner(
        model_config=ModelConfig(
            model_id=cfg.model_id,
            path=cfg.model_path,
            device="cuda:0",
        ),
        quantizer=quantizer,
        calibration_config=CalibrationConfig(
            max_length=1024, num_calibration_samples=128
        ),
        qep=True,
    )

    runner.run()
    runner.save_quantized_model("./quantized")

    original_ppl, _, quantized_ppl = runner.calculate_perplexity(
        original_model=True, dequantized_model=False, quantized_model=True
    )

    print(f"Original model perplexity: {original_ppl}")
    print(f"Quantized model perplexity: {quantized_ppl}")


if __name__ == "__main__":
    main()
