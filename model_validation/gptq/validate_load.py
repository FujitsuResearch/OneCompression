"""

Model validation: load a saved GPTQ-quantized model and verify it runs

Hydra entry point that loads a quantized model directory produced by
``validate_gptq.py`` (``runner.save_quantized_model``) via
``load_quantized_model`` and runs a short greedy generation to confirm
the load + inference path works end-to-end without errors.

Copyright 2025-2026 Fujitsu Ltd.

Usage:
    python validate_load.py quantized_path=/path/to/saved/quantized

"""

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from onecomp import load_quantized_model, setup_logger


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@hydra.main(version_base=None, config_path="conf", config_name="validate_load")
def main(cfg: DictConfig):
    setup_logger()
    print(OmegaConf.to_yaml(cfg))

    if cfg.quantized_path is None:
        raise ValueError(
            "quantized_path must be provided "
            "(e.g. quantized_path=/path/to/saved/quantized)"
        )

    if cfg.torch_dtype is None:
        torch_dtype = None
    elif cfg.torch_dtype not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown torch_dtype: {cfg.torch_dtype} "
            f"(supported: {list(_DTYPE_MAP)})"
        )
    else:
        torch_dtype = _DTYPE_MAP[cfg.torch_dtype]

    model, tokenizer = load_quantized_model(
        cfg.quantized_path, torch_dtype=torch_dtype
    )
    print(f"Loaded model type  : {type(model).__name__}")
    print(f"Loaded model device: {next(model.parameters()).device}")

    inputs = tokenizer(cfg.prompt, return_tensors="pt").to(
        next(model.parameters()).device
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nPrompt   : {cfg.prompt}")
    print(f"Generated: {generated}")


if __name__ == "__main__":
    main()
