"""

Model validation: load a saved GPTQ-quantized model with vLLM and verify it runs

Hydra entry point that loads a quantized model directory produced by
``validate_gptq.py`` (``runner.save_quantized_model``) via vLLM's
offline ``LLM`` interface and runs a short greedy generation to confirm
the load + inference path through vLLM works end-to-end without errors.

vLLM spawns worker subprocesses that re-import this script. ``LLM(...)``
and ``llm.generate(...)`` must therefore stay inside ``main()``, behind
the ``if __name__ == "__main__":`` guard, so that re-imports do not
recursively spawn new engines.

Copyright 2025-2026 Fujitsu Ltd.

Usage:
    python validate_vllm.py quantized_path=/path/to/saved/quantized

"""

import hydra
from omegaconf import DictConfig, OmegaConf

from onecomp import setup_logger
from vllm import LLM, SamplingParams


@hydra.main(version_base=None, config_path="conf", config_name="validate_vllm")
def main(cfg: DictConfig):
    setup_logger()
    print(OmegaConf.to_yaml(cfg))

    if cfg.quantized_path is None:
        raise ValueError(
            "quantized_path must be provided "
            "(e.g. quantized_path=/path/to/saved/quantized)"
        )

    llm = LLM(
        model=cfg.quantized_path,
        max_model_len=cfg.max_model_len,
        dtype=cfg.dtype,
        enforce_eager=True,
    )

    outputs = llm.generate(
        [cfg.prompt],
        SamplingParams(max_tokens=cfg.max_tokens, temperature=0.0),
    )

    generated = outputs[0].outputs[0].text
    print(f"\nPrompt   : {cfg.prompt}")
    print(f"Generated: {generated}")


if __name__ == "__main__":
    main()
