"""

Example: Quantize a model with OneComp and run inference with vLLM

Performs the following steps:
  1. Quantize with GPTQ (4-bit, groupsize=128)
  2. Save the quantized model
  3. Load the quantized model with vLLM's offline LLM interface
  4. Generate text

Requirements:
  pip install vllm

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import gc

import json
import os
import torch
from onecomp import CalibrationConfig, GPTQ, ModelConfig, Runner, setup_logger
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTBase

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    raise SystemExit(
        "This example requires vllm to be installed. "
        "Install with: uv sync --extra vllm"
    ) from e

if not hasattr(_PTBase, "all_special_tokens_extended"):
    _PTBase.all_special_tokens_extended = property(
        lambda self: list(self.all_special_tokens)
    )

def _ensure_fast_tokenizer_class(save_dir: str) -> None:
    """Rewrite tokenizer_config.json so vLLM loads the fast tokenizer."""
    tok_json = os.path.join(save_dir, "tokenizer.json")
    tok_cfg = os.path.join(save_dir, "tokenizer_config.json")
    if not (os.path.isfile(tok_json) and os.path.isfile(tok_cfg)):
        return

    with open(tok_cfg, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    current = cfg.get("tokenizer_class")
    if current and current.endswith("Fast"):
        return

    slow_to_fast = {
        "LlamaTokenizer": "LlamaTokenizerFast",
        "CodeLlamaTokenizer": "CodeLlamaTokenizerFast",
    }
    replacement = slow_to_fast.get(current, "LlamaTokenizerFast")
    cfg["tokenizer_class"] = replacement

    with open(tok_cfg, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(
        f"Patched {tok_cfg}: tokenizer_class {current!r} -> {replacement!r} "
        "(forces vLLM to use the fast tokenizer)"
    )


def main():
    setup_logger()

    # Step 1: Quantize with GPTQ
    save_dir = "./TinyLlama-1.1B-Chat-gptq-4bit"

    model_config = ModelConfig(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device="cuda:0",
    )
    quantizer = GPTQ(wbits=4, groupsize=128)
    calibration_config = CalibrationConfig(
        max_length=128,
        num_calibration_samples=16,
        batch_size=8
    )
    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=calibration_config,
        qep=False,
    )
    # NOTE: The calibration settings above are kept compact so the demo
    # runs fast and may be insufficient for real quantisation.  For
    # higher quality, prefer the CalibrationConfig() defaults
    # (max_length=2048, num_calibration_samples=512).
    # For qep=False runs with large calibration data, also pass
    # ``batch_size`` as a CalibrationConfig argument, e.g.
    #   CalibrationConfig(
    #       max_length=2048,
    #       num_calibration_samples=512,
    #       batch_size=128,
    #   )
    # so that Runner.quantize_with_calibration_chunked runs instead of
    # a single all-at-once forward pass.
    runner.run()

    # Step 2: Save the quantized model
    runner.save_quantized_model(save_dir)
    print(f"\nSaved GPTQ model to: {save_dir}")

    # Free GPU memory used by quantization before loading vLLM
    del runner
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Load the quantized model with vLLM.
    # gpu_memory_utilization=0.78 leaves headroom for the residual
    # quantizer process (~16 GiB) on a UMA 121.7 GiB device (e.g. DGX
    # Spark / GB200). The vLLM default 0.92 cgroup-OOMs on shared-memory
    # GPUs.
    _ensure_fast_tokenizer_class(save_dir)

    llm = LLM(
        model=save_dir,
        max_model_len=512,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.78,
    )

    # Step 4: Generate text
    prompts = [
        "Explain what post-training quantization is in one sentence:",
        "The capital of France is",
    ]

    outputs = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0.0))

    for output in outputs:
        print(f"Prompt:   {output.prompt}")
        print(f"Response: {output.outputs[0].text}")
        print()


if __name__ == "__main__":
    main()
