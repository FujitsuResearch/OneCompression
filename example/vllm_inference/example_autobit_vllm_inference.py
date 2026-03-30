"""

Example: Quantize a model with auto_run and run inference with vLLM

Performs the following steps:
  1. Quantize with AutoBit (mixed-precision) + QEP using auto_run
  2. Load the quantized model with vLLM's offline LLM interface
  3. Generate text

Requirements:
  pip install vllm

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import gc

import torch
from onecomp import Runner
from vllm import LLM, SamplingParams


def main():
    # Step 1: Quantize and save the model
    save_dir = "./TinyLlama-1.1B-autobit"

    runner = Runner.auto_run(
        model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        save_dir=save_dir,
        evaluate=False,
    )

    # Free GPU memory used by quantization before loading vLLM
    del runner
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: Load the quantized model with vLLM and generate text
    llm = LLM(
        model=save_dir,
        max_model_len=512,
        dtype="float16",
        enforce_eager=True,
    )

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
