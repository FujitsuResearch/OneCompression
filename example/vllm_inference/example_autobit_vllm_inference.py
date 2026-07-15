"""

Example: Quantize a model with auto_run and run inference with vLLM

Performs the following steps:
  1. Quantize with AutoBit (mixed-precision) + QEP using auto_run
  2. Load the quantized model with vLLM's offline LLM interface
  3. Generate text

Requirements:
  pip install vllm

Note:
  vLLM runs a DeepGEMM (FP8) kernel warmup at engine startup even for
  non-FP8 quantization such as mixed-precision GPTQ. If ``deep_gemm`` is not
  installed this fails with ``RuntimeError: DeepGEMM backend is not available
  or outdated``. OneComp-quantized models do not need DeepGEMM, so disable the
  FP8 path before running this script::

      export VLLM_USE_DEEP_GEMM=0
      export VLLM_DEEP_GEMM_WARMUP=skip

  See docs/user-guide/vllm-inference.md (Troubleshooting) for details.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import gc

import torch
from vllm import LLM, SamplingParams

from onecomp import Runner


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

    # Step 2: Load the quantized model with vLLM and generate text.
    # gpu_memory_utilization=0.78 leaves headroom for the residual
    # quantizer process (~16 GiB) on a UMA 121.7 GiB device (e.g. DGX
    # Spark / GB200). The vLLM default 0.92 cgroup-OOMs on shared-memory
    # GPUs.
    llm = LLM(
        model=save_dir,
        max_model_len=512,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.78,
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
