"""

Example: Quantize GPT-OSS-20b (MoE) with OneComp and run inference with vLLM

Performs the following steps:
  1. Quantize with GPTQ (4-bit, groupsize=64) keeping the MoE experts 4-bit
  2. Save the quantized (mixed_gptq) checkpoint
  3. Load the quantized model with vLLM's offline LLM interface
  4. Generate text

Prerequisite: apply the vLLM runtime patches BEFORE running this script
------------------------------------------------------------------------
GPT-OSS ``mixed_gptq`` serving needs two idempotent, marker-guarded source
patches applied to the installed vLLM. They rewrite vLLM source files on disk,
so they must be applied in a separate step *before* this process imports vLLM::

    python -m vllm_plugins.patches.apply_all      # add --dry-run to preview

The patches are verified against vLLM 0.20.2 (``apply_all`` warns if the
installed version is outside the verified set). The expert routing itself
(wrapping ``MoeWNA16Method`` in ``GptOssWNA16MoEMethod``) is registered
automatically by the ``mixed_gptq`` plugin entry point — no extra step needed.
See docs/user-guide/gptoss.md for details.

Copyright 2025-2026 Fujitsu Ltd.

"""

import gc
import os

# Disable the FP8 DeepGEMM warmup before vLLM is imported.
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")

import torch
from vllm import LLM, SamplingParams

from onecomp import GPTQ, ModelConfig, Runner, setup_logger


def main():
    setup_logger()

    # Step 1: Quantize with GPTQ, keeping the MoE experts 4-bit.
    # group_size=64 is REQUIRED: GPT-OSS hidden_size=2880 is not divisible
    # by 128, and the WNA16 MoE kernel expects group size 64
    # moe_quant_experts=True keeps each expert as per-expert
    # GPTQ INT4 tensors instead of dequantizing them to dense bf16.
    save_dir = "./gpt-oss-20b-mixed_gptq"

    model_config = ModelConfig(
        model_id="openai/gpt-oss-20b",  # or openai/gpt-oss-120b
    )
    quantizer = GPTQ(wbits=4, groupsize=64)
    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        moe_quant_experts=True,
        qep=True,
    )
    runner.run()

    # Step 2: Save the quantized model.
    # GPT-OSS saves in the standard HF layout, so no save_format is needed;
    # the runner writes sharded *.safetensors + config.json with the
    # per-expert GPTQ quantization_config that the mixed_gptq plugin serves.
    runner.save_quantized_model(save_dir)

    # Free GPU memory used by quantization before loading vLLM
    del runner
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Load the quantized model with vLLM.
    # Requires `python -m vllm_plugins.patches.apply_all` to have been run first
    llm = LLM(
        model=save_dir,
        max_model_len=512,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.55,
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
