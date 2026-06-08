"""

Example: Quantize a model with JointQ and run inference with vLLM

Performs the following steps:
  1. Quantize with JointQ (4-bit, group_size=128)
  2. Save the quantized model
  3. Load the quantized model with vLLM's offline LLM interface
  4. Generate text

JointQ emits ``quant_method="gptq"`` in the saved ``config.json`` because it
reuses the same scale/zero/assignment structure as GPTQ. vLLM therefore serves
JointQ-quantized models with its built-in GPTQ plugin -- no OneComp-specific
vLLM plugin is required.

Notes:
  - JointQ requires a CUDA GPU (the local search is CUDA-based).
  - JointQ does not support QEP, so the Runner is created with ``qep=False``.
  - Use ``bits >= 2`` for vLLM serving. ``bits=1`` cannot be bit-packed by
    GPTQLinear and is saved with ``pack_weights=False``, which vLLM cannot load.

Requirements:
  pip install vllm

Note:
  vLLM runs a DeepGEMM (FP8) kernel warmup at engine startup even for
  non-FP8 quantization. If ``deep_gemm`` is not installed this fails with
  ``RuntimeError: DeepGEMM backend is not available or outdated``.
  OneComp-quantized models do not need DeepGEMM, so disable the FP8 path
  before running this script::

      export VLLM_USE_DEEP_GEMM=0
      export VLLM_DEEP_GEMM_WARMUP=skip

  See docs/user-guide/vllm-inference.md (Troubleshooting) for details.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import gc

import torch
from vllm import LLM, SamplingParams

from onecomp import CalibrationConfig, JointQ, ModelConfig, Runner, setup_logger


def main():
    setup_logger()

    # Step 1: Quantize with JointQ
    save_dir = "./TinyLlama-1.1B-Chat-jointq-4bit"

    model_config = ModelConfig(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    # bits=4 (>= 2 required for vLLM bit-packing), group_size=128.
    quantizer = JointQ(bits=4, group_size=128)
    calibration_config = CalibrationConfig(
        num_calibration_samples=128,
        max_length=512,
    )
    # NOTE: JointQ does not support QEP, so qep=False is required here.
    # The calibration settings above are kept compact so the demo runs fast
    # and may be insufficient for real quantisation. For higher quality,
    # prefer the CalibrationConfig() defaults
    # (max_length=2048, num_calibration_samples=512).
    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=calibration_config,
        qep=False,
    )
    runner.run()

    # Step 2: Save the quantized model (quant_method="gptq")
    runner.save_quantized_model(save_dir)

    # Free GPU memory used by quantization before loading vLLM
    del runner
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Load the quantized model with vLLM.
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
