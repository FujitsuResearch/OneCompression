"""

Example: Quantize a model with DBF and run inference with vLLM

Performs the following steps:
  1. Quantize with DBF
  2. Save the quantized model
  3. Load the quantized model with vLLM's offline LLM interface
  4. Generate text

Requirements:
  pip install vllm

Notes:
  vLLM loads DBF models through OneComp's DBF plugin. This example uses the
  naive DBF linear path, which is the supported DBF vLLM inference path.
  The setting is applied before importing vLLM below.

  vLLM runs a DeepGEMM (FP8) kernel warmup at engine startup even for
  non-FP8 quantization such as DBF. If ``deep_gemm`` is not installed this
  fails with ``RuntimeError: DeepGEMM backend is not available or outdated``.
  OneComp-quantized models do not need DeepGEMM, so disable the FP8 path
  before running this script::

      export VLLM_USE_DEEP_GEMM=0
      export VLLM_DEEP_GEMM_WARMUP=skip

  See docs/user-guide/vllm-inference.md (Troubleshooting) for details.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import gc
import os

import torch

os.environ.setdefault("ONECOMP_DBF_NAIVE_LINEAR", "1")

from vllm import LLM, SamplingParams

from onecomp import DBF, CalibrationConfig, ModelConfig, Runner, setup_logger


def main():
    setup_logger()

    # Step 1: Quantize with DBF
    save_dir = "./TinyLlama-1.1B-Chat-dbf"

    model_config = ModelConfig(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    quantizer = DBF(
        target_bits=1.5,
        # Keep the example compact. Increase these values for quality-focused runs.
        iters=10,
        balance_iters=5,
    )
    calibration_config = CalibrationConfig(
        num_calibration_samples=32,
        max_length=512,
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