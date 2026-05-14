"""

Example: Quantize a model with OneComp and run inference with vLLM + LoRA

Performs the following steps:
  1. Quantize with GPTQ (4-bit, groupsize=128) + LoRA SFT on OneCompression knowledge data
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
from onecomp import CalibrationConfig, GPTQ, ModelConfig, PostProcessLoraSFT, Runner, setup_logger
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTBase

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
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

    # Step 1: Quantize with GPTQ + LoRA SFT
    save_dir = "./TinyLlama-1.1B-Chat-gptq-4bit-lora"
    lora_path = os.path.join(save_dir, "lora_adapter")
    knowledge_data = str(os.path.join(os.path.dirname(__file__), "onecomp_knowledge.jsonl"))

    model_config = ModelConfig(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device="cuda:0",
    )
    quantizer = GPTQ(wbits=4, groupsize=128)
    post_process = PostProcessLoraSFT(
        data_files=knowledge_data,
        max_length=256,
        epochs=50,
        batch_size=2,
        gradient_accumulation_steps=1,
        lr=3e-4,
        lora_r=16,
        lora_alpha=32,
        logging_steps=5,
    )
    calibration_config = CalibrationConfig(
        max_length=128,
        num_calibration_samples=16,
        batch_size=8
    )
    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=calibration_config,
        post_processes=[post_process],
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
    print(f"\nSaved GPTQ + LoRA model (base + adapter sidecar) to: {save_dir}")

    # Free GPU memory used by quantization before loading vLLM
    del runner
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Load the quantized model with vLLM and enable LoRA
    # _ensure_fast_tokenizer_class(save_dir)
    llm = LLM(
        model=save_dir,
        max_model_len=512,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.55,
        max_num_batched_tokens=512,
        enable_prefix_caching=False,
        enable_lora=True,
        max_lora_rank=16,
    )

    lora_request = LoRARequest(
        lora_name="gptq_lora_sft",
        lora_int_id=1,
        lora_path=lora_path,
    )

    # Step 4: Generate text
    prompts = [
        "Q: What is OneCompression?\nA:",
    ]
    sampling_params = SamplingParams(max_tokens=64, temperature=0.0)

    base_outputs = llm.generate(
        prompts,
        sampling_params,
    )

    lora_outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,
    )

    print("vLLM inference — GPTQ base vs GPTQ + LoRA SFT adapter")
    for base_out, lora_out in zip(base_outputs, lora_outputs):
        print(f"\nPrompt:         {base_out.prompt}")
        print(f"  base only  :  {base_out.outputs[0].text}")
        print(f"  base + LoRA:  {lora_out.outputs[0].text}")


if __name__ == "__main__":
    main()
