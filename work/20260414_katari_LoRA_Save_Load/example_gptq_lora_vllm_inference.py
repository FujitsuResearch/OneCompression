"""
 
Example: Quantize with GPTQ + LoRA SFT, save, and serve via vLLM with LoRA
 
Performs the following steps:
  1. Quantize TinyLlama with GPTQ (4-bit, groupsize=128)
  2. Apply ``PostProcessLoraSFT`` (WikiText-2) to fine-tune LoRA adapters
     on top of the frozen GPTQ base
  3. Save via ``runner.save_quantized_model(save_dir)``. This writes:
       - ``model.safetensors`` / ``config.json``  (base GPTQ, HF format)
       - ``lora_adapter/adapter_model.safetensors`` (PEFT-format LoRA)
       - ``lora_adapter/adapter_config.json``       (PEFT-format LoRA config)
     The adapter lives in the ``lora_adapter/`` subdirectory because vLLM
     globs ``*.safetensors`` at the top level of the model directory to load
     base weights, and would otherwise try to load the adapter file as part
     of the base model.
  4. Load the base model with vLLM and apply the LoRA adapter via
     ``LoRARequest(..., lora_path=os.path.join(save_dir, "lora_adapter"))``.
  5. Generate text with and without the LoRA adapter and print both
     outputs side by side.
 
Requirements:
  - pip install vllm  (>= 0.5 recommended; older releases may lack
    LoRA-over-GPTQ support)
  - The ``mixed_gptq`` quantization method is registered by OneComp's vLLM
    plugin (see ``vllm_plugins/gptq``), which is installed automatically
    with this package.
 
vLLM LoRA constraints
---------------------
  - ``max_lora_rank`` must be >= the adapter's ``r`` (from
    ``adapter_config.json``).
  - On Llama-style models, q/k/v (and gate/up) projections are fused by
    vLLM into ``qkv_proj`` / ``gate_up_proj``. vLLM's LoRA loader
    concatenates the per-projection adapter weights automatically, provided
    all three projections share the same ``r`` and ``lora_alpha`` —
    ``PostProcessLoraSFT`` always uses a single global rank/alpha, so this
    invariant holds.
  - ``lm_head`` is excluded from LoRA targets by ``PostProcessLoraSFT``, so
    ``lora_extra_vocab_size=0`` is safe.
 
Copyright 2025-2026 Fujitsu Ltd.
 
Author: Keiji Kimura
 
"""
 
import gc
import json
import os
 
import torch
 
# ---------------------------------------------------------------------------
# transformers >= 5.x / vLLM compatibility shim
# ---------------------------------------------------------------------------
# OneComp requires transformers>=5.3, which removed the
# `all_special_tokens_extended` property from the tokenizer base class.
# vLLM (<= 0.11.x) still accesses it in ``get_cached_tokenizer`` and crashes
# with ``AttributeError: LlamaTokenizer has no attribute
# all_special_tokens_extended``. Re-add the property as an alias for
# ``all_special_tokens`` so vLLM's cache build succeeds. Must run BEFORE vLLM
# imports so subclasses inherit the attribute.
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTBase
 
if not hasattr(_PTBase, "all_special_tokens_extended"):
    _PTBase.all_special_tokens_extended = property(
        lambda self: list(self.all_special_tokens)
    )
 
from onecomp import (
    CalibrationConfig,
    GPTQ,
    ModelConfig,
    PostProcessLoraSFT,
    Runner,
    setup_logger,
)
 
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.model_executor.layers.quantization import register_quantization_config  # noqa: F401
except ImportError as e:
    raise SystemExit(
        "This example requires vllm>=0.6.3 "
        "(needs both `vllm.lora.request.LoRARequest` and "
        "`register_quantization_config` used by OneComp's mixed_gptq plugin). "
        "Install with: pip install -U 'vllm>=0.6.3'"
    ) from e
 
 
def _ensure_fast_tokenizer_class(save_dir: str) -> None:
    """Rewrite tokenizer_config.json so vLLM loads the fast tokenizer.
 
    Some upstream Llama-family checkpoints (TinyLlama included) ship with
    ``"tokenizer_class": "LlamaTokenizer"`` in ``tokenizer_config.json``.
    Recent ``transformers`` releases have removed ``all_special_tokens_extended``
    from the slow ``LlamaTokenizer``, which causes vLLM's tokenizer cache to
    crash with ``AttributeError: LlamaTokenizer has no attribute
    all_special_tokens_extended``. When ``tokenizer.json`` is present the fast
    variant is available, so we patch the class name to force vLLM down that
    path.
    """
    tok_json = os.path.join(save_dir, "tokenizer.json")
    tok_cfg = os.path.join(save_dir, "tokenizer_config.json")
    if not (os.path.isfile(tok_json) and os.path.isfile(tok_cfg)):
        return
 
    with open(tok_cfg, "r", encoding="utf-8") as f:
        cfg = json.load(f)
 
    current = cfg.get("tokenizer_class")
    if current and current.endswith("Fast"):
        return
 
    # Map slow → fast for Llama-family tokenizers. Extend this table if other
    # architectures hit the same issue in the future.
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
 
    # ================================================================
    # Step 1: Quantize + LoRA SFT and save
    # ================================================================
    save_dir = "./TinyLlama-1.1B-gptq-4bit-lora"
 
    model_config = ModelConfig(
        model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        device="cuda:0",
    )
    quantizer = GPTQ(wbits=4, groupsize=128)
    post_process = PostProcessLoraSFT(
        dataset_name="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        train_split="train",
        text_column="text",
        max_train_samples=128,
        max_length=256,
        epochs=2,
        batch_size=1,
        gradient_accumulation_steps=8,
        lr=1e-4,
        lora_r=8,
        lora_alpha=16,
        logging_steps=5,
    )
 
    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=CalibrationConfig(max_length=128, num_calibration_samples=16, batch_size=8),
        post_processes=[post_process],
    )
    runner.run()
    runner.save_quantized_model(save_dir)
    print(f"\nSaved GPTQ+LoRA model (base + adapter sidecar) to: {save_dir}")
 
    # Free GPU memory used by quantization / training before loading vLLM.
    del runner
    gc.collect()
    torch.cuda.empty_cache()
 
    # ================================================================
    # Step 2: Load base GPTQ with vLLM, enable LoRA
    # ================================================================
    # Work around the LlamaTokenizer / transformers compatibility issue
    # (see _ensure_fast_tokenizer_class for details).
    _ensure_fast_tokenizer_class(save_dir)
 
    # ``max_lora_rank`` must be >= the saved ``r`` (16 here).
    llm = LLM(
        model=save_dir,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        max_model_len=512,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.55,    # VRAM 8GB 制約ありのため削減
        max_num_batched_tokens=512,     # 同上
        enable_prefix_caching=False,    # 同上
    )
 
    # The adapter sidecar lives in the lora_adapter/ subdirectory to avoid
    # colliding with vLLM's base-model safetensors glob.
    lora_request = LoRARequest(
        lora_name="gptq_sft",
        lora_int_id=1,
        lora_path=os.path.join(save_dir, "lora_adapter"),
    )
 
    prompts = [
        "Explain what post-training quantization is in one sentence:",
        "Fujitsu is",
    ]
    sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
 
    # ----- (a) Base GPTQ only (LoRA disabled) -----
    base_outputs = llm.generate(prompts, sampling_params)
 
    # ----- (b) Base GPTQ + LoRA adapter -----
    lora_outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,
    )
 
    print("\n" + "=" * 70)
    print("vLLM inference — GPTQ base vs GPTQ + LoRA SFT adapter")
    print("=" * 70)
    for base_out, lora_out in zip(base_outputs, lora_outputs):
        print(f"\nPrompt:         {base_out.prompt}")
        print(f"  base only  :  {base_out.outputs[0].text}")
        print(f"  base + LoRA:  {lora_out.outputs[0].text}")
 
 
if __name__ == "__main__":
    main()
 