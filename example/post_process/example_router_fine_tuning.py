"""Example: MoE quantization + router-only fine-tuning.

End-to-end demonstration of the RouterFineTuning post-process workflow:
    1. Quantize a mixture-of-experts model with GPTQ
    2. Fine-tune only its routers with next-token prediction loss
    3. Evaluate PPL (original vs quantized + router fine-tuning)
    4. Save the fine-tuned model to HF-compatible safetensors

Copyright 2025-2026 Fujitsu Ltd.

Usage:
    python example/post_process/example_router_fine_tuning.py
"""

from onecomp import (
    GPTQ,
    CalibrationConfig,
    ModelConfig,
    RouterFineTuning,
    Runner,
    setup_logger,
)


def main():
    setup_logger()

    save_dir = "./gpt-oss-20b-mixed_gptq_router_ft"

    model_config = ModelConfig(
        model_id="openai/gpt-oss-20b",
    )
    quantizer = GPTQ(wbits=4, groupsize=128)
    print(
        f"Quantizer: {type(quantizer).__name__} (wbits={quantizer.wbits}, groupsize={quantizer.groupsize})"
    )

    router_fine_tuning = RouterFineTuning(
        dataset_name="Salesforce/wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        train_split="train",
        text_column="text",
        max_train_samples=512,
        max_length=512,
        epochs=1,
        batch_size=1,
        gradient_accumulation_steps=8,
        lr=1e-5,
        logging_steps=10,
    )

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=CalibrationConfig(
            max_length=512,
            num_calibration_samples=128,
        ),
        moe_quant_experts=True,
        post_processes=[router_fine_tuning],
    )
    runner.run()

    original_ppl, _, fine_tuned_ppl = runner.calculate_perplexity(
        original_model=True,
        quantized_model=True,
    )
    print(f"\nOriginal MoE PPL:                {original_ppl:.4f}")
    print(f"Quantized MoE + router FT PPL:   {fine_tuned_ppl:.4f}")

    runner.save_quantized_model(save_dir)
    print(f"\nModel saved (safetensors) to {save_dir}")


if __name__ == "__main__":
    main()
