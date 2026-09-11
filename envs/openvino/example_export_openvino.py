"""Export a local OneComp GPTQ 4-bit checkpoint to OpenVINO IR.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from pathlib import Path

from openvino_export_utils import (
    modules_shape,
    read_checkpoint_quantization,
    save_openvino_tokenizer,
    temporary_model_path_for_openvino_export,
    validate_ir_files,
)

# Replace this placeholder with the path to your local OneComp GPTQ 4-bit model.
MODEL_PATH = "CHANGE_TO_ONECOMP_GPTQ_MODEL_PATH"

# Directory that receives the OpenVINO IR and tokenizer files.
OUT_DIR = Path("ov_gptq_int4_model_from_onecomp")


def main() -> None:
    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoTokenizer

    model_path = Path(MODEL_PATH)
    quantization = read_checkpoint_quantization(model_path)
    print(
        "[INFO] checkpoint modules_in_block_to_quantize shape: "
        f"{modules_shape(quantization.get('modules_in_block_to_quantize'))}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with temporary_model_path_for_openvino_export(model_path) as prepared_model_path:
        # For an already-GPTQ-quantized model, do not pass OVWeightQuantizationConfig
        # here. Keep the existing GPTQ 4-bit weights when exporting to OpenVINO IR.
        model = OVModelForCausalLM.from_pretrained(
            prepared_model_path,
            export=True,
            compile=False,
            local_files_only=True,
            trust_remote_code=True,
            load_in_8bit=False,
        )
        model.save_pretrained(OUT_DIR)

        # Save the Hugging Face tokenizer, plus the OpenVINO tokenizer and
        # detokenizer required by OpenVINO GenAI.
        tokenizer = AutoTokenizer.from_pretrained(
            prepared_model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        tokenizer.save_pretrained(OUT_DIR)
        save_openvino_tokenizer(tokenizer, OUT_DIR)

    parsed_models = validate_ir_files(OUT_DIR)
    print(f"[INFO] Parsed OpenVINO IR files: {', '.join(parsed_models)}")
    print(f"[INFO] Export completed: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
