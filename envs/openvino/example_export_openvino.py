# Sample script: convert a GPTQ-quantized model to OpenVINO IR and save it.
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types


def install_auto_gptq_shim():
    """
    Compatibility shim for environments where optimum-intel expects
    `from auto_gptq import exllama_set_max_input_length`.

    In this version of optimum-intel, `auto_gptq` is still referenced
    internally. As a temporary workaround, this shim exposes the same-named
    function from gptqmodel under the expected `auto_gptq` module path.

    When optimum-intel imports `auto_gptq`, this shim exposes
    `gptqmodel.exllama_set_max_input_length` under that module name.
    Call this function before importing optimum-intel.
    """
    if "auto_gptq" in sys.modules:
        return

    try:
        from gptqmodel import exllama_set_max_input_length
    except Exception as e:
        raise ImportError(
            "auto_gptq shim failed: "
            "gptqmodel.exllama_set_max_input_length could not be imported"
        ) from e

    auto_gptq = types.ModuleType("auto_gptq")
    auto_gptq.__dict__["exllama_set_max_input_length"] = exllama_set_max_input_length
    auto_gptq.__dict__["__version__"] = "shim-for-gptqmodel"
    auto_gptq.__dict__["__all__"] = ["exllama_set_max_input_length"]

    sys.modules["auto_gptq"] = auto_gptq

install_auto_gptq_shim()

import openvino as ov
from transformers import AutoTokenizer

try:
    from optimum.intel.openvino import OVModelForCausalLM
except ImportError:
    from optimum.intel import OVModelForCausalLM

from openvino_tokenizers import convert_tokenizer

HF_LOAD_KWARGS = {
    "trust_remote_code": True,
    "local_files_only": True,
}
OUT_DIR = Path("ov_gptq_int4_model_from_onecomp")
# Replace this placeholder with the path to your local onecomp GPTQ model.
MODEL_PATH = "CHANGE_TO_ONECOMP_GPTQ_MODEL_PATH"


def add_torch_fused_to_supported_quant_types():
    """
    Some OpenVINO GPTQ models are loaded with the quantization type
    "torch_fused".
    It is not always included in `supported_quant_types` by default, so
    append it here.

    This is a temporary compatibility workaround and can be removed once
    upstream support includes "torch_fused" by default.
    """
    import openvino.frontend.pytorch.gptq as ov_gptq

    if hasattr(ov_gptq, "supported_quant_types"):
        type_add = "torch_fused"
        if type_add not in ov_gptq.supported_quant_types:
            ov_gptq.supported_quant_types.append(type_add)

    print("OpenVINO GPTQ supported_quant_types:", ov_gptq.supported_quant_types)


def prepare_model_path_for_openvino_export(model_path: str) -> tuple[str, Path | None]:
    """Normalize legacy onecomp GPTQ config for OpenVINO export.

    Some onecomp saves use List[str] for quantization_config.modules_in_block_to_quantize,
    while current HF/Optimum GPTQ expects List[List[str]].
    When legacy shape is detected, copy the model directory to a temporary path,
    patch config.json there, and return that temp path for loading.
    """
    src_dir = Path(model_path)
    config_path = src_dir / "config.json"
    if not config_path.is_file():
        return model_path, None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read config.json for compatibility patch: {e}")
        return model_path, None

    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict):
        return model_path, None

    modules = quant_config.get("modules_in_block_to_quantize")
    is_legacy_list_shape = (
        isinstance(modules, list)
        and len(modules) > 0
        and all(isinstance(name, str) for name in modules)
    )
    if not is_legacy_list_shape:
        return model_path, None

    temp_dir = Path(tempfile.mkdtemp(prefix="onecomp_ov_export_compat_"))
    shutil.copytree(src_dir, temp_dir, dirs_exist_ok=True)

    temp_config_path = temp_dir / "config.json"
    with open(temp_config_path, "r", encoding="utf-8") as f:
        temp_config = json.load(f)
    temp_config["quantization_config"]["modules_in_block_to_quantize"] = [modules]
    with open(temp_config_path, "w", encoding="utf-8") as f:
        json.dump(temp_config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        "[INFO] Applied compatibility patch: "
        "modules_in_block_to_quantize List[str] -> List[List[str]]"
    )
    print(f"[INFO] Using temporary model directory: {temp_dir}")
    return str(temp_dir), temp_dir


def main():
    add_torch_fused_to_supported_quant_types()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path_for_load, temp_model_dir = prepare_model_path_for_openvino_export(MODEL_PATH)

    try:
        # For an already-GPTQ-quantized model, do not pass
        # OVWeightQuantizationConfig(bits=4) here.
        # Keep the existing GPTQ INT4 weights when exporting to OpenVINO IR.
        model = OVModelForCausalLM.from_pretrained(
            model_path_for_load,
            export=True,
            compile=False,
            **HF_LOAD_KWARGS,
            load_in_8bit=False,
        )
        print(f"model.save_pretrained({OUT_DIR.resolve()})")
        model.save_pretrained(OUT_DIR)

        # Save the Hugging Face tokenizer as well.
        tokenizer = AutoTokenizer.from_pretrained(
            model_path_for_load,
            **HF_LOAD_KWARGS,
        )
        print(f"tokenizer.save_pretrained({OUT_DIR.resolve()})")
        tokenizer.save_pretrained(OUT_DIR)

        # Save OpenVINO tokenizer and detokenizer in the same directory for OpenVINO GenAI.
        try:
            ov_tokenizer, ov_detokenizer = convert_tokenizer(
                tokenizer,
                with_detokenizer=True,
            )
            ov.save_model(ov_tokenizer, OUT_DIR / "openvino_tokenizer.xml")
            ov.save_model(ov_detokenizer, OUT_DIR / "openvino_detokenizer.xml")
        except Exception as e:
            print(f"[WARN] OpenVINO tokenizer conversion failed: {e}")
            print("[WARN] HF tokenizer files were still saved. Check tokenizer compatibility.")

        print(f"Saved OpenVINO IR model to: {OUT_DIR.resolve()}")
    finally:
        if temp_model_dir is not None:
            shutil.rmtree(temp_model_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
