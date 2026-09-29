# Export OneComp GPTQ Models to OpenVINO 2026.3

This directory provides an isolated Python 3.12 environment and a conversion example for
exporting local OneComp GPTQ 4-bit checkpoints to OpenVINO IR.

- `example_export_openvino.py`: GPTQ 4-bit export for text-generation models

Run the following commands from the repository root.

## 1. Create the isolated environment

uv creates the project environment at `envs/openvino/.venv`. The lock file pins OpenVINO
2026.3.1 and the matching conversion dependencies. An explicit sync is optional because the
first `uv run --project envs/openvino ...` command also creates and synchronizes `.venv`.

```bash
uv sync --project envs/openvino --locked

uv lock --check --project envs/openvino
```

## 2. Prepare a OneComp GPTQ 4-bit checkpoint

Start from an existing local OneComp GPTQ 4-bit checkpoint containing model weights and a
`quantization_config` in `config.json`. Its architecture must be supported by Transformers,
Optimum Intel, and OpenVINO.

When `modules_in_block_to_quantize` uses the flat `List[str]` shape, the exporter
copies the checkpoint to a temporary directory and normalizes only the copied config to the
`List[List[str]]` shape that current Transformers and Optimum require. The source checkpoint
is not modified. Set `TMPDIR` to a large local filesystem when exporting a large checkpoint
and the default temporary directory does not have enough capacity. Create the target
directory before running the exporter; otherwise Python silently falls back to a different
temporary directory such as `/tmp`.

## 3. Update the model path in the export example

Open `envs/openvino/example_export_openvino.py` and edit the two constants at the top of the
file: set `MODEL_PATH` to the OneComp GPTQ 4-bit checkpoint from step 2, and `OUT_DIR` to the
directory that should receive the OpenVINO IR.

```python
# Replace this placeholder with the path to your local OneComp GPTQ 4-bit model.
MODEL_PATH = "CHANGE_TO_ONECOMP_GPTQ_MODEL_PATH"

# Directory that receives the OpenVINO IR and tokenizer files.
OUT_DIR = Path("ov_gptq_int4_model_from_onecomp")
```

## 4. Run the export

```bash
uv run --project envs/openvino --locked \
  python envs/openvino/example_export_openvino.py
```

The example keeps the checkpoint's GPTQ 4-bit weights, so it passes neither
`OVWeightQuantizationConfig` nor another OpenVINO weight-compression option. It writes the
model IR, the Hugging Face tokenizer files, and the OpenVINO tokenizer and detokenizer IR
required by OpenVINO GenAI, then parses every generated model IR as a check.

## 5. VLM checkpoints

The example uses `OVModelForCausalLM`, which does not handle multimodal models. For a VLM,
use the model-specific Optimum class such as `OVModelForVisualCausalLM` and keep the rest of
the flow, including the `modules_in_block_to_quantize` normalization from step 2. Text input
and output need only the tokenizer files that the checkpoint already contains. Processor
metadata such as `processor_config.json` is required for image or audio input, and can be
taken from a separately pinned upstream revision when the quantized checkpoint omits it.

A VLM export produces several component IR files, such as the language model, text
embeddings, per-layer embeddings, and vision embeddings. The GPTQ weights cover only the
language model, so the embedding components stay uncompressed. Those submodels alone can be
reduced with NNCF weight compression (`nncf.compress_weights` with
`CompressWeightsMode.INT8_ASYM`), leaving the GPTQ language model untouched. NNCF is already
part of this locked environment.

## 6. Run inference on an NPU machine

Copy the exported directory to an NPU machine with the same environment, then run inference
with OpenVINO GenAI. For a text-generation export:

```python
import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline("COPIED_MODEL_DIR", "NPU")
result = pipe.generate(["YOUR_PROMPT"], max_new_tokens=100)
print(result.texts[0])
```

For VLM inference, use `ov_genai.VLMPipeline` instead. Pass a plain prompt string because the
pipeline applies the model's chat template; do not apply it yourself.
