# Export OneComp GPTQ 4-bit Models to OpenVINO

This guide explains how to export a OneComp GPTQ 4-bit model to OpenVINO format using an isolated environment under `envs/openvino`.

Unless otherwise noted, run all commands from the repository root.

## 1. Prerequisite: Prepare a GPTQ 4-bit Model with OneComp

Create the GPTQ 4-bit model first.

Example:

```bash
uv run example/example_gptq.py
```

## 2. Update the Model Path in the Export Script

Open `envs/openvino/example_export_openvino.py` and set `MODEL_PATH` to the local path of the GPTQ model generated in Step 1.

## 3. Run Export in an Isolated OpenVINO Environment

The command below creates and uses an OpenVINO-specific environment that is separated from the main onecomp environment, then runs the export script.

```bash
uv run --project envs/openvino python envs/openvino/example_export_openvino.py
```

## 4. Run Inference on an NPU Machine

Copy the exported model files to a machine with an NPU, then run inference with a script such as:

```python
import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline("MODEL_PATH", "NPU")
res = pipe.generate(["YOUR_PROMPT"], max_new_tokens=100)
print(res.texts[0])
```
