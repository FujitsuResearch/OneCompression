# Export OneComp GPTQ 4-bit Models to OpenVINO

This guide explains how to export a OneComp GPTQ 4-bit model to OpenVINO format using an isolated environment under `envs/openvino`.

Unless otherwise noted, run all commands from the repository root.

## 1. Prerequisite: Prepare a GPTQ 4-bit Model with OneComp

Create the GPTQ 4-bit model first.

Example:

```python
from onecomp import Runner, ModelConfig, CalibrationConfig, GPTQ, setup_logger

def main():
    setup_logger()

    save_dir = "./TinyLlama-1.1B-Chat-gptq-4bit"

    model_config = ModelConfig(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    quantizer = GPTQ(wbits=4, groupsize=128)
    calibration_config = CalibrationConfig(
        num_calibration_samples=128,
        max_length=512,
    )
    
    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=calibration_config,
        qep=False,
    )
    runner.run()

    runner.save_quantized_model(save_dir)

if __name__ == "__main__":
    main()
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
