# Basic Model Validation (AutoBit, no QEP)

Validates OneComp's `AutoBitQuantizer` on multiple models using a fixed
`target_bit=4` budget with QEP disabled. Configuration is managed with
[Hydra](https://hydra.cc/).

## Purpose

- Confirm that AutoBit (4-bit target, QEP off) runs end-to-end on a
  variety of model architectures and sizes.
- Save the quantized model and report original vs quantized perplexity
  as a sanity check (not an accuracy benchmark).

## Requirements

Hydra is not part of OneComp's runtime dependencies. Install it via the
`hydra` extra:

```bash
# uv
uv sync --extra <cuXXX> --extra hydra

# pip
pip install "onecomp[hydra]"
```

Replace `<cuXXX>` with the CUDA variant matching your environment
(`cpu`, `cu118`, `cu121`, `cu124`, `cu126`, `cu128`, `cu130`).

## Usage

Specify a model via either `model_path` (local directory) or
`model_id` (Hugging Face Hub). Exactly one of the two is required.

```bash
# Local model
python validate_autobit.py model_path=/path/to/model

# Hugging Face Hub
python validate_autobit.py model_id=<HF Hub ID>
```

### Hydra Overrides

Any field in [conf/validate.yaml](conf/validate.yaml) can be overridden
on the command line, for example:

```bash
python validate_autobit.py model_path=/path/to/model output_dir=outputs/custom
```

### Outputs

For each run, Hydra changes into `output_dir` and the following are
produced:

- `quantized/`  - quantized model saved via `runner.save_quantized_model(...)`
- standard Hydra logs (`.hydra/`, `*.log`)
- stdout: original / quantized perplexity

## Validated Models

The validation set covers the following models:

- TinyLlama-1.1B (`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`)
- Llama-2-7B
- Llama-3-8B
- Qwen3-8B
- gemma-4-E2B

## Results

- Perplexity dataset: `wikitext-2-raw-v1` (OneComp default).
- AutoBit candidates: `GPTQ(wbits=b, groupsize=128) for b in (2, 3, 4, 8)`.
- `target_bit=4`, `assignment_strategy="activation_aware"`, `qep=False`.
- Calibration: `CalibrationConfig(max_length=512, num_calibration_samples=128)`.

| Model | Original PPL | Quantized PPL | Notes |
|---|---:|---:|---|
| TinyLlama-1.1B | 7.77 | 8.75 | OK |
| gemma-4-E2B (base) | 25.99 | 2.40e13 | NG |
| Llama-2-7B | 5.47 | 5.94 | OK |
| Llama-3-8B | 6.14 | 7.27 | OK |
| Qwen3-8B | 9.72 | 10.74 | OK |
