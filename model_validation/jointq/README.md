# Model Validation (JointQ 4-bit, group_size=128, symmetric)

Validates OneComp's `JointQ` quantizer on multiple models with a fixed
`bits=4`, `group_size=128`, `symmetric=True` configuration and
`qep=False`. Configuration is managed with [Hydra](https://hydra.cc/).

## Purpose

- Confirm that pure JointQ (4-bit, gs=128, symmetric, no QEP) runs
  end-to-end on a variety of model architectures and sizes.
- Compare original vs dequantized perplexity for each model.

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
python validate_jointq.py model_path=/path/to/model

# Hugging Face Hub
python validate_jointq.py model_id=<HF Hub ID>
```

### Hydra Overrides

Any field in [conf/validate.yaml](conf/validate.yaml) can be overridden
on the command line, for example:

```bash
python validate_jointq.py model_path=/path/to/model output_dir=outputs/custom
```

### Outputs

For each run, Hydra changes into `output_dir` and the following are
produced:

- standard Hydra logs (`.hydra/`, `*.log`)
- stdout: original / dequantized perplexity

JointQ does not currently expose a quantized-inference layer, so
`runner.save_quantized_model(...)` is not available for this quantizer.
Quality is therefore measured on the dequantized model (weights
reconstructed from JointQ's quantization parameters), which is
numerically equivalent to running the quantized weights at inference
time before any kernel-level fusion.

## Validated Models

The validation set covers the following models:

- TinyLlama-1.1B (`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`)
- gemma-4-E2B
- Llama-2-7B
- Llama-3-8B
- Qwen3-8B

## Results

Perplexity is measured on `wikitext-2-raw-v1` (OneComp default).
Quantizer is `JointQ(bits=4, group_size=128, symmetric=True)` with
`qep=False` and
`CalibrationConfig(max_length=512, num_calibration_samples=128)`.

| Model | Original PPL | Dequantized PPL | Notes |
|---|---:|---:|---|
| TinyLlama-1.1B | 7.77 | 8.25 | OK |
| gemma-4-E2B (base) | 25.99 | 27.88 | OK |
| Llama-2-7B | 5.47 | 5.64 | OK |
| Llama-3-8B | 6.14 | 6.67 | OK |
| Qwen3-8B | 9.72 | 10.21 | OK |

### Notes

The calibration here (`num_calibration_samples=128`, `max_length=512`)
sits below typical research settings. The PPL numbers above are
reported only as a sanity check that this recipe runs and produces a
dequantized model that is not obviously broken on a per-model basis;
PPL differences against other recipes (e.g. `gptq/`, `qep_gptq/`,
`autobit/`, `autobit_qep/`) under their own respective calibrations
are not interpreted as which recipe is "better" or as evidence about
which mechanism (symmetric, QEP, AutoBit, etc.) fixed or broke a given
model. The cross-recipe table in the parent
[`model_validation/README.md`](../README.md) should be read with the
same caveat.

Per-model observations (this recipe only):

- **TinyLlama-1.1B**: PPL 7.77 → 8.25 — passes the sanity check.
- **gemma-4-E2B (base)**: PPL 25.99 → 27.88 — passes the sanity check.
- **Llama-2-7B**: PPL 5.47 → 5.64 — passes the sanity check.
- **Llama-3-8B**: PPL 6.14 → 6.67 — passes the sanity check.
- **Qwen3-8B**: PPL 9.72 → 10.21 — passes the sanity check.

All five models fit within the DGX Spark 128 GB UMA budget at this
calibration with `qep=False`; no OOM at `mlp.down_proj` for the 7–8B
models.
