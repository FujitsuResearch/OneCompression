# Basic Model Validation (QEP + GPTQ)

Validates OneComp's `GPTQ` quantizer (`wbits=4, groupsize=128`) with QEP
enabled on multiple models. Configuration is managed with
[Hydra](https://hydra.cc/).

## Purpose

- Confirm that GPTQ (`wbits=4, groupsize=128`, QEP on) runs end-to-end on
  a variety of model architectures and sizes.
- Save the quantized model and compare original vs quantized
  perplexity for each model.

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
python validate_gptq.py model_path=/path/to/model

# Hugging Face Hub
python validate_gptq.py model_id=<HF Hub ID>
```

### Hydra Overrides

Any field in [conf/validate.yaml](conf/validate.yaml) can be overridden
on the command line, for example:

```bash
python validate_gptq.py model_path=/path/to/model output_dir=outputs/custom
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

Perplexity is measured on `wikitext-2-raw-v1` (OneComp default).
Quantizer is `GPTQ(wbits=4, groupsize=128)` with `qep=True`.
Calibration: `max_length=1024, num_calibration_samples=128`.

| Model | Original PPL | Quantized PPL | Notes |
|---|---:|---:|---|
| TinyLlama-1.1B | 7.77 | 8.63 | OK |
| gemma-4-E2B (base) | 25.99 | 37.82 | OK\* |
| Llama-2-7B | 5.47 | 6.10 | OK |
| Llama-3-8B | 6.14 | 7.14 | OK |
| Qwen3-8B | 9.72 | 10.83 | OK |

### Notes

The calibration here (`num_calibration_samples=128`, `max_length=1024`)
sits below typical research settings, partly because larger
calibration does not fit the DGX Spark 128 GB UMA budget for 7–8B
models with QEP on. The PPL numbers above are reported only as a
sanity check that this recipe runs and produces a model that is not
obviously broken on a per-model basis; PPL differences against other
recipes (e.g. `gptq/`, `autobit/`, `autobit_qep/`) under their own
respective calibrations are not interpreted as which recipe is
"better" or as evidence about which mechanism (QEP, AutoBit, etc.)
fixed or broke a given model. The cross-recipe table in the parent
[`model_validation/README.md`](../README.md) should be read with the
same caveat.

Per-model observations (this recipe only):

- **TinyLlama-1.1B**: PPL 7.77 → 8.63 — passes the sanity check.
- **gemma-4-E2B (base)**: PPL 25.99 → 37.82 — borderline (OK\*),
  warn-level regression similar to `gptq/`'s 35.03. Within this
  recipe alone there is no run-log signal of breakage (no diverged
  PPL, no anomalous logs); load + inference is not exercised here,
  so visual quality of generation is not verified in this
  subdirectory.
- **Llama-2-7B**: PPL 5.47 → 6.10 — passes the sanity check.
- **Llama-3-8B**: PPL 6.14 → 7.14 — passes the sanity check.
- **Qwen3-8B**: PPL 9.72 → 10.83 — passes the sanity check.

All five models fit within the DGX Spark 128 GB UMA budget at this
calibration; no OOM at `mlp.down_proj` for the 7–8B models.
