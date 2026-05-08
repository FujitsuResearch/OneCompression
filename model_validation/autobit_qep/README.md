# Basic Model Validation (AutoBit)

Validates OneComp's `AutoBitQuantizer` on multiple models using a fixed
`target_bit=4` budget with QEP enabled. Configuration is managed with
[Hydra](https://hydra.cc/).

## Purpose

- Confirm that AutoBit (4-bit target, QEP on) runs end-to-end on a
  variety of model architectures and sizes.
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

Perplexity is measured on `wikitext-2-raw-v1` (OneComp default).
AutoBit candidates are `GPTQ(wbits=b, groupsize=128) for b in (2, 3, 4, 8)`,
with `target_bit=4`, `assignment_strategy="activation_aware"`, and `qep=True`.
Runner calibration: `max_length=1024`, `num_calibration_samples=128`
(reduced from defaults to keep 7-8B models within the DGX Spark
128 GB UMA budget).

| Model | Original PPL | Quantized PPL | Notes |
|---|---:|---:|---|
| TinyLlama-1.1B | 7.77 | 8.67 | OK |
| gemma-4-E2B (base) | 25.99 | 1.64e14 | NG (see below) |
| Llama-2-7B | 5.47 | 5.90 | OK |
| Llama-3-8B | 6.14 | 7.24 | OK |
| Qwen3-8B | 9.72 | 10.82 | OK |

### Notes on gemma-4-E2B

Quantized perplexity diverges. AutoBit's activation-aware ILP returned a
strongly bimodal bit assignment for this model: every module in the first
15 transformer blocks was assigned 8-bit, and every module in the
remaining 20 blocks (and `per_layer_model_projection`) was assigned 2-bit;
3-bit and 4-bit were never selected. Effective bpw was reported as
~3.94 (target 4.16). Reducing the Runner calibration
(`max_length=2048→1024`, `num_calibration_samples=512→128`) did not change
this assignment — the layer split, candidates chosen, and effective bpw
were identical, and quantized PPL stayed in the same diverged regime
(9.51e13 → 1.64e14).

The 2-bit half of the network is the likely cause of the divergence.
Candidate follow-ups include:

- restricting the candidate set (e.g. `b in (3, 4, 8)`) to remove 2-bit,
- disabling QEP to isolate quantization-error propagation effects,
- switching `assignment_strategy` away from `activation_aware`.

For comparison, on the other four models the bit assignment was
balanced (`GPTQ_<b>_gs128: <count> layers` from the same run):

| Model | 2-bit | 3-bit | 4-bit | 8-bit | Effective bpw |
|---|---:|---:|---:|---:|---:|
| TinyLlama-1.1B | 0 | 28 | 91 | 35 | 4.16 |
| Llama-2-7B | 2 | 45 | 153 | 24 | 4.16 |
| Llama-3-8B | 0 | 40 | 131 | 53 | 4.16 |
| Qwen3-8B | 0 | 50 | 152 | 50 | 4.16 |
| gemma-4-E2B | 181 | 0 | 0 | 135 | 3.94 |

The 2-bit count for Llama-2-7B (2 layers / 224 total) is small enough
that PPL stays in range; gemma-4-E2B is the only model where ILP
collapses the assignment onto the 2/8-bit corners.
