# JointQ 2-bit/group-size-32 Benchmark

This benchmark evaluates a 2-bit JointQ configuration with group size 32.

## Models

| Model | `model_path` |
|---|---|
| Llama-2-7B | `meta-llama/Llama-2-7b-hf` |
| Llama-3-8B | `meta-llama/Meta-Llama-3-8B` |
| Qwen3-8B | `Qwen/Qwen3-8B` |
| Qwen3-14B | `Qwen/Qwen3-14B` |

Other models can be evaluated by passing their path in the same way.

## Common Parameters

| Parameter | Value |
|---|---|
| Model dtype | `bfloat16` |
| Calibration samples | `1024` |
| Calibration strategy | `drop_rand` |
| Calibration sequence length | `2048` |
| Calibration seed | `0` |
| Calibration batch size | `64` |
| Symmetric quantization | `true` |
| Quantized layers | All (`jointq.num_layers=null`) |
| Quantization error statistics | Enabled |

The default model device is `cuda:0`. The reported runs used the following
resource overrides; these do not change the quantization configuration.

| Model | `model_device` | Calibration batch size |
|---|---|---:|
| Llama-2-7B | `auto` | 64 |
| Llama-3-8B | `auto` | 64 |
| Qwen3-8B | `cuda:0` | 64 |
| Qwen3-14B | `cuda:0` | 16 |

## Configuration

Regularization lambda values are specified for every quantized layer type.

| Setting | Quantizer | Bits | Group size | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2b/g32 | `JointQ_2bit_g32` | 2 | 32 | 0.1 | 0.1 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 |

This setting uses `actorder=false`, `enable_group_swap=false`, and disables
regularization retry (`regularization_retry_lambda=null`).

## Evaluation

- Perplexity: WikiText-2 test split, sequence length and stride 2048
- Accuracy: lm-eval-harness 0-shot with batch size 8; ARC Easy, ARC Challenge,
  PIQA, WinoGrande, HellaSwag, and BoolQ

Evaluation runs on the original model and each dequantized JointQ model. Packed
quantized models are not evaluated.

## Run

Run from this directory, changing `model_path` and `output_dir` for each model:

```bash
../../../.venv/bin/python quant_benchmark.py \
  model_path=meta-llama/Llama-2-7b-hf \
  output_dir=llama2-7b
```

No model-specific run script is required.

## Results

Accuracy columns use normalized accuracy where available and raw accuracy for
WinoGrande and BoolQ. `Accuracy Avg.` is the arithmetic mean of all six accuracy
tasks.

### Llama-2-7B

| Quantizer | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| - (Original) | 5.47 | 0.4480 | 0.7378 | 0.7845 | 0.6961 | 0.7617 | 0.7945 | 0.7038 | - |
| `JointQ_2bit_g32` | 26.33 | 0.2483 | 0.4474 | 0.6415 | 0.5683 | 0.4359 | 0.6147 | 0.4927 | 1164.1 |

Quantization phase elapsed time: 3572.21 s (~0h 59m).

### Llama-3-8B

| Quantizer | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| - (Original) | 6.14 | 0.5384 | 0.7757 | 0.8058 | 0.7332 | 0.7921 | 0.8199 | 0.7442 | - |
| `JointQ_2bit_g32` | 108.77 | 0.2560 | 0.4339 | 0.6349 | 0.5683 | 0.4054 | 0.6211 | 0.4866 | 1330.4 |

Quantization phase elapsed time: 3805.63 s (~1h 3m).

### Qwen3-8B

| Quantizer | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| - (Original) | 9.73 | 0.5640 | 0.8093 | 0.7758 | 0.6772 | 0.7489 | 0.8664 | 0.7403 | - |
| `JointQ_2bit_g32` | 45.98 | 0.3200 | 0.5896 | 0.6861 | 0.6038 | 0.4885 | 0.7651 | 0.5755 | 1971.2 |

Quantization phase elapsed time: 4739.99 s (~1h 18m).

### Qwen3-14B

| Quantizer | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| - (Original) | 8.65 | 0.6041 | 0.8279 | 0.7976 | 0.7309 | 0.7877 | 0.8936 | 0.7736 | - |
| `JointQ_2bit_g32` | 20.78 | 0.4087 | 0.6654 | 0.7182 | 0.6559 | 0.5805 | 0.6911 | 0.6200 | 6219.7 |

Quantization phase elapsed time: 11365.61 s (~3h 9m).

## Notes

- Original rows report the unmodified model; quantizer rows report dequantized
  model evaluation.
- `Quant time (s)` is the quantizer's total processing time. Quantization phase
  elapsed time also includes calibration preparation and framework overhead.