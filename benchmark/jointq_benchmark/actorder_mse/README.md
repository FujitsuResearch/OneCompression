# JointQ Actorder and MSE Ablation

This benchmark evaluates JointQ with activation ordering and GPTQ MSE enabled.

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

## Configurations

Regularization lambda values are specified for every quantized layer type.

| Setting | Quantizer | Bits | Group size | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4b/g128 | `JointQ_4bit_g128_actorder_mse` | 4 | 128 | 0.0 | 0.0 | 0.01 | 0.0 | 0.01 | 0.01 | 0.0 |
| 3b/g128 | `JointQ_3bit_g128_actorder_mse` | 3 | 128 | 0.01 | 0.01 | 0.01 | 0.0 | 0.01 | 0.01 | 0.0 |

Both settings use `actorder=true`, GPTQ `mse=true`,
`enable_group_swap=false`, regularization retry lambda `0.01`, and retry
threshold `0.2`. GPTQ uses the same bits, group size, and symmetric setting as
JointQ.

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
| `JointQ_4bit_g128_actorder_mse` | 5.66 | 0.4403 | 0.7201 | 0.7813 | 0.6946 | 0.7511 | 0.7887 | 0.6960 | 1805.4 |
| `JointQ_3bit_g128_actorder_mse` | 6.56 | 0.4164 | 0.6877 | 0.7704 | 0.6740 | 0.7046 | 0.7303 | 0.6639 | 2226.3 |

Quantization phase elapsed time: 6448.48 s (~1h 47m).

### Llama-3-8B

| Quantizer | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| - (Original) | 6.14 | 0.5384 | 0.7757 | 0.8058 | 0.7332 | 0.7921 | 0.8199 | 0.7442 | - |
| `JointQ_4bit_g128_actorder_mse` | 6.55 | 0.5375 | 0.7908 | 0.7982 | 0.7356 | 0.7841 | 0.8174 | 0.7439 | 1913.8 |
| `JointQ_3bit_g128_actorder_mse` | 8.41 | 0.5043 | 0.7466 | 0.7731 | 0.7222 | 0.7530 | 0.8156 | 0.7191 | 2341.4 |

Quantization phase elapsed time: 6799.82 s (~1h 53m).

### Qwen3-8B

| Quantizer | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| - (Original) | 9.73 | 0.5640 | 0.8093 | 0.7758 | 0.6772 | 0.7489 | 0.8664 | 0.7403 | - |
| `JointQ_4bit_g128_actorder_mse` | 9.88 | 0.5529 | 0.8009 | 0.7748 | 0.6898 | 0.7442 | 0.8664 | 0.7382 | 2089.2 |
| `JointQ_3bit_g128_actorder_mse` | 11.58 | 0.5341 | 0.7551 | 0.7650 | 0.6914 | 0.7101 | 0.8602 | 0.7193 | 2549.3 |

Quantization phase elapsed time: 7464.65 s (~2h 4m).

### Qwen3-14B

| Quantizer | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| - (Original) | 8.65 | 0.6041 | 0.8279 | 0.7976 | 0.7309 | 0.7877 | 0.8936 | 0.7736 | - |
| `JointQ_4bit_g128_actorder_mse` | 8.93 | 0.6118 | 0.8321 | 0.7987 | 0.7293 | 0.7837 | 0.8893 | 0.7741 | 3747.3 |
| `JointQ_3bit_g128_actorder_mse` | 9.48 | 0.5819 | 0.8089 | 0.7867 | 0.7245 | 0.7598 | 0.8746 | 0.7561 | 4499.6 |

Quantization phase elapsed time: 13507.28 s (~3h 45m).

## Notes

- Original rows report the unmodified model; quantizer rows report dequantized
  model evaluation.
- `Quant time (s)` is the quantizer's total processing time. Quantization phase
  elapsed time also includes calibration preparation and framework overhead.