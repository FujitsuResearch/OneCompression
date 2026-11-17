# JointQ Unregularization Ablation

This benchmark evaluates JointQ with regularization disabled.

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
| Llama-3-8B | `cuda:0` | 64 |
| Qwen3-8B | `cuda:0` | 64 |
| Qwen3-14B | `cuda:0` | 16 |

## Configurations

| Setting | Quantizer | Bits | Group size |
|---|---|---:|---:|
| 4b/g128 | `JointQ_4bit_g128_unregularized` | 4 | 128 |
| 4b/pc | `JointQ_4bit_pc_unregularized` | 4 | `null` |
| 3b/g128 | `JointQ_3bit_g128_unregularized` | 3 | 128 |
| 3b/pc | `JointQ_3bit_pc_unregularized` | 3 | `null` |

Every q/k/v/o/gate/up/down layer uses regularization lambda `0.0`. All settings
use `actorder=false`, `enable_group_swap=false`, and disable regularization retry
(`regularization_retry_lambda=null`).

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
| `JointQ_4bit_g128_unregularized` | 5.67 | 0.4360 | 0.7252 | 0.7829 | 0.6930 | 0.7500 | 0.7505 | 0.6896 | 675.5 |
| `JointQ_4bit_pc_unregularized` | 6.27 | 0.4258 | 0.6843 | 0.7769 | 0.6819 | 0.7253 | 0.7223 | 0.6694 | 724.0 |
| `JointQ_3bit_g128_unregularized` | 6.89 | 0.3985 | 0.6684 | 0.7628 | 0.6803 | 0.6781 | 0.7028 | 0.6485 | 929.9 |
| `JointQ_3bit_pc_unregularized` | 23.02 | 0.2551 | 0.3321 | 0.5892 | 0.5201 | 0.3961 | 0.6138 | 0.4511 | 1519.6 |

Quantization phase elapsed time: 6223.08 s (~1h 43m).

### Llama-3-8B

| Quantizer | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| - (Original) | 6.14 | 0.5384 | 0.7757 | 0.8058 | 0.7332 | 0.7921 | 0.8199 | 0.7442 | - |
| `JointQ_4bit_g128_unregularized` | 6.65 | 0.5179 | 0.7811 | 0.7927 | 0.7206 | 0.7790 | 0.8171 | 0.7348 | 967.5 |
| `JointQ_4bit_pc_unregularized` | 7.59 | 0.5068 | 0.7689 | 0.7884 | 0.7190 | 0.7586 | 0.8116 | 0.7256 | 883.5 |
| `JointQ_3bit_g128_unregularized` | 8.90 | 0.4548 | 0.6944 | 0.7731 | 0.7143 | 0.7253 | 0.7813 | 0.6905 | 1239.7 |
| `JointQ_3bit_pc_unregularized` | 315.46 | 0.3763 | 0.6385 | 0.7263 | 0.6898 | 0.6352 | 0.7281 | 0.6324 | 1704.2 |

Quantization phase elapsed time: 7393.30 s (~2h 3m).

### Qwen3-8B

| Quantizer | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| - (Original) | 9.73 | 0.5640 | 0.8093 | 0.7758 | 0.6772 | 0.7489 | 0.8664 | 0.7403 | - |
| `JointQ_4bit_g128_unregularized` | 10.11 | 0.5401 | 0.7904 | 0.7655 | 0.6764 | 0.7421 | 0.8685 | 0.7305 | 1907.6 |
| `JointQ_4bit_pc_unregularized` | 10.66 | 0.5341 | 0.7378 | 0.7748 | 0.6938 | 0.7240 | 0.8596 | 0.7207 | 1188.4 |
| `JointQ_3bit_g128_unregularized` | 11.54 | 0.5239 | 0.7715 | 0.7622 | 0.6938 | 0.6970 | 0.8630 | 0.7186 | 1809.6 |
| `JointQ_3bit_pc_unregularized` | 30.19 | 0.4599 | 0.7045 | 0.7459 | 0.6654 | 0.6273 | 0.8150 | 0.6697 | 1691.3 |

Quantization phase elapsed time: 9501.17 s (~2h 38m).

### Qwen3-14B

| Quantizer | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| - (Original) | 8.65 | 0.6041 | 0.8279 | 0.7976 | 0.7309 | 0.7877 | 0.8936 | 0.7736 | - |
| `JointQ_4bit_g128_unregularized` | 8.88 | 0.6109 | 0.8266 | 0.7965 | 0.7159 | 0.7813 | 0.8939 | 0.7709 | 6839.2 |
| `JointQ_4bit_pc_unregularized` | 9.41 | 0.6126 | 0.8270 | 0.7943 | 0.7167 | 0.7674 | 0.8927 | 0.7684 | 3228.0 |
| `JointQ_3bit_g128_unregularized` | 9.91 | 0.5538 | 0.7967 | 0.7927 | 0.7230 | 0.7481 | 0.8731 | 0.7479 | 6020.7 |
| `JointQ_3bit_pc_unregularized` | 20.20 | 0.5213 | 0.7698 | 0.7720 | 0.7119 | 0.7147 | 0.8706 | 0.7267 | 4317.4 |

Quantization phase elapsed time: 25756.81 s (~7h 9m).

## Notes

- Original rows report the unmodified model; quantizer rows report dequantized
  model evaluation.
- `Quant time (s)` is the quantizer's total processing time. Quantization phase
  elapsed time also includes calibration preparation and framework overhead.