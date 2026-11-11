# JointQ Main Benchmark

This benchmark evaluates the selected 4-bit and 3-bit JointQ configurations.

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
| Qwen3-8B | `auto` | 64 |
| Qwen3-14B | `cuda:0` | 16 |

## Configurations

Regularization lambda values are specified for every quantized layer type.

| Setting | Quantizer | Bits | Group size | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4b/g128 | `JointQ_4bit_g128` | 4 | 128 | 0.0 | 0.0 | 0.01 | 0.0 | 0.01 | 0.01 | 0.0 |
| 4b/pc | `JointQ_4bit_pc` | 4 | `null` | 0.0 | 0.0 | 0.15 | 0.0 | 0.15 | 0.15 | 0.01 |
| 3b/g128 | `JointQ_3bit_g128` | 3 | 128 | 0.01 | 0.01 | 0.01 | 0.0 | 0.01 | 0.01 | 0.0 |
| 3b/pc | `JointQ_3bit_pc` | 3 | `null` | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.01 |

All settings use `actorder=false`, `enable_group_swap=false`, regularization
retry lambda `0.01`, and retry threshold `0.2`.

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
| `JointQ_4bit_g128` | 5.67 | 0.4352 | 0.7176 | 0.7797 | 0.6961 | 0.7510 | 0.7688 | 0.6914 | 706.2 |
| `JointQ_4bit_pc` | 5.96 | 0.4480 | 0.7113 | 0.7791 | 0.6922 | 0.7336 | 0.7373 | 0.6836 | 1906.5 |
| `JointQ_3bit_g128` | 6.78 | 0.3959 | 0.6894 | 0.7693 | 0.6882 | 0.6926 | 0.7306 | 0.6610 | 965.9 |
| `JointQ_3bit_pc` | 7.98 | 0.3601 | 0.5934 | 0.7405 | 0.6267 | 0.6236 | 0.6749 | 0.6032 | 2161.8 |

Quantization phase elapsed time: 8214.39 s (~2h 16m).

### Llama-3-8B

| Quantizer | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| - (Original) | 6.14 | 0.5384 | 0.7757 | 0.8058 | 0.7332 | 0.7921 | 0.8199 | 0.7442 | - |
| `JointQ_4bit_g128` | 6.65 | 0.5401 | 0.7828 | 0.7965 | 0.7293 | 0.7843 | 0.8226 | 0.7426 | 937.0 |
| `JointQ_4bit_pc` | 7.41 | 0.4940 | 0.7580 | 0.7829 | 0.7301 | 0.7739 | 0.8092 | 0.7247 | 2564.0 |
| `JointQ_3bit_g128` | 8.75 | 0.4693 | 0.7168 | 0.7742 | 0.7395 | 0.7424 | 0.7936 | 0.7060 | 1202.8 |
| `JointQ_3bit_pc` | 13.09 | 0.4010 | 0.6334 | 0.7388 | 0.7056 | 0.7065 | 0.7783 | 0.6606 | 2736.9 |

Quantization phase elapsed time: 10058.42 s (~2h 47m).

### Qwen3-8B

| Quantizer | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| - (Original) | 9.73 | 0.5640 | 0.8093 | 0.7758 | 0.6772 | 0.7489 | 0.8664 | 0.7403 | - |
| `JointQ_4bit_g128` | 10.17 | 0.5572 | 0.7950 | 0.7628 | 0.6756 | 0.7393 | 0.8719 | 0.7336 | 2019.8 |
| `JointQ_4bit_pc` | 10.71 | 0.5367 | 0.7740 | 0.7688 | 0.6859 | 0.7323 | 0.8465 | 0.7240 | 1631.7 |
| `JointQ_3bit_g128` | 11.50 | 0.5375 | 0.7950 | 0.7666 | 0.6882 | 0.6968 | 0.8661 | 0.7250 | 1997.8 |
| `JointQ_3bit_pc` | 22.11 | 0.4582 | 0.7104 | 0.7345 | 0.6448 | 0.6366 | 0.8269 | 0.6686 | 2236.2 |

Quantization phase elapsed time: 10828.24 s (~3h 0m).

### Qwen3-14B

| Quantizer | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| - (Original) | 8.65 | 0.6041 | 0.8279 | 0.7976 | 0.7309 | 0.7877 | 0.8936 | 0.7736 | - |
| `JointQ_4bit_g128` | 8.89 | 0.5998 | 0.8245 | 0.7938 | 0.7269 | 0.7806 | 0.8939 | 0.7699 | 7435.3 |
| `JointQ_4bit_pc` | 9.40 | 0.5956 | 0.8266 | 0.8020 | 0.7324 | 0.7767 | 0.8930 | 0.7710 | 3923.5 |
| `JointQ_3bit_g128` | 9.91 | 0.5486 | 0.7959 | 0.7873 | 0.7214 | 0.7526 | 0.8774 | 0.7472 | 6937.9 |
| `JointQ_3bit_pc` | 18.92 | 0.5162 | 0.7504 | 0.7709 | 0.7080 | 0.7141 | 0.8621 | 0.7203 | 5184.3 |

Quantization phase elapsed time: 29003.69 s (~8h 3m).

## Notes

- Original rows report the unmodified model; quantizer rows report dequantized
  model evaluation.
- `Quant time (s)` is the quantizer's total processing time. Quantization phase
  elapsed time also includes calibration preparation and framework overhead.