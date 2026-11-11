# JointQ Benchmark Suite

This directory contains the consolidated JointQ benchmark configurations and
results. Scenario directories contain the executable scripts and detailed
reproduction instructions.

## Scenarios

| Scenario | Description |
|---|---|
| [main](main/) | Selected 4-bit and 3-bit configurations |
| [unregularization](unregularization/) | Regularization-disabled ablation |
| [actorder_mse](actorder_mse/) | Activation-ordering and GPTQ MSE ablation |
| [2bit_g32](2bit_g32/) | 2-bit, group-size-32 configuration |
| [calibration_stability](calibration_stability/) | Stability with small calibration datasets across JointQ and GPTQ |

## Models

| Model | `model_path` |
|---|---|
| Llama-2-7B | `meta-llama/Llama-2-7b-hf` |
| Llama-3-8B | `meta-llama/Meta-Llama-3-8B` |
| Qwen3-8B | `Qwen/Qwen3-8B` |
| Qwen3-14B | `Qwen/Qwen3-14B` |

## Benchmark Configuration

### Common Parameters

| Parameter | Value |
|---|---|
| Model dtype | `bfloat16` |
| Calibration samples | `1024` |
| Calibration strategy | `drop_rand` |
| Calibration sequence length | `2048` |
| Calibration seed | `0` |
| Calibration batch size | `64` (Qwen3-14B reported runs: `16`) |
| Symmetric quantization | `true` |
| Quantized layers | All (`jointq.num_layers=null`) |
| Quantization error statistics | Enabled |

Resource-only device and batch-size overrides used for the reported runs are
documented in each scenario README.

### JointQ Settings

Regularization lambdas are listed in q/k/v/o/gate/up/down order. Group size
`null` means per-channel quantization. Group swap is disabled for all settings.

| Scenario | Setting | Quantizer | Bits | Group size | Regularization lambdas | Actorder | GPTQ MSE | Retry lambda / threshold |
|---|---|---|---:|---:|---|---|---|---|
| Main | 4b/g128 | `JointQ_4bit_g128` | 4 | 128 | 0/0/0.01/0/0.01/0.01/0 | false | false | 0.01 / 0.2 |
| Main | 4b/pc | `JointQ_4bit_pc` | 4 | `null` | 0/0/0.15/0/0.15/0.15/0.01 | false | false | 0.01 / 0.2 |
| Main | 3b/g128 | `JointQ_3bit_g128` | 3 | 128 | 0.01/0.01/0.01/0/0.01/0.01/0 | false | false | 0.01 / 0.2 |
| Main | 3b/pc | `JointQ_3bit_pc` | 3 | `null` | 0.15/0.15/0.15/0.15/0.15/0.15/0.01 | false | false | 0.01 / 0.2 |
| Unregularization | 4b/g128 | `JointQ_4bit_g128_unregularized` | 4 | 128 | 0/0/0/0/0/0/0 | false | false | Disabled |
| Unregularization | 4b/pc | `JointQ_4bit_pc_unregularized` | 4 | `null` | 0/0/0/0/0/0/0 | false | false | Disabled |
| Unregularization | 3b/g128 | `JointQ_3bit_g128_unregularized` | 3 | 128 | 0/0/0/0/0/0/0 | false | false | Disabled |
| Unregularization | 3b/pc | `JointQ_3bit_pc_unregularized` | 3 | `null` | 0/0/0/0/0/0/0 | false | false | Disabled |
| Actorder + MSE | 4b/g128 | `JointQ_4bit_g128_actorder_mse` | 4 | 128 | 0/0/0.01/0/0.01/0.01/0 | true | true | 0.01 / 0.2 |
| Actorder + MSE | 3b/g128 | `JointQ_3bit_g128_actorder_mse` | 3 | 128 | 0.01/0.01/0.01/0/0.01/0.01/0 | true | true | 0.01 / 0.2 |
| 2bit/g32 | 2b/g32 | `JointQ_2bit_g32` | 2 | 32 | 0.1/0.1/0.1/0/0/0/0 | false | false | Disabled |

### Evaluation

- Perplexity: WikiText-2 test split, sequence length and stride 2048
- Accuracy: lm-eval-harness 0-shot with batch size 8; ARC Easy, ARC Challenge,
  PIQA, WinoGrande, HellaSwag, and BoolQ

Accuracy columns use normalized accuracy where available and raw accuracy for
WinoGrande and BoolQ. `Accuracy Avg.` is the arithmetic mean of all six accuracy
tasks. Original rows report the unmodified model; quantized setting rows report
dequantized model evaluation. `Quant time (s)` is the quantizer's total
processing time.

## Original Baselines

| Model | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-2-7B | 5.47 | 0.4480 | 0.7378 | 0.7845 | 0.6961 | 0.7617 | 0.7945 | 0.7038 |
| Llama-3-8B | 6.14 | 0.5384 | 0.7757 | 0.8058 | 0.7332 | 0.7921 | 0.8199 | 0.7442 |
| Qwen3-8B | 9.73 | 0.5640 | 0.8093 | 0.7758 | 0.6772 | 0.7489 | 0.8664 | 0.7403 |
| Qwen3-14B | 8.65 | 0.6041 | 0.8279 | 0.7976 | 0.7309 | 0.7877 | 0.8936 | 0.7736 |

## Main Benchmark

See [main](main/) for reproduction details.

| Model | Setting | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-2-7B | 4b/g128 | 5.67 | 0.4352 | 0.7176 | 0.7797 | 0.6961 | 0.7510 | 0.7688 | 0.6914 | 706.2 |
| Llama-2-7B | 4b/pc | 5.96 | 0.4480 | 0.7113 | 0.7791 | 0.6922 | 0.7336 | 0.7373 | 0.6836 | 1906.5 |
| Llama-2-7B | 3b/g128 | 6.78 | 0.3959 | 0.6894 | 0.7693 | 0.6882 | 0.6926 | 0.7306 | 0.6610 | 965.9 |
| Llama-2-7B | 3b/pc | 7.98 | 0.3601 | 0.5934 | 0.7405 | 0.6267 | 0.6236 | 0.6749 | 0.6032 | 2161.8 |
| Llama-3-8B | 4b/g128 | 6.65 | 0.5401 | 0.7828 | 0.7965 | 0.7293 | 0.7843 | 0.8226 | 0.7426 | 937.0 |
| Llama-3-8B | 4b/pc | 7.41 | 0.4940 | 0.7580 | 0.7829 | 0.7301 | 0.7739 | 0.8092 | 0.7247 | 2564.0 |
| Llama-3-8B | 3b/g128 | 8.75 | 0.4693 | 0.7168 | 0.7742 | 0.7395 | 0.7424 | 0.7936 | 0.7060 | 1202.8 |
| Llama-3-8B | 3b/pc | 13.09 | 0.4010 | 0.6334 | 0.7388 | 0.7056 | 0.7065 | 0.7783 | 0.6606 | 2736.9 |
| Qwen3-8B | 4b/g128 | 10.17 | 0.5572 | 0.7950 | 0.7628 | 0.6756 | 0.7393 | 0.8719 | 0.7336 | 2019.8 |
| Qwen3-8B | 4b/pc | 10.71 | 0.5367 | 0.7740 | 0.7688 | 0.6859 | 0.7323 | 0.8465 | 0.7240 | 1631.7 |
| Qwen3-8B | 3b/g128 | 11.50 | 0.5375 | 0.7950 | 0.7666 | 0.6882 | 0.6968 | 0.8661 | 0.7250 | 1997.8 |
| Qwen3-8B | 3b/pc | 22.11 | 0.4582 | 0.7104 | 0.7345 | 0.6448 | 0.6366 | 0.8269 | 0.6686 | 2236.2 |
| Qwen3-14B | 4b/g128 | 8.89 | 0.5998 | 0.8245 | 0.7938 | 0.7269 | 0.7806 | 0.8939 | 0.7699 | 7435.3 |
| Qwen3-14B | 4b/pc | 9.40 | 0.5956 | 0.8266 | 0.8020 | 0.7324 | 0.7767 | 0.8930 | 0.7710 | 3923.5 |
| Qwen3-14B | 3b/g128 | 9.91 | 0.5486 | 0.7959 | 0.7873 | 0.7214 | 0.7526 | 0.8774 | 0.7472 | 6937.9 |
| Qwen3-14B | 3b/pc | 18.92 | 0.5162 | 0.7504 | 0.7709 | 0.7080 | 0.7141 | 0.8621 | 0.7203 | 5184.3 |

## Unregularization Ablation

See [unregularization](unregularization/) for reproduction details.

| Model | Setting | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-2-7B | 4b/g128 | 5.67 | 0.4360 | 0.7252 | 0.7829 | 0.6930 | 0.7500 | 0.7505 | 0.6896 | 675.5 |
| Llama-2-7B | 4b/pc | 6.27 | 0.4258 | 0.6843 | 0.7769 | 0.6819 | 0.7253 | 0.7223 | 0.6694 | 724.0 |
| Llama-2-7B | 3b/g128 | 6.89 | 0.3985 | 0.6684 | 0.7628 | 0.6803 | 0.6781 | 0.7028 | 0.6485 | 929.9 |
| Llama-2-7B | 3b/pc | 23.02 | 0.2551 | 0.3321 | 0.5892 | 0.5201 | 0.3961 | 0.6138 | 0.4511 | 1519.6 |
| Llama-3-8B | 4b/g128 | 6.65 | 0.5179 | 0.7811 | 0.7927 | 0.7206 | 0.7790 | 0.8171 | 0.7348 | 967.5 |
| Llama-3-8B | 4b/pc | 7.59 | 0.5068 | 0.7689 | 0.7884 | 0.7190 | 0.7586 | 0.8116 | 0.7256 | 883.5 |
| Llama-3-8B | 3b/g128 | 8.90 | 0.4548 | 0.6944 | 0.7731 | 0.7143 | 0.7253 | 0.7813 | 0.6905 | 1239.7 |
| Llama-3-8B | 3b/pc | 315.46 | 0.3763 | 0.6385 | 0.7263 | 0.6898 | 0.6352 | 0.7281 | 0.6324 | 1704.2 |
| Qwen3-8B | 4b/g128 | 10.11 | 0.5401 | 0.7904 | 0.7655 | 0.6764 | 0.7421 | 0.8685 | 0.7305 | 1907.6 |
| Qwen3-8B | 4b/pc | 10.66 | 0.5341 | 0.7378 | 0.7748 | 0.6938 | 0.7240 | 0.8596 | 0.7207 | 1188.4 |
| Qwen3-8B | 3b/g128 | 11.54 | 0.5239 | 0.7715 | 0.7622 | 0.6938 | 0.6970 | 0.8630 | 0.7186 | 1809.6 |
| Qwen3-8B | 3b/pc | 30.19 | 0.4599 | 0.7045 | 0.7459 | 0.6654 | 0.6273 | 0.8150 | 0.6697 | 1691.3 |
| Qwen3-14B | 4b/g128 | 8.88 | 0.6109 | 0.8266 | 0.7965 | 0.7159 | 0.7813 | 0.8939 | 0.7709 | 6839.2 |
| Qwen3-14B | 4b/pc | 9.41 | 0.6126 | 0.8270 | 0.7943 | 0.7167 | 0.7674 | 0.8927 | 0.7684 | 3228.0 |
| Qwen3-14B | 3b/g128 | 9.91 | 0.5538 | 0.7967 | 0.7927 | 0.7230 | 0.7481 | 0.8731 | 0.7479 | 6020.7 |
| Qwen3-14B | 3b/pc | 20.20 | 0.5213 | 0.7698 | 0.7720 | 0.7119 | 0.7147 | 0.8706 | 0.7267 | 4317.4 |

## Actorder and MSE Ablation

See [actorder_mse](actorder_mse/) for reproduction details.

| Model | Setting | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-2-7B | 4b/g128 | 5.66 | 0.4403 | 0.7201 | 0.7813 | 0.6946 | 0.7511 | 0.7887 | 0.6960 | 1805.4 |
| Llama-2-7B | 3b/g128 | 6.56 | 0.4164 | 0.6877 | 0.7704 | 0.6740 | 0.7046 | 0.7303 | 0.6639 | 2226.3 |
| Llama-3-8B | 4b/g128 | 6.55 | 0.5375 | 0.7908 | 0.7982 | 0.7356 | 0.7841 | 0.8174 | 0.7439 | 1913.8 |
| Llama-3-8B | 3b/g128 | 8.41 | 0.5043 | 0.7466 | 0.7731 | 0.7222 | 0.7530 | 0.8156 | 0.7191 | 2341.4 |
| Qwen3-8B | 4b/g128 | 9.88 | 0.5529 | 0.8009 | 0.7748 | 0.6898 | 0.7442 | 0.8664 | 0.7382 | 2089.2 |
| Qwen3-8B | 3b/g128 | 11.58 | 0.5341 | 0.7551 | 0.7650 | 0.6914 | 0.7101 | 0.8602 | 0.7193 | 2549.3 |
| Qwen3-14B | 4b/g128 | 8.93 | 0.6118 | 0.8321 | 0.7987 | 0.7293 | 0.7837 | 0.8893 | 0.7741 | 3747.3 |
| Qwen3-14B | 3b/g128 | 9.48 | 0.5819 | 0.8089 | 0.7867 | 0.7245 | 0.7598 | 0.8746 | 0.7561 | 4499.6 |

## 2-bit / Group-size-32

See [2bit_g32](2bit_g32/) for reproduction details.

| Model | Setting | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-2-7B | 2b/g32 | 26.33 | 0.2483 | 0.4474 | 0.6415 | 0.5683 | 0.4359 | 0.6147 | 0.4927 | 1164.1 |
| Llama-3-8B | 2b/g32 | 108.77 | 0.2560 | 0.4339 | 0.6349 | 0.5683 | 0.4054 | 0.6211 | 0.4866 | 1330.4 |
| Qwen3-8B | 2b/g32 | 45.98 | 0.3200 | 0.5896 | 0.6861 | 0.6038 | 0.4885 | 0.7651 | 0.5755 | 1971.2 |
| Qwen3-14B | 2b/g32 | 20.78 | 0.4087 | 0.6654 | 0.7182 | 0.6559 | 0.5805 | 0.6911 | 0.6200 | 6219.7 |
