# Qwen3-8B JointQ Benchmark

JointQ benchmark for [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) using OneComp v1.1.0.

All combinations of `bits × group_size` are run in a single pass, sharing calibration data accumulation across quantizers for efficiency.

Twenty-seven configurations are benchmarked:

1. **JointQ (incremental)** — `lambda_mode="incremental_lambda"`, `lambda_list=[0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]`
2. **JointQ (fixed, λ=0.0)** — `lambda_mode="fixed_lambda"`, `regularization_lambda=0.0`
3. **JointQ (fixed, λ=0.001)** — `lambda_mode="fixed_lambda"`, `regularization_lambda=0.001`
4. **JointQ (fixed, λ=0.01)** — `lambda_mode="fixed_lambda"`, `regularization_lambda=0.01`
5. **JointQ (fixed, λ=0.05)** — `lambda_mode="fixed_lambda"`, `regularization_lambda=0.05`
6. **JointQ (fixed, λ=0.1)** — `lambda_mode="fixed_lambda"`, `regularization_lambda=0.1`
7. **JointQ (fixed, λ=0.15)** — `lambda_mode="fixed_lambda"`, `regularization_lambda=0.15`
8. **JointQ (fixed, λ=0.2)** — `lambda_mode="fixed_lambda"`, `regularization_lambda=0.2`
9. **JointQ (fixed, λ=0.3)** — `lambda_mode="fixed_lambda"`, `regularization_lambda=0.3`
10. **JointQ (fixed, λ=0.5)** — `lambda_mode="fixed_lambda"`, `regularization_lambda=0.5`
11. **JointQ (incremental+actorder+mse)** — `lambda_mode="incremental_lambda"`, `actorder=true`, `gptq_mse=true`
12. **JointQ (fixed, λ=0.0, actorder+mse)** — `lambda_mode="fixed_lambda"`, `regularization_lambda=0.0`, `actorder=true`, `gptq_mse=true`
13. **JointQ (fixed, λ=0.001, actorder+mse)** — `lambda_mode="fixed_lambda"`, `regularization_lambda=0.001`, `actorder=true`, `gptq_mse=true`
14. **JointQ (fixed, λ=0.01, actorder+mse)** — `lambda_mode="fixed_lambda"`, `regularization_lambda=0.01`, `actorder=true`, `gptq_mse=true`
15. **JointQ (diagonal, λ=0.0)** — `lambda_mode="fixed_lambda"`, `regularization_mode="diagonal"`, `regularization_lambda=0.0`
16. **JointQ (diagonal, λ=0.001)** — `lambda_mode="fixed_lambda"`, `regularization_mode="diagonal"`, `regularization_lambda=0.001`
17. **JointQ (diagonal, λ=0.01)** — `lambda_mode="fixed_lambda"`, `regularization_mode="diagonal"`, `regularization_lambda=0.01`
18. **JointQ (diagonal, λ=0.05)** — `lambda_mode="fixed_lambda"`, `regularization_mode="diagonal"`, `regularization_lambda=0.05`
19. **JointQ (diagonal, λ=0.1)** — `lambda_mode="fixed_lambda"`, `regularization_mode="diagonal"`, `regularization_lambda=0.1`
20. **JointQ (diagonal, λ=0.15)** — `lambda_mode="fixed_lambda"`, `regularization_mode="diagonal"`, `regularization_lambda=0.15`
21. **JointQ (diagonal, λ=0.2)** — `lambda_mode="fixed_lambda"`, `regularization_mode="diagonal"`, `regularization_lambda=0.2`
22. **JointQ (diagonal, λ=0.3)** — `lambda_mode="fixed_lambda"`, `regularization_mode="diagonal"`, `regularization_lambda=0.3`
23. **JointQ (diagonal, λ=0.5)** — `lambda_mode="fixed_lambda"`, `regularization_mode="diagonal"`, `regularization_lambda=0.5`
24. **JointQ (diagonal, λ=0.01, actorder+mse)** — `lambda_mode="fixed_lambda"`, `regularization_mode="diagonal"`, `regularization_lambda=0.01`, `actorder=true`, `gptq_mse=true`
25. **JointQ (diagonal, λ=0.05, actorder+mse)** — `lambda_mode="fixed_lambda"`, `regularization_mode="diagonal"`, `regularization_lambda=0.05`, `actorder=true`, `gptq_mse=true`
26. **JointQ (diagonal, λ=0.1, actorder+mse)** — `lambda_mode="fixed_lambda"`, `regularization_mode="diagonal"`, `regularization_lambda=0.1`, `actorder=true`, `gptq_mse=true`
27. **JointQ (diagonal, λ=0.15, actorder+mse)** — `lambda_mode="fixed_lambda"`, `regularization_mode="diagonal"`, `regularization_lambda=0.15`, `actorder=true`, `gptq_mse=true`

## Benchmark Configuration

### Common Parameters

| Parameter | Values |
|---|---|
| bits | 4, 3 |
| group_size | 128, null (per-channel) |
| symmetric | true |
| num_calibration_samples | 1024 |
| calibration_strategy | drop_rand |
| max_length | 2048 |

This produces **4 quantizers** (2 bits × 2 group sizes) per configuration.

### Configuration-Specific Parameters

| Parameter | incremental | fixed λ=0.0 | fixed λ=0.001 | fixed λ=0.01 | fixed λ=0.05 | fixed λ=0.1 | fixed λ=0.15 | fixed λ=0.2 | fixed λ=0.3 | fixed λ=0.5 | incremental+actorder+mse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lambda_mode | incremental_lambda | fixed_lambda | fixed_lambda | fixed_lambda | fixed_lambda | fixed_lambda | fixed_lambda | fixed_lambda | fixed_lambda | fixed_lambda | incremental_lambda |
| regularization_lambda | — | 0.0 | 0.001 | 0.01 | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.5 | — |
| lambda_list | [0.001, …, 0.5] | — | — | — | — | — | — | — | — | — | [0.001, …, 0.5] |
| actorder | false | false | false | false | false | false | false | false | false | false | true |
| gptq_mse | false | false | false | false | false | false | false | false | false | false | true |

### Evaluation

- Perplexity (WikiText-2)
- Accuracy (lm-eval-harness)

Both are computed for the original (unquantized) model and all dequantized models.

## Usage

Requires [Hydra](https://hydra.cc/) (see [benchmark/README.md](../README.md) for installation).

```bash
# 1. incremental
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=incremental_lambda \
    output_dir=qwen3-8b-incremental

# 2. fixed, lambda=0.0
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.0 \
    output_dir=qwen3-8b-fixed-lam0.0

# 3. fixed, lambda=0.001
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.001 \
    output_dir=qwen3-8b-fixed-lam0.001

# 4. fixed, lambda=0.01
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.01 \
    output_dir=qwen3-8b-fixed-lam0.01

# 5. fixed, lambda=0.05
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.05 \
    output_dir=qwen3-8b-fixed-lam0.05

# 6. fixed, lambda=0.1
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.1 \
    output_dir=qwen3-8b-fixed-lam0.1

# 7. fixed, lambda=0.15
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.15 \
    output_dir=qwen3-8b-fixed-lam0.15

# 8. fixed, lambda=0.2
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.2 \
    output_dir=qwen3-8b-fixed-lam0.2

# 9. fixed, lambda=0.3
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.3 \
    output_dir=qwen3-8b-fixed-lam0.3

# 10. fixed, lambda=0.5
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.5 \
    output_dir=qwen3-8b-fixed-lam0.5

# 11. incremental+actorder+mse
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=incremental_lambda \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=qwen3-8b-incremental-actorder-mse

# 12. fixed, lambda=0.0, actorder+mse
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.0 \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=qwen3-8b-fixed-lam0.0-actorder

# 13. fixed, lambda=0.001, actorder+mse
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.001 \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=qwen3-8b-fixed-lam0.001-actorder

# 14. fixed, lambda=0.01, actorder+mse
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.01 \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=qwen3-8b-fixed-lam0.01-actorder

# 15. diagonal, lambda=0.0
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.0 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-8b-diagonal-lam0.0

# 16. diagonal, lambda=0.001
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.001 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-8b-diagonal-lam0.001

# 17. diagonal, lambda=0.01
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.01 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-8b-diagonal-lam0.01

# 18. diagonal, lambda=0.05
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.05 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-8b-diagonal-lam0.05

# 19. diagonal, lambda=0.1
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.1 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-8b-diagonal-lam0.1

# 20. diagonal, lambda=0.15
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.15 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-8b-diagonal-lam0.15

# 21. diagonal, lambda=0.2
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.2 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-8b-diagonal-lam0.2

# 22. diagonal, lambda=0.3
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.3 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-8b-diagonal-lam0.3

# 23. diagonal, lambda=0.5
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.5 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-8b-diagonal-lam0.5

# 24. diagonal, lambda=0.01, actorder+mse
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.01 \
    jointq.regularization_mode=diagonal \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=qwen3-8b-diagonal-lam0.01-actorder

# 25. diagonal, lambda=0.05, actorder+mse
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.05 \
    jointq.regularization_mode=diagonal \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=qwen3-8b-diagonal-lam0.05-actorder

# 26. diagonal, lambda=0.1, actorder+mse
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.1 \
    jointq.regularization_mode=diagonal \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=qwen3-8b-diagonal-lam0.1-actorder

# 27. diagonal, lambda=0.15, actorder+mse
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.15 \
    jointq.regularization_mode=diagonal \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=qwen3-8b-diagonal-lam0.15-actorder
```

### Hydra Overrides

You can override any parameter from the command line:

```bash
# Run only 4-bit
python quant_benchmark.py model_path=/path/to/model 'jointq.bits=[4]'

# Change calibration samples
python quant_benchmark.py model_path=/path/to/model num_calibration_samples=512
```

## Results

PPL = perplexity on WikiText-2 (lower is better). Accuracy = 0-shot `acc_norm` where available, `acc` otherwise (winogrande) (higher is better).

### JointQ (incremental)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| 4 | 128 | 10.10 | 0.5486 | 0.7942 | 0.7682 | 0.6859 | 2026.1 |
| 4 | per-channel | 10.89 | 0.5316 | 0.7517 | 0.7655 | 0.6654 | 6956.0 |
| 3 | 128 | 11.65 | 0.5435 | 0.7715 | 0.7677 | 0.6938 | 3846.0 |
| 3 | per-channel | 40.53 | 0.3456 | 0.5547 | 0.7008 | 0.6204 | 8943.6 |

Total elapsed time (including calibration data preparation): 24723.5 s (~6h 52m).

### JointQ (fixed, λ=0.0)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.10 | 0.5512 | 0.7988 | 0.7682 | 0.6867 | 1759.5 |
| 4 | per-channel | 10.71 | 0.5068 | 0.7395 | 0.7688 | 0.6819 | 1098.7 |
| 3 | 128 | 11.39 | 0.5205 | 0.7774 | 0.7655 | 0.6669 | 1690.6 |
| 3 | per-channel | 28.24 | 0.4522 | 0.6940 | 0.7437 | 0.6606 | 1587.9 |

Total elapsed time (including calibration data preparation): 9110.6 s (~2h 31m).

### JointQ (fixed, λ=0.001)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.09 | 0.5469 | 0.7929 | 0.7709 | 0.6890 | 916.2 |
| 4 | per-channel | 10.74 | 0.5307 | 0.7622 | 0.7715 | 0.7001 | 912.6 |
| 3 | 128 | 11.42 | 0.5273 | 0.7820 | 0.7590 | 0.6677 | 1255.0 |
| 3 | per-channel | 28.65 | 0.4206 | 0.6524 | 0.7416 | 0.6811 | 1613.8 |

Total elapsed time (including calibration data preparation): 7743.5 s (~2h 9m).

### JointQ (fixed, λ=0.01)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.09 | 0.5478 | 0.7879 | 0.7699 | 0.6930 | 1001.9 |
| 4 | per-channel | 10.63 | 0.5128 | 0.7534 | 0.7682 | 0.6803 | 1429.3 |
| 3 | 128 | 11.62 | 0.5324 | 0.7896 | 0.7590 | 0.6835 | 1401.5 |
| 3 | per-channel | 28.85 | 0.4010 | 0.6292 | 0.7361 | 0.6511 | 1888.3 |

Total elapsed time (including calibration data preparation): 8716.0 s (~2h 25m).

### JointQ (fixed, λ=0.05)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.19 | 0.5520 | 0.7841 | 0.7688 | 0.6875 | 1004.2 |
| 4 | per-channel | 10.88 | 0.5333 | 0.7437 | 0.7682 | 0.6709 | 1827.4 |
| 3 | 128 | 11.93 | 0.5137 | 0.7563 | 0.7639 | 0.6859 | 1496.3 |
| 3 | per-channel | 33.58 | 0.3532 | 0.5644 | 0.7008 | 0.5919 | 2173.8 |

Total elapsed time (including calibration data preparation): 9460.9 s (~2h 37m).

### JointQ (fixed, λ=0.1)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.23 | 0.5469 | 0.7795 | 0.7644 | 0.6946 | 940.4 |
| 4 | per-channel | 10.97 | 0.4974 | 0.7235 | 0.7595 | 0.6590 | 2030.1 |
| 3 | 128 | 11.80 | 0.5392 | 0.7774 | 0.7590 | 0.7017 | 1434.9 |
| 3 | per-channel | 41.71 | 0.3114 | 0.5139 | 0.6850 | 0.5533 | 2248.2 |

Total elapsed time (including calibration data preparation): 9607.8 s (~2h 40m).

### JointQ (fixed, λ=0.15)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.25 | 0.5444 | 0.7740 | 0.7622 | 0.6669 | 957.7 |
| 4 | per-channel | 11.16 | 0.5017 | 0.7483 | 0.7584 | 0.6385 | 2207.7 |
| 3 | 128 | 11.99 | 0.5333 | 0.7748 | 0.7552 | 0.6906 | 1463.7 |
| 3 | per-channel | 41.37 | 0.3131 | 0.5034 | 0.6785 | 0.5714 | 2368.2 |

Total elapsed time (including calibration data preparation): 9912.5 s (~2h 45m).

### JointQ (fixed, λ=0.2)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.23 | 0.5384 | 0.7715 | 0.7650 | 0.6732 | 931.8 |
| 4 | per-channel | 11.39 | 0.4940 | 0.7475 | 0.7590 | 0.6267 | 2314.6 |
| 3 | 128 | 12.03 | 0.5222 | 0.7635 | 0.7601 | 0.6882 | 1463.0 |
| 3 | per-channel | 45.85 | 0.3080 | 0.5055 | 0.6719 | 0.5699 | 2429.0 |

Total elapsed time (including calibration data preparation): 10110.3 s (~2h 48m).

### JointQ (fixed, λ=0.3)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.26 | 0.5418 | 0.7723 | 0.7666 | 0.6732 | 850.8 |
| 4 | per-channel | 11.45 | 0.4795 | 0.7382 | 0.7557 | 0.6377 | 2357.8 |
| 3 | 128 | 12.09 | 0.5111 | 0.7504 | 0.7541 | 0.6646 | 1339.4 |
| 3 | per-channel | 52.72 | 0.2901 | 0.4832 | 0.6556 | 0.5533 | 2398.7 |

Total elapsed time (including calibration data preparation): 9892.0 s (~2h 44m).

### JointQ (fixed, λ=0.5)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.32 | 0.5375 | 0.7609 | 0.7699 | 0.6654 | 817.1 |
| 4 | per-channel | 11.56 | 0.4846 | 0.7357 | 0.7568 | 0.6472 | 2554.7 |
| 3 | 128 | 12.08 | 0.4974 | 0.7294 | 0.7497 | 0.6638 | 1294.6 |
| 3 | per-channel | 56.57 | 0.2978 | 0.4630 | 0.6523 | 0.5359 | 2495.3 |

Total elapsed time (including calibration data preparation): 10075.3 s (~2h 47m).

### JointQ (incremental+actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 9.92 | 0.5486 | 0.8056 | 0.7813 | 0.6882 | 2180.1 |
| 4 | per-channel | 10.99 | 0.5273 | 0.7736 | 0.7704 | 0.6732 | 3603.3 |
| 3 | 128 | 11.84 | 0.5213 | 0.7706 | 0.7655 | 0.6882 | 2903.6 |
| 3 | per-channel | 31.55 | 0.3370 | 0.4933 | 0.6997 | 0.5951 | 5539.2 |

Total elapsed time (including calibration data preparation): 17150.6 s (~4h 46m).

### JointQ (fixed, λ=0.0, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 9.91 | 0.5418 | 0.8026 | 0.7797 | 0.6890 | 1763.7 |
| 4 | per-channel | 10.80 | 0.5435 | 0.7921 | 0.7753 | 0.6796 | 1670.5 |
| 3 | 128 | 11.23 | 0.5162 | 0.7681 | 0.7606 | 0.6914 | 2102.7 |
| 3 | per-channel | 26.46 | 0.3439 | 0.5328 | 0.7046 | 0.5967 | 2044.3 |

Total elapsed time (including calibration data preparation): 10449.4 s (~2h 54m).

### JointQ (fixed, λ=0.001, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 9.91 | 0.5410 | 0.8043 | 0.7797 | 0.6867 | 1722.3 |
| 4 | per-channel | 10.89 | 0.5256 | 0.7765 | 0.7715 | 0.6788 | 1146.4 |
| 3 | 128 | 11.80 | 0.5230 | 0.7698 | 0.7601 | 0.6867 | 2142.4 |
| 3 | per-channel | 30.91 | 0.3345 | 0.4903 | 0.6975 | 0.5991 | 1848.1 |

Total elapsed time (including calibration data preparation): 9742.6 s (~2h 42m).

### JointQ (fixed, λ=0.01, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 9.92 | 0.5435 | 0.8009 | 0.7797 | 0.6811 | 1826.1 |
| 4 | per-channel | 11.14 | 0.5401 | 0.7719 | 0.7720 | 0.6693 | 1656.0 |
| 3 | 128 | 12.13 | 0.5085 | 0.7635 | 0.7579 | 0.6882 | 2301.2 |
| 3 | per-channel | 30.99 | 0.3268 | 0.4949 | 0.7029 | 0.6109 | 2097.4 |

Total elapsed time (including calibration data preparation): 10825.7 s (~3h 0m).

### JointQ (diagonal, λ=0.0)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.10 | 0.5512 | 0.7988 | 0.7682 | 0.6867 | 1890.0 |
| 4 | per-channel | 10.71 | 0.5068 | 0.7395 | 0.7688 | 0.6819 | 1162.0 |
| 3 | 128 | 11.39 | 0.5205 | 0.7774 | 0.7655 | 0.6669 | 1811.0 |
| 3 | per-channel | 28.24 | 0.4522 | 0.6940 | 0.7437 | 0.6606 | 1666.8 |

Total elapsed time (including calibration data preparation): 9387.7 s (~2h 36m).

### JointQ (diagonal, λ=0.001)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.06 | 0.5546 | 0.7904 | 0.7699 | 0.6803 | 977.0 |
| 4 | per-channel | 10.63 | 0.5299 | 0.7710 | 0.7726 | 0.6922 | 843.5 |
| 3 | 128 | 11.58 | 0.5256 | 0.7795 | 0.7617 | 0.6661 | 1283.2 |
| 3 | per-channel | 27.69 | 0.4488 | 0.6793 | 0.7356 | 0.6875 | 1542.5 |

Total elapsed time (including calibration data preparation): 7524.9 s (~2h 5m).

### JointQ (diagonal, λ=0.01)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.09 | 0.5486 | 0.7845 | 0.7693 | 0.6811 | 947.4 |
| 4 | per-channel | 10.62 | 0.5324 | 0.7614 | 0.7704 | 0.6843 | 1262.4 |
| 3 | 128 | 11.53 | 0.5239 | 0.7807 | 0.7595 | 0.6811 | 1313.2 |
| 3 | per-channel | 29.58 | 0.4309 | 0.6633 | 0.7519 | 0.6614 | 1749.2 |

Total elapsed time (including calibration data preparation): 8210.8 s (~2h 16m).

### JointQ (diagonal, λ=0.05)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.17 | 0.5367 | 0.7795 | 0.7682 | 0.6985 | 982.5 |
| 4 | per-channel | 10.60 | 0.5273 | 0.7370 | 0.7758 | 0.6867 | 1623.7 |
| 3 | 128 | 11.82 | 0.5341 | 0.7723 | 0.7568 | 0.6946 | 1443.1 |
| 3 | per-channel | 26.79 | 0.4471 | 0.6734 | 0.7372 | 0.6567 | 2028.9 |

Total elapsed time (including calibration data preparation): 9016.6 s (~2h 30m).

### JointQ (diagonal, λ=0.1)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.15 | 0.5392 | 0.7744 | 0.7682 | 0.6882 | 1032.1 |
| 4 | per-channel | 10.71 | 0.5333 | 0.7652 | 0.7720 | 0.6961 | 1931.8 |
| 3 | 128 | 11.73 | 0.5154 | 0.7601 | 0.7563 | 0.6867 | 1546.6 |
| 3 | per-channel | 23.97 | 0.4420 | 0.6709 | 0.7383 | 0.6425 | 2227.4 |

Total elapsed time (including calibration data preparation): 9604.5 s (~2h 40m).

### JointQ (diagonal, λ=0.15)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.18 | 0.5401 | 0.7761 | 0.7655 | 0.6969 | 982.9 |
| 4 | per-channel | 10.81 | 0.5367 | 0.7685 | 0.7699 | 0.6764 | 2011.3 |
| 3 | 128 | 11.68 | 0.5478 | 0.7984 | 0.7612 | 0.6788 | 1482.3 |
| 3 | per-channel | 24.15 | 0.4386 | 0.6646 | 0.7296 | 0.6259 | 2240.9 |

Total elapsed time (including calibration data preparation): 9600.6 s (~2h 40m).

### JointQ (diagonal, λ=0.2)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.18 | 0.5452 | 0.7765 | 0.7682 | 0.6867 | 936.9 |
| 4 | per-channel | 10.68 | 0.5299 | 0.7723 | 0.7677 | 0.6811 | 2071.5 |
| 3 | 128 | 11.79 | 0.5580 | 0.7866 | 0.7622 | 0.6717 | 1435.6 |
| 3 | per-channel | 25.03 | 0.4360 | 0.6763 | 0.7345 | 0.6433 | 2263.5 |

Total elapsed time (including calibration data preparation): 9563.4 s (~2h 39m).

### JointQ (diagonal, λ=0.3)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.22 | 0.5538 | 0.7774 | 0.7655 | 0.6788 | 950.1 |
| 4 | per-channel | 10.79 | 0.5410 | 0.7740 | 0.7655 | 0.6875 | 2262.1 |
| 3 | 128 | 11.84 | 0.5461 | 0.7858 | 0.7644 | 0.6772 | 1482.2 |
| 3 | per-channel | 25.47 | 0.4360 | 0.6785 | 0.7285 | 0.6290 | 2385.9 |

Total elapsed time (including calibration data preparation): 10053.6 s (~2h 47m).

### JointQ (diagonal, λ=0.5)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.25 | 0.5520 | 0.7710 | 0.7688 | 0.6725 | 910.7 |
| 4 | per-channel | 10.67 | 0.5401 | 0.7727 | 0.7639 | 0.6835 | 2403.6 |
| 3 | 128 | 11.91 | 0.5580 | 0.7879 | 0.7639 | 0.6646 | 1426.0 |
| 3 | per-channel | 24.83 | 0.4266 | 0.6810 | 0.7187 | 0.6219 | 2433.7 |

Total elapsed time (including calibration data preparation): 10145.6 s (~2h 49m).

### JointQ (diagonal, λ=0.01, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 9.93 | 0.5461 | 0.7976 | 0.7835 | 0.6875 | 2136.2 |
| 4 | per-channel | 11.05 | 0.5478 | 0.7799 | 0.7688 | 0.6764 | 1672.5 |
| 3 | 128 | 11.89 | 0.5154 | 0.7614 | 0.7639 | 0.6843 | 2664.9 |
| 3 | per-channel | 29.77 | 0.3652 | 0.5366 | 0.6997 | 0.5896 | 2158.6 |

Total elapsed time (including calibration data preparation): 11656.0 s (~3h 14m).

### JointQ (diagonal, λ=0.05, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 9.96 | 0.5461 | 0.7997 | 0.7742 | 0.6867 | 1960.1 |
| 4 | per-channel | 11.05 | 0.5435 | 0.7896 | 0.7786 | 0.6867 | 1929.0 |
| 3 | 128 | 11.86 | 0.5111 | 0.7576 | 0.7617 | 0.6772 | 2447.0 |
| 3 | per-channel | 30.34 | 0.3763 | 0.5783 | 0.7296 | 0.6117 | 2322.3 |

Total elapsed time (including calibration data preparation): 11530.4 s (~3h 12m).

### JointQ (diagonal, λ=0.1, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 9.93 | 0.5435 | 0.7997 | 0.7758 | 0.6890 | 1979.9 |
| 4 | per-channel | 10.98 | 0.5324 | 0.7845 | 0.7693 | 0.6851 | 2165.9 |
| 3 | 128 | 12.08 | 0.5196 | 0.7546 | 0.7579 | 0.6756 | 2455.2 |
| 3 | per-channel | 28.91 | 0.3942 | 0.5875 | 0.7301 | 0.6014 | 2406.3 |

Total elapsed time (including calibration data preparation): 11893.0 s (~3h 18m).

### JointQ (diagonal, λ=0.15, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 9.91 | 0.5461 | 0.7992 | 0.7775 | 0.6946 | 1998.9 |
| 4 | per-channel | 10.91 | 0.5427 | 0.7862 | 0.7671 | 0.6567 | 2302.1 |
| 3 | 128 | 12.10 | 0.5085 | 0.7559 | 0.7655 | 0.6914 | 2487.8 |
| 3 | per-channel | 29.06 | 0.3951 | 0.5758 | 0.7318 | 0.6172 | 2485.1 |

Total elapsed time (including calibration data preparation): 12189.5 s (~3h 23m).

### Cross-configuration comparison

Bold values indicate the top-3 results in each column (lower is better for PPL and Time; higher is better for accuracy).

#### 4-bit, group_size=128 — fixed (identity)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| incremental | **10.10** | **0.5486** | **0.7942** | 0.7682 | 0.6859 | 2026.1 |
| fixed λ=0.0 | 10.10 | **0.5512** | **0.7988** | 0.7682 | 0.6867 | 1759.5 |
| fixed λ=0.001 | **10.09** | 0.5469 | **0.7929** | **0.7709** | **0.6890** | **916.2** |
| fixed λ=0.01 | **10.09** | 0.5478 | 0.7879 | **0.7699** | **0.6930** | 1001.9 |
| fixed λ=0.05 | 10.19 | **0.5520** | 0.7841 | 0.7688 | 0.6875 | 1004.2 |
| fixed λ=0.1 | 10.23 | 0.5469 | 0.7795 | 0.7644 | **0.6946** | 940.4 |
| fixed λ=0.15 | 10.25 | 0.5444 | 0.7740 | 0.7622 | 0.6669 | 957.7 |
| fixed λ=0.2 | 10.23 | 0.5384 | 0.7715 | 0.7650 | 0.6732 | 931.8 |
| fixed λ=0.3 | 10.26 | 0.5418 | 0.7723 | 0.7666 | 0.6732 | **850.8** |
| fixed λ=0.5 | 10.32 | 0.5375 | 0.7609 | **0.7699** | 0.6654 | **817.1** |

#### 4-bit, group_size=128 — diagonal

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| diagonal λ=0.0 | **10.10** | 0.5512 | **0.7988** | 0.7682 | 0.6867 | 1890.0 |
| diagonal λ=0.001 | **10.06** | **0.5546** | **0.7904** | **0.7699** | 0.6803 | 977.0 |
| diagonal λ=0.01 | **10.09** | 0.5486 | **0.7845** | **0.7693** | 0.6811 | **947.4** |
| diagonal λ=0.05 | 10.17 | 0.5367 | 0.7795 | 0.7682 | **0.6985** | 982.5 |
| diagonal λ=0.1 | 10.15 | 0.5392 | 0.7744 | 0.7682 | **0.6882** | 1032.1 |
| diagonal λ=0.15 | 10.18 | 0.5401 | 0.7761 | 0.7655 | **0.6969** | 982.9 |
| diagonal λ=0.2 | 10.18 | 0.5452 | 0.7765 | 0.7682 | 0.6867 | **936.9** |
| diagonal λ=0.3 | 10.22 | **0.5538** | 0.7774 | 0.7655 | 0.6788 | 950.1 |
| diagonal λ=0.5 | 10.25 | **0.5520** | 0.7710 | **0.7688** | 0.6725 | **910.7** |

#### 4-bit, group_size=128 — actorder+mse

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| incremental+actorder+mse | 9.92 | **0.5486** | **0.8056** | **0.7813** | 0.6882 | 2180.1 |
| fixed λ=0.0 actorder+mse | **9.91** | 0.5418 | **0.8026** | **0.7797** | **0.6890** | **1763.7** |
| fixed λ=0.001 actorder+mse | **9.91** | 0.5410 | **0.8043** | 0.7797 | 0.6867 | **1722.3** |
| fixed λ=0.01 actorder+mse | 9.92 | 0.5435 | 0.8009 | 0.7797 | 0.6811 | **1826.1** |
| diagonal λ=0.01 actorder+mse | 9.93 | **0.5461** | 0.7976 | **0.7835** | 0.6875 | 2136.2 |
| diagonal λ=0.05 actorder+mse | 9.96 | **0.5461** | 0.7997 | 0.7742 | 0.6867 | 1960.1 |
| diagonal λ=0.1 actorder+mse | 9.93 | 0.5435 | 0.7997 | 0.7758 | **0.6890** | 1979.9 |
| diagonal λ=0.15 actorder+mse | **9.91** | 0.5461 | 0.7992 | 0.7775 | **0.6946** | 1998.9 |

#### 4-bit, per-channel — fixed (identity)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| incremental | 10.89 | **0.5316** | **0.7517** | 0.7655 | 0.6654 | 6956.0 |
| fixed λ=0.0 | **10.71** | 0.5068 | 0.7395 | **0.7688** | **0.6819** | **1098.7** |
| fixed λ=0.001 | **10.74** | **0.5307** | **0.7622** | **0.7715** | **0.7001** | **912.6** |
| fixed λ=0.01 | **10.63** | 0.5128 | **0.7534** | **0.7682** | **0.6803** | **1429.3** |
| fixed λ=0.05 | 10.88 | **0.5333** | 0.7437 | 0.7682 | 0.6709 | 1827.4 |
| fixed λ=0.1 | 10.97 | 0.4974 | 0.7235 | 0.7595 | 0.6590 | 2030.1 |
| fixed λ=0.15 | 11.16 | 0.5017 | 0.7483 | 0.7584 | 0.6385 | 2207.7 |
| fixed λ=0.2 | 11.39 | 0.4940 | 0.7475 | 0.7590 | 0.6267 | 2314.6 |
| fixed λ=0.3 | 11.45 | 0.4795 | 0.7382 | 0.7557 | 0.6377 | 2357.8 |
| fixed λ=0.5 | 11.56 | 0.4846 | 0.7357 | 0.7568 | 0.6472 | 2554.7 |

#### 4-bit, per-channel — diagonal

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| diagonal λ=0.0 | 10.71 | 0.5068 | 0.7395 | 0.7688 | 0.6819 | **1162.0** |
| diagonal λ=0.001 | **10.63** | 0.5299 | 0.7710 | **0.7726** | **0.6922** | **843.5** |
| diagonal λ=0.01 | **10.62** | 0.5324 | 0.7614 | 0.7704 | 0.6843 | **1262.4** |
| diagonal λ=0.05 | **10.60** | 0.5273 | 0.7370 | **0.7758** | 0.6867 | 1623.7 |
| diagonal λ=0.1 | 10.71 | 0.5333 | 0.7652 | **0.7720** | **0.6961** | 1931.8 |
| diagonal λ=0.15 | 10.81 | **0.5367** | 0.7685 | 0.7699 | 0.6764 | 2011.3 |
| diagonal λ=0.2 | 10.68 | 0.5299 | **0.7723** | 0.7677 | 0.6811 | 2071.5 |
| diagonal λ=0.3 | 10.79 | **0.5410** | **0.7740** | 0.7655 | **0.6875** | 2262.1 |
| diagonal λ=0.5 | 10.67 | **0.5401** | **0.7727** | 0.7639 | 0.6835 | 2403.6 |

#### 4-bit, per-channel — actorder+mse

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| incremental+actorder+mse | 10.99 | 0.5273 | 0.7736 | 0.7704 | 0.6732 | 3603.3 |
| fixed λ=0.0 actorder+mse | **10.80** | **0.5435** | **0.7921** | **0.7753** | **0.6796** | **1670.5** |
| fixed λ=0.001 actorder+mse | **10.89** | 0.5256 | 0.7765 | 0.7715 | 0.6788 | **1146.4** |
| fixed λ=0.01 actorder+mse | 11.14 | 0.5401 | 0.7719 | **0.7720** | 0.6693 | **1656.0** |
| diagonal λ=0.01 actorder+mse | 11.05 | **0.5478** | 0.7799 | 0.7688 | 0.6764 | 1672.5 |
| diagonal λ=0.05 actorder+mse | 11.05 | **0.5435** | **0.7896** | **0.7786** | **0.6867** | 1929.0 |
| diagonal λ=0.1 actorder+mse | 10.98 | 0.5324 | 0.7845 | 0.7693 | **0.6851** | 2165.9 |
| diagonal λ=0.15 actorder+mse | **10.91** | 0.5427 | **0.7862** | 0.7671 | 0.6567 | 2302.1 |

#### 3-bit, group_size=128 — fixed (identity)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| incremental | 11.65 | **0.5435** | 0.7715 | **0.7677** | **0.6938** | 3846.0 |
| fixed λ=0.0 | **11.39** | 0.5205 | **0.7774** | **0.7655** | 0.6669 | 1690.6 |
| fixed λ=0.001 | **11.42** | 0.5273 | **0.7820** | 0.7590 | 0.6677 | **1255.0** |
| fixed λ=0.01 | **11.62** | 0.5324 | **0.7896** | 0.7590 | 0.6835 | 1401.5 |
| fixed λ=0.05 | 11.93 | 0.5137 | 0.7563 | **0.7639** | 0.6859 | 1496.3 |
| fixed λ=0.1 | 11.80 | **0.5392** | 0.7774 | 0.7590 | **0.7017** | 1434.9 |
| fixed λ=0.15 | 11.99 | **0.5333** | 0.7748 | 0.7552 | **0.6906** | 1463.7 |
| fixed λ=0.2 | 12.03 | 0.5222 | 0.7635 | 0.7601 | 0.6882 | 1463.0 |
| fixed λ=0.3 | 12.09 | 0.5111 | 0.7504 | 0.7541 | 0.6646 | **1339.4** |
| fixed λ=0.5 | 12.08 | 0.4974 | 0.7294 | 0.7497 | 0.6638 | **1294.6** |

#### 3-bit, group_size=128 — diagonal

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| diagonal λ=0.0 | **11.39** | 0.5205 | 0.7774 | **0.7655** | 0.6669 | 1811.0 |
| diagonal λ=0.001 | **11.58** | 0.5256 | 0.7795 | 0.7617 | 0.6661 | **1283.2** |
| diagonal λ=0.01 | **11.53** | 0.5239 | 0.7807 | 0.7595 | **0.6811** | **1313.2** |
| diagonal λ=0.05 | 11.82 | 0.5341 | 0.7723 | 0.7568 | **0.6946** | 1443.1 |
| diagonal λ=0.1 | 11.73 | 0.5154 | 0.7601 | 0.7563 | **0.6867** | 1546.6 |
| diagonal λ=0.15 | 11.68 | **0.5478** | **0.7984** | 0.7612 | 0.6788 | 1482.3 |
| diagonal λ=0.2 | 11.79 | **0.5580** | **0.7866** | 0.7622 | 0.6717 | 1435.6 |
| diagonal λ=0.3 | 11.84 | 0.5461 | 0.7858 | **0.7644** | 0.6772 | 1482.2 |
| diagonal λ=0.5 | 11.91 | **0.5580** | **0.7879** | **0.7639** | 0.6646 | **1426.0** |

#### 3-bit, group_size=128 — actorder+mse

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| incremental+actorder+mse | **11.84** | **0.5213** | **0.7706** | **0.7655** | **0.6882** | 2903.6 |
| fixed λ=0.0 actorder+mse | **11.23** | 0.5162 | **0.7681** | 0.7606 | **0.6914** | **2102.7** |
| fixed λ=0.001 actorder+mse | **11.80** | **0.5230** | **0.7698** | 0.7601 | 0.6867 | **2142.4** |
| fixed λ=0.01 actorder+mse | 12.13 | 0.5085 | 0.7635 | 0.7579 | 0.6882 | **2301.2** |
| diagonal λ=0.01 actorder+mse | 11.89 | 0.5154 | 0.7614 | **0.7639** | 0.6843 | 2664.9 |
| diagonal λ=0.05 actorder+mse | 11.86 | 0.5111 | 0.7576 | 0.7617 | 0.6772 | 2447.0 |
| diagonal λ=0.1 actorder+mse | 12.08 | **0.5196** | 0.7546 | 0.7579 | 0.6756 | 2455.2 |
| diagonal λ=0.15 actorder+mse | 12.10 | 0.5085 | 0.7559 | **0.7655** | **0.6914** | 2487.8 |

#### 3-bit, per-channel — fixed (identity)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| incremental | 40.53 | 0.3456 | 0.5547 | 0.7008 | 0.6204 | 8943.6 |
| fixed λ=0.0 | **28.24** | **0.4522** | **0.6940** | **0.7437** | **0.6606** | **1587.9** |
| fixed λ=0.001 | **28.65** | **0.4206** | **0.6524** | **0.7416** | **0.6811** | **1613.8** |
| fixed λ=0.01 | **28.85** | **0.4010** | **0.6292** | **0.7361** | **0.6511** | **1888.3** |
| fixed λ=0.05 | 33.58 | 0.3532 | 0.5644 | 0.7008 | 0.5919 | 2173.8 |
| fixed λ=0.1 | 41.71 | 0.3114 | 0.5139 | 0.6850 | 0.5533 | 2248.2 |
| fixed λ=0.15 | 41.37 | 0.3131 | 0.5034 | 0.6785 | 0.5714 | 2368.2 |
| fixed λ=0.2 | 45.85 | 0.3080 | 0.5055 | 0.6719 | 0.5699 | 2429.0 |
| fixed λ=0.3 | 52.72 | 0.2901 | 0.4832 | 0.6556 | 0.5533 | 2398.7 |
| fixed λ=0.5 | 56.57 | 0.2978 | 0.4630 | 0.6523 | 0.5359 | 2495.3 |

#### 3-bit, per-channel — diagonal

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| diagonal λ=0.0 | 28.24 | **0.4522** | **0.6940** | **0.7437** | **0.6606** | **1666.8** |
| diagonal λ=0.001 | 27.69 | **0.4488** | **0.6793** | 0.7356 | **0.6875** | **1542.5** |
| diagonal λ=0.01 | 29.58 | 0.4309 | 0.6633 | **0.7519** | **0.6614** | **1749.2** |
| diagonal λ=0.05 | 26.79 | **0.4471** | 0.6734 | 0.7372 | 0.6567 | 2028.9 |
| diagonal λ=0.1 | **23.97** | 0.4420 | 0.6709 | **0.7383** | 0.6425 | 2227.4 |
| diagonal λ=0.15 | **24.15** | 0.4386 | 0.6646 | 0.7296 | 0.6259 | 2240.9 |
| diagonal λ=0.2 | 25.03 | 0.4360 | 0.6763 | 0.7345 | 0.6433 | 2263.5 |
| diagonal λ=0.3 | 25.47 | 0.4360 | 0.6785 | 0.7285 | 0.6290 | 2385.9 |
| diagonal λ=0.5 | **24.83** | 0.4266 | **0.6810** | 0.7187 | 0.6219 | 2433.7 |

#### 3-bit, per-channel — actorder+mse

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| incremental+actorder+mse | 31.55 | 0.3370 | 0.4933 | 0.6997 | 0.5951 | 5539.2 |
| fixed λ=0.0 actorder+mse | **26.46** | 0.3439 | 0.5328 | 0.7046 | 0.5967 | **2044.3** |
| fixed λ=0.001 actorder+mse | 30.91 | 0.3345 | 0.4903 | 0.6975 | 0.5991 | **1848.1** |
| fixed λ=0.01 actorder+mse | 30.99 | 0.3268 | 0.4949 | 0.7029 | **0.6109** | **2097.4** |
| diagonal λ=0.01 actorder+mse | 29.77 | 0.3652 | 0.5366 | 0.6997 | 0.5896 | 2158.6 |
| diagonal λ=0.05 actorder+mse | 30.34 | **0.3763** | **0.5783** | **0.7296** | **0.6117** | 2322.3 |
| diagonal λ=0.1 actorder+mse | **28.91** | **0.3942** | **0.5875** | **0.7301** | 0.6014 | 2406.3 |
| diagonal λ=0.15 actorder+mse | **29.06** | **0.3951** | **0.5758** | **0.7318** | **0.6172** | 2485.1 |

## Environment

- GPU: NVIDIA B200 × 1

## Output

Results are saved under the `output_dir` directory (default: `qwen3-8b/`):

- `quantization_statistics_<quantizer_name>.json` — per-layer quantization error statistics
- Perplexity and accuracy results are printed to stdout

## Notes

- All lambda parameters (`lambda_mode`, `regularization_lambda`, `lambda_list`) are explicitly set in the benchmark script and YAML config. The benchmark does not depend on JointQ's default values for these parameters, ensuring reproducibility even if defaults change in the future.
- This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
