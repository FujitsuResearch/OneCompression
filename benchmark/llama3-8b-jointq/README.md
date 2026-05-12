# Llama-3-8B JointQ Benchmark

JointQ benchmark for [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) using OneComp v1.1.0.

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
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=incremental_lambda \
    output_dir=llama3-8b-incremental

# 2. fixed, lambda=0.0
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.0 \
    output_dir=llama3-8b-fixed-lam0.0

# 3. fixed, lambda=0.001
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.001 \
    output_dir=llama3-8b-fixed-lam0.001

# 4. fixed, lambda=0.01
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.01 \
    output_dir=llama3-8b-fixed-lam0.01

# 5. fixed, lambda=0.05
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.05 \
    output_dir=llama3-8b-fixed-lam0.05

# 6. fixed, lambda=0.1
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.1 \
    output_dir=llama3-8b-fixed-lam0.1

# 7. fixed, lambda=0.15
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.15 \
    output_dir=llama3-8b-fixed-lam0.15

# 8. fixed, lambda=0.2
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.2 \
    output_dir=llama3-8b-fixed-lam0.2

# 9. fixed, lambda=0.3
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.3 \
    output_dir=llama3-8b-fixed-lam0.3

# 10. fixed, lambda=0.5
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.5 \
    output_dir=llama3-8b-fixed-lam0.5

# 11. incremental+actorder+mse
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=incremental_lambda \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=llama3-8b-incremental-actorder-mse

# 12. fixed, lambda=0.0, actorder+mse
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.0 \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=llama3-8b-fixed-lam0.0-actorder

# 13. fixed, lambda=0.001, actorder+mse
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.001 \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=llama3-8b-fixed-lam0.001-actorder

# 14. fixed, lambda=0.01, actorder+mse
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.01 \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=llama3-8b-fixed-lam0.01-actorder

# 15. diagonal, lambda=0.0
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.0 \
    jointq.regularization_mode=diagonal \
    output_dir=llama3-8b-diagonal-lam0.0

# 16. diagonal, lambda=0.001
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.001 \
    jointq.regularization_mode=diagonal \
    output_dir=llama3-8b-diagonal-lam0.001

# 17. diagonal, lambda=0.01
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.01 \
    jointq.regularization_mode=diagonal \
    output_dir=llama3-8b-diagonal-lam0.01

# 18. diagonal, lambda=0.05
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.05 \
    jointq.regularization_mode=diagonal \
    output_dir=llama3-8b-diagonal-lam0.05

# 19. diagonal, lambda=0.1
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.1 \
    jointq.regularization_mode=diagonal \
    output_dir=llama3-8b-diagonal-lam0.1

# 20. diagonal, lambda=0.15
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.15 \
    jointq.regularization_mode=diagonal \
    output_dir=llama3-8b-diagonal-lam0.15

# 21. diagonal, lambda=0.2
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.2 \
    jointq.regularization_mode=diagonal \
    output_dir=llama3-8b-diagonal-lam0.2

# 22. diagonal, lambda=0.3
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.3 \
    jointq.regularization_mode=diagonal \
    output_dir=llama3-8b-diagonal-lam0.3

# 23. diagonal, lambda=0.5
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.5 \
    jointq.regularization_mode=diagonal \
    output_dir=llama3-8b-diagonal-lam0.5

# 24. diagonal, lambda=0.01, actorder+mse
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.01 \
    jointq.regularization_mode=diagonal \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=llama3-8b-diagonal-lam0.01-actorder

# 25. diagonal, lambda=0.05, actorder+mse
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.05 \
    jointq.regularization_mode=diagonal \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=llama3-8b-diagonal-lam0.05-actorder

# 26. diagonal, lambda=0.1, actorder+mse
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.1 \
    jointq.regularization_mode=diagonal \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=llama3-8b-diagonal-lam0.1-actorder

# 27. diagonal, lambda=0.15, actorder+mse
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.15 \
    jointq.regularization_mode=diagonal \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=llama3-8b-diagonal-lam0.15-actorder
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
| — (Original) | — | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| 4 | 128 | 6.61 | 0.5077 | 0.7790 | 0.7884 | 0.7238 | 2196.7 |
| 4 | per-channel | 7.56 | 0.5051 | 0.7715 | 0.7894 | 0.7214 | 8345.1 |
| 3 | 128 | 8.44 | 0.4590 | 0.7020 | 0.7813 | 0.7167 | 4142.3 |
| 3 | per-channel | 14.92 | 0.3865 | 0.6275 | 0.7388 | 0.6275 | 8975.6 |

Total elapsed time (including calibration data preparation): 26447.4 s (~7h 21m).

### JointQ (fixed, λ=0.0)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.66 | 0.5213 | 0.7824 | 0.7911 | 0.7285 | 928.3 |
| 4 | per-channel | 7.57 | 0.4932 | 0.7551 | 0.7884 | 0.7088 | 857.9 |
| 3 | 128 | 8.96 | 0.4599 | 0.7075 | 0.7699 | 0.7159 | 1204.7 |
| 3 | per-channel | 274.13 | 0.3780 | 0.6334 | 0.7285 | 0.6764 | 1664.0 |

Total elapsed time (including calibration data preparation): 7427.3 s (~2h 4m).

### JointQ (fixed, λ=0.001)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.66 | 0.5145 | 0.7866 | 0.7933 | 0.7253 | 974.6 |
| 4 | per-channel | 7.72 | 0.4991 | 0.7685 | 0.7769 | 0.7277 | 1354.5 |
| 3 | 128 | 9.01 | 0.4309 | 0.6831 | 0.7650 | 0.7198 | 1470.1 |
| 3 | per-channel | 25.38 | 0.3660 | 0.6406 | 0.7388 | 0.6906 | 2015.5 |

Total elapsed time (including calibration data preparation): 8593.9 s (~2h 23m).

### JointQ (fixed, λ=0.01)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.66 | 0.5290 | 0.7761 | 0.7971 | 0.7261 | 976.2 |
| 4 | per-channel | 7.83 | 0.4846 | 0.7551 | 0.7862 | 0.7103 | 2162.0 |
| 3 | 128 | 9.01 | 0.4394 | 0.6873 | 0.7726 | 0.7198 | 1538.5 |
| 3 | per-channel | 19.97 | 0.3959 | 0.6524 | 0.7383 | 0.6803 | 2322.7 |

Total elapsed time (including calibration data preparation): 10057.8 s (~2h 48m).

### JointQ (fixed, λ=0.05)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.65 | 0.5239 | 0.7710 | 0.7971 | 0.7174 | 860.5 |
| 4 | per-channel | 8.49 | 0.4061 | 0.6705 | 0.7535 | 0.7009 | 2503.4 |
| 3 | 128 | 8.85 | 0.4497 | 0.6898 | 0.7753 | 0.7103 | 1446.1 |
| 3 | per-channel | 16.94 | 0.3396 | 0.5543 | 0.7258 | 0.6361 | 2595.6 |

Total elapsed time (including calibration data preparation): 10453.1 s (~2h 54m).

### JointQ (fixed, λ=0.1)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.66 | 0.5265 | 0.7740 | 0.7933 | 0.7285 | 836.7 |
| 4 | per-channel | 8.80 | 0.4821 | 0.7344 | 0.7758 | 0.7001 | 2731.0 |
| 3 | 128 | 8.89 | 0.4317 | 0.6780 | 0.7661 | 0.7009 | 1371.2 |
| 3 | per-channel | 18.01 | 0.3695 | 0.5905 | 0.7258 | 0.6898 | 2722.8 |

Total elapsed time (including calibration data preparation): 10745.2 s (~2h 59m).

### JointQ (fixed, λ=0.15)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.66 | 0.5205 | 0.7685 | 0.8009 | 0.7230 | 835.6 |
| 4 | per-channel | 8.44 | 0.4710 | 0.7361 | 0.7769 | 0.7206 | 2921.3 |
| 3 | 128 | 9.11 | 0.4164 | 0.6662 | 0.7612 | 0.6953 | 1357.4 |
| 3 | per-channel | 20.34 | 0.3618 | 0.5589 | 0.7160 | 0.6953 | 2833.4 |

Total elapsed time (including calibration data preparation): 11007.5 s (~3h 4m).

### JointQ (fixed, λ=0.2)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.68 | 0.5213 | 0.7618 | 0.7960 | 0.7285 | 767.5 |
| 4 | per-channel | 8.44 | 0.4659 | 0.7365 | 0.7748 | 0.7293 | 2967.4 |
| 3 | 128 | 9.24 | 0.4189 | 0.6557 | 0.7726 | 0.7032 | 1241.4 |
| 3 | per-channel | 21.16 | 0.3712 | 0.5762 | 0.7171 | 0.6898 | 2807.1 |

Total elapsed time (including calibration data preparation): 10842.0 s (~3h 1m).

### JointQ (fixed, λ=0.3)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.70 | 0.5034 | 0.7618 | 0.7987 | 0.7324 | 738.7 |
| 4 | per-channel | 8.10 | 0.4565 | 0.7214 | 0.7639 | 0.7245 | 3097.7 |
| 3 | 128 | 9.43 | 0.4070 | 0.6574 | 0.7671 | 0.7032 | 1186.2 |
| 3 | per-channel | 19.06 | 0.3609 | 0.5581 | 0.7078 | 0.6693 | 2849.3 |

Total elapsed time (including calibration data preparation): 10599.9 s (~2h 57m).

### JointQ (fixed, λ=0.5)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.74 | 0.5085 | 0.7618 | 0.7992 | 0.7261 | 668.1 |
| 4 | per-channel | 8.13 | 0.4061 | 0.6536 | 0.7394 | 0.7245 | 3232.1 |
| 3 | 128 | 9.64 | 0.4241 | 0.6587 | 0.7579 | 0.6922 | 1114.2 |
| 3 | per-channel | 22.86 | 0.3336 | 0.5400 | 0.7100 | 0.6590 | 2881.4 |

Total elapsed time (including calibration data preparation): 10612.9 s (~2h 57m).

### JointQ (incremental+actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.55 | 0.5230 | 0.7887 | 0.7954 | 0.7427 | 2433.2 |
| 4 | per-channel | 7.19 | 0.4812 | 0.7652 | 0.7829 | 0.7388 | 4256.6 |
| 3 | 128 | 9.80 | 0.4727 | 0.7189 | 0.7764 | 0.7301 | 3132.1 |
| 3 | per-channel | 12.97 | 0.3584 | 0.5850 | 0.7301 | 0.7064 | 5780.0 |

Total elapsed time (including calibration data preparation): 18339.9 s (~5h 6m).

### JointQ (fixed, λ=0.0, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.55 | 0.5333 | 0.7845 | 0.7954 | 0.7372 | 1677.1 |
| 4 | per-channel | 7.07 | 0.5017 | 0.7618 | 0.7884 | 0.7380 | 1099.6 |
| 3 | 128 | 8.26 | 0.4710 | 0.7340 | 0.7682 | 0.7056 | 2047.4 |
| 3 | per-channel | 13.90 | 0.3857 | 0.6019 | 0.7481 | 0.7127 | 1924.6 |

Total elapsed time (including calibration data preparation): 9468.5 s (~2h 37m).

### JointQ (fixed, λ=0.001, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.56 | 0.5282 | 0.7849 | 0.7965 | 0.7419 | 1788.5 |
| 4 | per-channel | 7.21 | 0.4821 | 0.7563 | 0.7742 | 0.7411 | 1644.3 |
| 3 | 128 | 9.19 | 0.4701 | 0.7138 | 0.7731 | 0.7277 | 2201.3 |
| 3 | per-channel | 13.48 | 0.3703 | 0.5947 | 0.7280 | 0.6930 | 2294.0 |

Total elapsed time (including calibration data preparation): 10628.9 s (~2h 57m).

### JointQ (fixed, λ=0.01, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.56 | 0.5273 | 0.7807 | 0.7884 | 0.7419 | 2002.2 |
| 4 | per-channel | 7.67 | 0.4778 | 0.7298 | 0.7748 | 0.7419 | 2402.6 |
| 3 | 128 | 12.27 | 0.4676 | 0.7205 | 0.7748 | 0.7419 | 2420.1 |
| 3 | per-channel | 13.63 | 0.3848 | 0.6284 | 0.7378 | 0.6843 | 2569.3 |

Total elapsed time (including calibration data preparation): 12111.3 s (~3h 21m).

### JointQ (diagonal, λ=0.0)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.66 | 0.5213 | 0.7824 | 0.7911 | 0.7285 | 940.0 |
| 4 | per-channel | 7.57 | 0.4932 | 0.7551 | 0.7884 | 0.7088 | 865.8 |
| 3 | 128 | 8.96 | 0.4599 | 0.7075 | 0.7699 | 0.7159 | 1216.8 |
| 3 | per-channel | 274.13 | 0.3780 | 0.6334 | 0.7285 | 0.6764 | 1671.7 |

Total elapsed time (including calibration data preparation): 7386.0 s (~2h 3m).

### JointQ (diagonal, λ=0.001)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.66 | 0.5213 | 0.7811 | 0.7943 | 0.7277 | 1030.6 |
| 4 | per-channel | 7.60 | 0.5358 | 0.7795 | 0.7884 | 0.7364 | 1318.2 |
| 3 | 128 | 9.00 | 0.4411 | 0.6923 | 0.7748 | 0.7072 | 1528.0 |
| 3 | per-channel | 77.28 | 0.4061 | 0.6490 | 0.7454 | 0.6906 | 2015.2 |

Total elapsed time (including calibration data preparation): 8761.0 s (~2h 26m).

### JointQ (diagonal, λ=0.01)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.67 | 0.5307 | 0.7807 | 0.7998 | 0.7309 | 1046.8 |
| 4 | per-channel | 7.86 | 0.5051 | 0.7694 | 0.7889 | 0.7285 | 2119.6 |
| 3 | 128 | 8.99 | 0.4386 | 0.7163 | 0.7715 | 0.7190 | 1600.8 |
| 3 | per-channel | 18.15 | 0.4275 | 0.6818 | 0.7437 | 0.7072 | 2312.1 |

Total elapsed time (including calibration data preparation): 9928.1 s (~2h 45m).

### JointQ (diagonal, λ=0.05)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.66 | 0.5401 | 0.7870 | 0.7992 | 0.7293 | 908.4 |
| 4 | per-channel | 7.65 | 0.4991 | 0.7622 | 0.7802 | 0.7159 | 2464.3 |
| 3 | 128 | 8.78 | 0.4411 | 0.6995 | 0.7699 | 0.7198 | 1526.6 |
| 3 | per-channel | 15.11 | 0.3447 | 0.5800 | 0.7291 | 0.6740 | 2580.5 |

Total elapsed time (including calibration data preparation): 10320.0 s (~2h 52m).

### JointQ (diagonal, λ=0.1)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.68 | 0.5273 | 0.7736 | 0.7949 | 0.7182 | 842.6 |
| 4 | per-channel | 7.49 | 0.4812 | 0.7580 | 0.7797 | 0.7119 | 2640.9 |
| 3 | 128 | 8.82 | 0.4428 | 0.6974 | 0.7693 | 0.7222 | 1398.7 |
| 3 | per-channel | 14.84 | 0.3584 | 0.6351 | 0.7388 | 0.7111 | 2669.9 |

Total elapsed time (including calibration data preparation): 10410.6 s (~2h 53m).

### JointQ (diagonal, λ=0.15)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.69 | 0.5162 | 0.7664 | 0.7976 | 0.7214 | 877.4 |
| 4 | per-channel | 7.55 | 0.4735 | 0.7517 | 0.7677 | 0.7198 | 2871.4 |
| 3 | 128 | 8.76 | 0.4275 | 0.6843 | 0.7633 | 0.7301 | 1425.0 |
| 3 | per-channel | 15.18 | 0.3677 | 0.6006 | 0.7155 | 0.7111 | 2811.8 |

Total elapsed time (including calibration data preparation): 10862.2 s (~3h 1m).

### JointQ (diagonal, λ=0.2)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.68 | 0.5299 | 0.7727 | 0.7982 | 0.7261 | 844.8 |
| 4 | per-channel | 7.49 | 0.4863 | 0.7588 | 0.7797 | 0.7190 | 2971.6 |
| 3 | 128 | 8.74 | 0.4386 | 0.6898 | 0.7688 | 0.7111 | 1359.9 |
| 3 | per-channel | 15.43 | 0.3592 | 0.5821 | 0.7095 | 0.6938 | 2847.0 |

Total elapsed time (including calibration data preparation): 10895.4 s (~3h 1m).

### JointQ (diagonal, λ=0.3)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.71 | 0.5222 | 0.7685 | 0.7954 | 0.7261 | 797.1 |
| 4 | per-channel | 7.55 | 0.4787 | 0.7433 | 0.7758 | 0.7064 | 3086.3 |
| 3 | 128 | 8.76 | 0.4360 | 0.6785 | 0.7731 | 0.7151 | 1273.3 |
| 3 | per-channel | 16.50 | 0.3319 | 0.5450 | 0.7057 | 0.6946 | 2878.4 |

Total elapsed time (including calibration data preparation): 10723.8 s (~2h 58m).

### JointQ (diagonal, λ=0.5)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.73 | 0.5154 | 0.7609 | 0.8003 | 0.7261 | 717.1 |
| 4 | per-channel | 7.65 | 0.4582 | 0.7315 | 0.7677 | 0.7127 | 3202.3 |
| 3 | 128 | 8.86 | 0.4420 | 0.6797 | 0.7709 | 0.7214 | 1184.6 |
| 3 | per-channel | 16.94 | 0.3302 | 0.5396 | 0.6937 | 0.6953 | 2925.3 |

Total elapsed time (including calibration data preparation): 10723.1 s (~2h 58m).

### JointQ (diagonal, λ=0.01, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.58 | 0.5333 | 0.7828 | 0.7954 | 0.7466 | 2001.1 |
| 4 | per-channel | 7.27 | 0.4761 | 0.7428 | 0.7797 | 0.7380 | 2348.7 |
| 3 | 128 | 8.24 | 0.4718 | 0.7239 | 0.7737 | 0.7190 | 2400.9 |
| 3 | per-channel | 13.90 | 0.3976 | 0.6380 | 0.7437 | 0.7072 | 2541.6 |

Total elapsed time (including calibration data preparation): 11996.8 s (~3h 19m).

### JointQ (diagonal, λ=0.05, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.59 | 0.5410 | 0.8001 | 0.7933 | 0.7364 | 2088.5 |
| 4 | per-channel | 7.47 | 0.4863 | 0.7424 | 0.7726 | 0.7293 | 2729.4 |
| 3 | 128 | 8.25 | 0.4710 | 0.7239 | 0.7737 | 0.7253 | 2589.8 |
| 3 | per-channel | 13.50 | 0.3857 | 0.6233 | 0.7552 | 0.7111 | 2849.8 |

Total elapsed time (including calibration data preparation): 12983.5 s (~3h 36m).

### JointQ (diagonal, λ=0.1, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.61 | 0.5265 | 0.7950 | 0.7938 | 0.7380 | 2267.2 |
| 4 | per-channel | 7.49 | 0.4650 | 0.7311 | 0.7606 | 0.7324 | 3017.7 |
| 3 | 128 | 8.32 | 0.4565 | 0.7302 | 0.7709 | 0.7119 | 2774.2 |
| 3 | per-channel | 14.48 | 0.3515 | 0.5715 | 0.7182 | 0.6969 | 3016.9 |

Total elapsed time (including calibration data preparation): 13785.9 s (~3h 49m).

### JointQ (diagonal, λ=0.15, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.64 | 0.5145 | 0.7858 | 0.7938 | 0.7380 | 1949.2 |
| 4 | per-channel | 7.51 | 0.4608 | 0.7336 | 0.7726 | 0.7198 | 3040.9 |
| 3 | 128 | 8.37 | 0.4616 | 0.7281 | 0.7693 | 0.7182 | 2367.2 |
| 3 | per-channel | 14.45 | 0.3302 | 0.5480 | 0.7089 | 0.6922 | 2931.8 |

Total elapsed time (including calibration data preparation): 12994.8 s (~3h 36m).

### Cross-configuration comparison

Bold values indicate the top-3 results in each column (lower is better for PPL and Time; higher is better for accuracy).

#### 4-bit, group_size=128 — fixed (identity)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| incremental | **6.61** | 0.5077 | **0.7790** | 0.7884 | 0.7238 | 2196.7 |
| fixed λ=0.0 | **6.66** | 0.5213 | **0.7824** | 0.7911 | **0.7285** | 928.3 |
| fixed λ=0.001 | 6.66 | 0.5145 | **0.7866** | 0.7933 | 0.7253 | 974.6 |
| fixed λ=0.01 | 6.66 | **0.5290** | 0.7761 | 0.7971 | 0.7261 | 976.2 |
| fixed λ=0.05 | **6.65** | **0.5239** | 0.7710 | 0.7971 | 0.7174 | 860.5 |
| fixed λ=0.1 | 6.66 | **0.5265** | 0.7740 | 0.7933 | **0.7285** | 836.7 |
| fixed λ=0.15 | 6.66 | 0.5205 | 0.7685 | **0.8009** | 0.7230 | 835.6 |
| fixed λ=0.2 | 6.68 | 0.5213 | 0.7618 | 0.7960 | 0.7285 | **767.5** |
| fixed λ=0.3 | 6.70 | 0.5034 | 0.7618 | **0.7987** | **0.7324** | **738.7** |
| fixed λ=0.5 | 6.74 | 0.5085 | 0.7618 | **0.7992** | 0.7261 | **668.1** |

#### 4-bit, group_size=128 — diagonal

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| diagonal λ=0.0 | **6.66** | 0.5213 | **0.7824** | 0.7911 | **0.7285** | 940.0 |
| diagonal λ=0.001 | **6.66** | 0.5213 | **0.7811** | 0.7943 | 0.7277 | 1030.6 |
| diagonal λ=0.01 | 6.67 | **0.5307** | 0.7807 | **0.7998** | **0.7309** | 1046.8 |
| diagonal λ=0.05 | **6.66** | **0.5401** | **0.7870** | **0.7992** | **0.7293** | 908.4 |
| diagonal λ=0.1 | 6.68 | 0.5273 | 0.7736 | 0.7949 | 0.7182 | **842.6** |
| diagonal λ=0.15 | 6.69 | 0.5162 | 0.7664 | 0.7976 | 0.7214 | 877.4 |
| diagonal λ=0.2 | 6.68 | **0.5299** | 0.7727 | 0.7982 | 0.7261 | 844.8 |
| diagonal λ=0.3 | 6.71 | 0.5222 | 0.7685 | 0.7954 | 0.7261 | **797.1** |
| diagonal λ=0.5 | 6.73 | 0.5154 | 0.7609 | **0.8003** | 0.7261 | **717.1** |

#### 4-bit, group_size=128 — actorder+mse

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| incremental+actorder+mse | **6.55** | 0.5230 | **0.7887** | **0.7954** | **0.7427** | 2433.2 |
| fixed λ=0.0 actorder+mse | **6.55** | **0.5333** | 0.7845 | **0.7954** | 0.7372 | **1677.1** |
| fixed λ=0.001 actorder+mse | **6.56** | 0.5282 | 0.7849 | **0.7965** | **0.7419** | **1788.5** |
| fixed λ=0.01 actorder+mse | 6.56 | 0.5273 | 0.7807 | 0.7884 | 0.7419 | 2002.2 |
| diagonal λ=0.01 actorder+mse | 6.58 | **0.5333** | 0.7828 | 0.7954 | **0.7466** | 2001.1 |
| diagonal λ=0.05 actorder+mse | 6.59 | **0.5410** | **0.8001** | 0.7933 | 0.7364 | 2088.5 |
| diagonal λ=0.1 actorder+mse | 6.61 | 0.5265 | **0.7950** | 0.7938 | 0.7380 | 2267.2 |
| diagonal λ=0.15 actorder+mse | 6.64 | 0.5145 | 0.7858 | 0.7938 | 0.7380 | **1949.2** |

#### 4-bit, per-channel — fixed (identity)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| incremental | **7.56** | **0.5051** | **0.7715** | **0.7894** | 0.7214 | 8345.1 |
| fixed λ=0.0 | **7.57** | **0.4932** | **0.7551** | **0.7884** | 0.7088 | **857.9** |
| fixed λ=0.001 | **7.72** | **0.4991** | **0.7685** | 0.7769 | **0.7277** | **1354.5** |
| fixed λ=0.01 | 7.83 | 0.4846 | 0.7551 | **0.7862** | 0.7103 | **2162.0** |
| fixed λ=0.05 | 8.49 | 0.4061 | 0.6705 | 0.7535 | 0.7009 | 2503.4 |
| fixed λ=0.1 | 8.80 | 0.4821 | 0.7344 | 0.7758 | 0.7001 | 2731.0 |
| fixed λ=0.15 | 8.44 | 0.4710 | 0.7361 | 0.7769 | 0.7206 | 2921.3 |
| fixed λ=0.2 | 8.44 | 0.4659 | 0.7365 | 0.7748 | **0.7293** | 2967.4 |
| fixed λ=0.3 | 8.10 | 0.4565 | 0.7214 | 0.7639 | **0.7245** | 3097.7 |
| fixed λ=0.5 | 8.13 | 0.4061 | 0.6536 | 0.7394 | 0.7245 | 3232.1 |

#### 4-bit, per-channel — diagonal

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| diagonal λ=0.0 | 7.57 | 0.4932 | 0.7551 | **0.7884** | 0.7088 | **865.8** |
| diagonal λ=0.001 | 7.60 | **0.5358** | **0.7795** | **0.7884** | **0.7364** | **1318.2** |
| diagonal λ=0.01 | 7.86 | **0.5051** | **0.7694** | **0.7889** | **0.7285** | **2119.6** |
| diagonal λ=0.05 | 7.65 | **0.4991** | **0.7622** | 0.7802 | 0.7159 | 2464.3 |
| diagonal λ=0.1 | **7.49** | 0.4812 | 0.7580 | 0.7797 | 0.7119 | 2640.9 |
| diagonal λ=0.15 | **7.55** | 0.4735 | 0.7517 | 0.7677 | **0.7198** | 2871.4 |
| diagonal λ=0.2 | **7.49** | 0.4863 | 0.7588 | 0.7797 | 0.7190 | 2971.6 |
| diagonal λ=0.3 | 7.55 | 0.4787 | 0.7433 | 0.7758 | 0.7064 | 3086.3 |
| diagonal λ=0.5 | 7.65 | 0.4582 | 0.7315 | 0.7677 | 0.7127 | 3202.3 |

#### 4-bit, per-channel — actorder+mse

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| incremental+actorder+mse | **7.19** | 0.4812 | **0.7652** | **0.7829** | **0.7388** | 4256.6 |
| fixed λ=0.0 actorder+mse | **7.07** | **0.5017** | **0.7618** | **0.7884** | 0.7380 | **1099.6** |
| fixed λ=0.001 actorder+mse | **7.21** | **0.4821** | **0.7563** | 0.7742 | **0.7411** | **1644.3** |
| fixed λ=0.01 actorder+mse | 7.67 | 0.4778 | 0.7298 | 0.7748 | **0.7419** | 2402.6 |
| diagonal λ=0.01 actorder+mse | 7.27 | 0.4761 | 0.7428 | **0.7797** | 0.7380 | **2348.7** |
| diagonal λ=0.05 actorder+mse | 7.47 | **0.4863** | 0.7424 | 0.7726 | 0.7293 | 2729.4 |
| diagonal λ=0.1 actorder+mse | 7.49 | 0.4650 | 0.7311 | 0.7606 | 0.7324 | 3017.7 |
| diagonal λ=0.15 actorder+mse | 7.51 | 0.4608 | 0.7336 | 0.7726 | 0.7198 | 3040.9 |

#### 3-bit, group_size=128 — fixed (identity)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| incremental | **8.44** | **0.4590** | **0.7020** | **0.7813** | **0.7167** | 4142.3 |
| fixed λ=0.0 | 8.96 | **0.4599** | **0.7075** | 0.7699 | 0.7159 | **1204.7** |
| fixed λ=0.001 | 9.01 | 0.4309 | 0.6831 | 0.7650 | **0.7198** | 1470.1 |
| fixed λ=0.01 | 9.01 | 0.4394 | 0.6873 | **0.7726** | **0.7198** | 1538.5 |
| fixed λ=0.05 | **8.85** | **0.4497** | **0.6898** | **0.7753** | 0.7103 | 1446.1 |
| fixed λ=0.1 | **8.89** | 0.4317 | 0.6780 | 0.7661 | 0.7009 | 1371.2 |
| fixed λ=0.15 | 9.11 | 0.4164 | 0.6662 | 0.7612 | 0.6953 | 1357.4 |
| fixed λ=0.2 | 9.24 | 0.4189 | 0.6557 | 0.7726 | 0.7032 | 1241.4 |
| fixed λ=0.3 | 9.43 | 0.4070 | 0.6574 | 0.7671 | 0.7032 | **1186.2** |
| fixed λ=0.5 | 9.64 | 0.4241 | 0.6587 | 0.7579 | 0.6922 | **1114.2** |

#### 3-bit, group_size=128 — diagonal

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| diagonal λ=0.0 | 8.96 | **0.4599** | **0.7075** | 0.7699 | 0.7159 | **1216.8** |
| diagonal λ=0.001 | 9.00 | 0.4411 | 0.6923 | **0.7748** | 0.7072 | 1528.0 |
| diagonal λ=0.01 | 8.99 | 0.4386 | **0.7163** | **0.7715** | 0.7190 | 1600.8 |
| diagonal λ=0.05 | 8.78 | 0.4411 | **0.6995** | 0.7699 | 0.7198 | 1526.6 |
| diagonal λ=0.1 | 8.82 | **0.4428** | 0.6974 | 0.7693 | **0.7222** | 1398.7 |
| diagonal λ=0.15 | **8.76** | 0.4275 | 0.6843 | 0.7633 | **0.7301** | 1425.0 |
| diagonal λ=0.2 | **8.74** | 0.4386 | 0.6898 | 0.7688 | 0.7111 | 1359.9 |
| diagonal λ=0.3 | **8.76** | 0.4360 | 0.6785 | **0.7731** | 0.7151 | **1273.3** |
| diagonal λ=0.5 | 8.86 | **0.4420** | 0.6797 | 0.7709 | **0.7214** | **1184.6** |

#### 3-bit, group_size=128 — actorder+mse

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| incremental+actorder+mse | 9.80 | **0.4727** | 0.7189 | **0.7764** | **0.7301** | 3132.1 |
| fixed λ=0.0 actorder+mse | **8.26** | **0.4710** | **0.7340** | 0.7682 | 0.7056 | **2047.4** |
| fixed λ=0.001 actorder+mse | 9.19 | 0.4701 | 0.7138 | 0.7731 | **0.7277** | **2201.3** |
| fixed λ=0.01 actorder+mse | 12.27 | 0.4676 | 0.7205 | **0.7748** | **0.7419** | 2420.1 |
| diagonal λ=0.01 actorder+mse | **8.24** | **0.4718** | 0.7239 | **0.7737** | 0.7190 | 2400.9 |
| diagonal λ=0.05 actorder+mse | **8.25** | 0.4710 | 0.7239 | 0.7737 | 0.7253 | 2589.8 |
| diagonal λ=0.1 actorder+mse | 8.32 | 0.4565 | **0.7302** | 0.7709 | 0.7119 | 2774.2 |
| diagonal λ=0.15 actorder+mse | 8.37 | 0.4616 | **0.7281** | 0.7693 | 0.7182 | **2367.2** |

#### 3-bit, per-channel — fixed (identity)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| incremental | **14.92** | **0.3865** | 0.6275 | **0.7388** | 0.6275 | 8975.6 |
| fixed λ=0.0 | 274.13 | **0.3780** | **0.6334** | 0.7285 | 0.6764 | **1664.0** |
| fixed λ=0.001 | 25.38 | 0.3660 | **0.6406** | **0.7388** | **0.6906** | **2015.5** |
| fixed λ=0.01 | 19.97 | **0.3959** | **0.6524** | **0.7383** | 0.6803 | **2322.7** |
| fixed λ=0.05 | **16.94** | 0.3396 | 0.5543 | 0.7258 | 0.6361 | 2595.6 |
| fixed λ=0.1 | **18.01** | 0.3695 | 0.5905 | 0.7258 | **0.6898** | 2722.8 |
| fixed λ=0.15 | 20.34 | 0.3618 | 0.5589 | 0.7160 | **0.6953** | 2833.4 |
| fixed λ=0.2 | 21.16 | 0.3712 | 0.5762 | 0.7171 | 0.6898 | 2807.1 |
| fixed λ=0.3 | 19.06 | 0.3609 | 0.5581 | 0.7078 | 0.6693 | 2849.3 |
| fixed λ=0.5 | 22.86 | 0.3336 | 0.5400 | 0.7100 | 0.6590 | 2881.4 |

#### 3-bit, per-channel — diagonal

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| diagonal λ=0.0 | 274.13 | **0.3780** | 0.6334 | 0.7285 | 0.6764 | **1671.7** |
| diagonal λ=0.001 | 77.28 | **0.4061** | **0.6490** | **0.7454** | 0.6906 | **2015.2** |
| diagonal λ=0.01 | 18.15 | **0.4275** | **0.6818** | **0.7437** | **0.7072** | **2312.1** |
| diagonal λ=0.05 | **15.11** | 0.3447 | 0.5800 | 0.7291 | 0.6740 | 2580.5 |
| diagonal λ=0.1 | **14.84** | 0.3584 | **0.6351** | **0.7388** | **0.7111** | 2669.9 |
| diagonal λ=0.15 | **15.18** | 0.3677 | 0.6006 | 0.7155 | **0.7111** | 2811.8 |
| diagonal λ=0.2 | 15.43 | 0.3592 | 0.5821 | 0.7095 | 0.6938 | 2847.0 |
| diagonal λ=0.3 | 16.50 | 0.3319 | 0.5450 | 0.7057 | 0.6946 | 2878.4 |
| diagonal λ=0.5 | 16.94 | 0.3302 | 0.5396 | 0.6937 | 0.6953 | 2925.3 |

#### 3-bit, per-channel — actorder+mse

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| incremental+actorder+mse | **12.97** | 0.3584 | 0.5850 | 0.7301 | 0.7064 | 5780.0 |
| fixed λ=0.0 actorder+mse | 13.90 | **0.3857** | 0.6019 | **0.7481** | **0.7127** | **1924.6** |
| fixed λ=0.001 actorder+mse | **13.48** | 0.3703 | 0.5947 | 0.7280 | 0.6930 | **2294.0** |
| fixed λ=0.01 actorder+mse | 13.63 | 0.3848 | **0.6284** | 0.7378 | 0.6843 | 2569.3 |
| diagonal λ=0.01 actorder+mse | 13.90 | **0.3976** | **0.6380** | **0.7437** | **0.7072** | **2541.6** |
| diagonal λ=0.05 actorder+mse | **13.50** | **0.3857** | **0.6233** | **0.7552** | **0.7111** | 2849.8 |
| diagonal λ=0.1 actorder+mse | 14.48 | 0.3515 | 0.5715 | 0.7182 | 0.6969 | 3016.9 |
| diagonal λ=0.15 actorder+mse | 14.45 | 0.3302 | 0.5480 | 0.7089 | 0.6922 | 2931.8 |

## Environment

- GPU: NVIDIA B200 × 1

## Output

Results are saved under the `output_dir` directory (default: `llama3-8b/`):

- `quantization_statistics_<quantizer_name>.json` — per-layer quantization error statistics
- Perplexity and accuracy results are printed to stdout

## Notes

- All lambda parameters (`lambda_mode`, `regularization_lambda`, `lambda_list`) are explicitly set in the benchmark script and YAML config. The benchmark does not depend on JointQ's default values for these parameters, ensuring reproducibility even if defaults change in the future.
- This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
