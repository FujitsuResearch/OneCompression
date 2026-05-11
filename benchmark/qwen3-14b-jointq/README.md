# Qwen3-14B JointQ Benchmark

JointQ benchmark for [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) using OneComp v1.1.0.

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
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=incremental_lambda \
    output_dir=qwen3-14b-incremental

# 2. fixed, lambda=0.0
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.0 \
    output_dir=qwen3-14b-fixed-lam0.0

# 3. fixed, lambda=0.001
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.001 \
    output_dir=qwen3-14b-fixed-lam0.001

# 4. fixed, lambda=0.01
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.01 \
    output_dir=qwen3-14b-fixed-lam0.01

# 5. fixed, lambda=0.05
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.05 \
    output_dir=qwen3-14b-fixed-lam0.05

# 6. fixed, lambda=0.1
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.1 \
    output_dir=qwen3-14b-fixed-lam0.1

# 7. fixed, lambda=0.15
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.15 \
    output_dir=qwen3-14b-fixed-lam0.15

# 8. fixed, lambda=0.2
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.2 \
    output_dir=qwen3-14b-fixed-lam0.2

# 9. fixed, lambda=0.3
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.3 \
    output_dir=qwen3-14b-fixed-lam0.3

# 10. fixed, lambda=0.5
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.5 \
    output_dir=qwen3-14b-fixed-lam0.5

# 11. incremental+actorder+mse
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=incremental_lambda \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=qwen3-14b-incremental-actorder-mse

# 12. fixed, lambda=0.0, actorder+mse
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.0 \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=qwen3-14b-fixed-lam0.0-actorder

# 13. fixed, lambda=0.001, actorder+mse
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.001 \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=qwen3-14b-fixed-lam0.001-actorder

# 14. fixed, lambda=0.01, actorder+mse
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.01 \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=qwen3-14b-fixed-lam0.01-actorder

# 15. diagonal, lambda=0.0
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.0 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-14b-diagonal-lam0.0

# 16. diagonal, lambda=0.001
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.001 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-14b-diagonal-lam0.001

# 17. diagonal, lambda=0.01
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.01 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-14b-diagonal-lam0.01

# 18. diagonal, lambda=0.05
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.05 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-14b-diagonal-lam0.05

# 19. diagonal, lambda=0.1
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.1 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-14b-diagonal-lam0.1

# 20. diagonal, lambda=0.15
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.15 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-14b-diagonal-lam0.15

# 21. diagonal, lambda=0.2
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.2 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-14b-diagonal-lam0.2

# 22. diagonal, lambda=0.3
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.3 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-14b-diagonal-lam0.3

# 23. diagonal, lambda=0.5
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.5 \
    jointq.regularization_mode=diagonal \
    output_dir=qwen3-14b-diagonal-lam0.5

# 24. diagonal, lambda=0.01, actorder+mse
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.01 \
    jointq.regularization_mode=diagonal \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=qwen3-14b-diagonal-lam0.01-actorder

# 25. diagonal, lambda=0.05, actorder+mse
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.05 \
    jointq.regularization_mode=diagonal \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=qwen3-14b-diagonal-lam0.05-actorder

# 26. diagonal, lambda=0.1, actorder+mse
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.1 \
    jointq.regularization_mode=diagonal \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=qwen3-14b-diagonal-lam0.1-actorder

# 27. diagonal, lambda=0.15, actorder+mse
python quant_benchmark.py model_path=/path/to/Qwen3-14B \
    jointq.lambda_mode=fixed_lambda \
    jointq.regularization_lambda=0.15 \
    jointq.regularization_mode=diagonal \
    jointq.actorder=true jointq.gptq_mse=true \
    output_dir=qwen3-14b-diagonal-lam0.15-actorder
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
| — (Original) | — | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| 4 | 128 | 8.83 | 0.5981 | 0.8165 | 0.7916 | 0.7230 | 5872.2 |
| 4 | per-channel | 9.57 | 0.5922 | 0.8051 | 0.7916 | 0.7080 | 17625.7 |
| 3 | 128 | 10.01 | 0.5486 | 0.7908 | 0.7851 | 0.7277 | 11589.2 |
| 3 | per-channel | 22.89 | 0.4804 | 0.7243 | 0.7644 | 0.6796 | 22158.8 |

Total elapsed time (including calibration data preparation): 62908.8 s (~17h 28m).

### JointQ (fixed, λ=0.0)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.85 | 0.6049 | 0.8300 | 0.7982 | 0.7245 | 6828.8 |
| 4 | per-channel | 9.41 | 0.6152 | 0.8237 | 0.7982 | 0.7159 | 3205.9 |
| 3 | 128 | 9.93 | 0.5597 | 0.7774 | 0.7889 | 0.7198 | 6000.1 |
| 3 | per-channel | 20.25 | 0.5367 | 0.7677 | 0.7699 | 0.7032 | 4286.8 |

Total elapsed time (including calibration data preparation): 25967.4 s (~7h 12m).

### JointQ (fixed, λ=0.001)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.86 | 0.6058 | 0.8266 | 0.7965 | 0.7174 | 2277.8 |
| 4 | per-channel | 9.53 | 0.5947 | 0.8211 | 0.7943 | 0.7293 | 1942.3 |
| 3 | 128 | 9.94 | 0.5563 | 0.7795 | 0.7911 | 0.7253 | 3309.5 |
| 3 | per-channel | 20.45 | 0.5068 | 0.7412 | 0.7726 | 0.7103 | 3757.0 |

Total elapsed time (including calibration data preparation): 16889.1 s (~4h 41m).

### JointQ (fixed, λ=0.01)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.83 | 0.5973 | 0.8186 | 0.7949 | 0.7309 | 2549.3 |
| 4 | per-channel | 9.53 | 0.6169 | 0.8245 | 0.7900 | 0.7182 | 3352.8 |
| 3 | 128 | 9.99 | 0.5734 | 0.8030 | 0.7933 | 0.7103 | 3713.7 |
| 3 | per-channel | 22.15 | 0.4889 | 0.7635 | 0.7639 | 0.6953 | 4422.4 |

Total elapsed time (including calibration data preparation): 19679.9 s (~5h 27m).

### JointQ (fixed, λ=0.05)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.84 | 0.6118 | 0.8173 | 0.7905 | 0.7253 | 2621.1 |
| 4 | per-channel | 9.68 | 0.6101 | 0.8178 | 0.7884 | 0.7206 | 4375.6 |
| 3 | 128 | 10.02 | 0.5299 | 0.7677 | 0.7884 | 0.7111 | 3897.4 |
| 3 | per-channel | 22.47 | 0.4642 | 0.7302 | 0.7573 | 0.6646 | 5042.0 |

Total elapsed time (including calibration data preparation): 21707.4 s (~6h 1m).

### JointQ (fixed, λ=0.1)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.83 | 0.6152 | 0.8220 | 0.7900 | 0.7340 | 2396.5 |
| 4 | per-channel | 9.99 | 0.5794 | 0.8081 | 0.7900 | 0.7072 | 4822.1 |
| 3 | 128 | 10.16 | 0.5461 | 0.7761 | 0.7807 | 0.7103 | 3794.4 |
| 3 | per-channel | 23.21 | 0.4804 | 0.7252 | 0.7628 | 0.6827 | 5211.4 |

Total elapsed time (including calibration data preparation): 21888.0 s (~6h 4m).

### JointQ (fixed, λ=0.15)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.83 | 0.6058 | 0.8237 | 0.7894 | 0.7261 | 2444.3 |
| 4 | per-channel | 9.82 | 0.5768 | 0.8018 | 0.7878 | 0.7190 | 5299.2 |
| 3 | 128 | 10.09 | 0.5452 | 0.7782 | 0.7813 | 0.7151 | 3939.3 |
| 3 | per-channel | 24.36 | 0.4667 | 0.7045 | 0.7481 | 0.6543 | 5527.3 |

Total elapsed time (including calibration data preparation): 22908.7 s (~6h 21m).

### JointQ (fixed, λ=0.2)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.85 | 0.6135 | 0.8211 | 0.7960 | 0.7269 | 2331.2 |
| 4 | per-channel | 9.91 | 0.5819 | 0.8152 | 0.7851 | 0.7167 | 5410.4 |
| 3 | 128 | 10.12 | 0.5350 | 0.7639 | 0.7813 | 0.7032 | 3722.8 |
| 3 | per-channel | 24.50 | 0.4650 | 0.7142 | 0.7427 | 0.6511 | 5494.6 |

Total elapsed time (including calibration data preparation): 22646.0 s (~6h 17m).

### JointQ (fixed, λ=0.3)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.85 | 0.6135 | 0.8211 | 0.7992 | 0.7293 | 2203.4 |
| 4 | per-channel | 9.89 | 0.5939 | 0.8085 | 0.7867 | 0.7088 | 5753.0 |
| 3 | 128 | 10.10 | 0.5427 | 0.7643 | 0.7824 | 0.7119 | 3612.5 |
| 3 | per-channel | 25.25 | 0.4701 | 0.7109 | 0.7448 | 0.6559 | 5639.9 |

Total elapsed time (including calibration data preparation): 22918.6 s (~6h 21m).

### JointQ (fixed, λ=0.5)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.87 | 0.6101 | 0.8232 | 0.7949 | 0.7206 | 2116.1 |
| 4 | per-channel | 9.92 | 0.5887 | 0.7984 | 0.7845 | 0.7167 | 6150.5 |
| 3 | 128 | 10.16 | 0.5222 | 0.7529 | 0.7813 | 0.7001 | 3396.7 |
| 3 | per-channel | 24.45 | 0.4889 | 0.7109 | 0.7481 | 0.6448 | 5770.3 |

Total elapsed time (including calibration data preparation): 23046.5 s (~6h 24m).

### JointQ (incremental+actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.90 | 0.6143 | 0.8237 | 0.7938 | 0.7301 | 4241.1 |
| 4 | per-channel | 9.13 | 0.6126 | 0.8199 | 0.7998 | 0.7348 | 9589.0 |
| 3 | 128 | 9.53 | 0.5836 | 0.8178 | 0.7922 | 0.7222 | 5801.3 |
| 3 | per-channel | 20.75 | 0.4727 | 0.7109 | 0.7573 | 0.6212 | 13840.5 |

Total elapsed time (including calibration data preparation): 39149.5 s (~10h 52m).

### JointQ (fixed, λ=0.0, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.89 | 0.6297 | 0.8300 | 0.7943 | 0.7269 | 2916.3 |
| 4 | per-channel | 9.12 | 0.6229 | 0.8274 | 0.7998 | 0.7285 | 5073.1 |
| 3 | 128 | 9.50 | 0.5922 | 0.8228 | 0.7905 | 0.7222 | 3436.4 |
| 3 | per-channel | 18.48 | 0.4701 | 0.7218 | 0.7573 | 0.6275 | 5495.4 |

Total elapsed time (including calibration data preparation): 22541.5 s (~6h 15m).

### JointQ (fixed, λ=0.001, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.90 | 0.6101 | 0.8253 | 0.7933 | 0.7285 | 3215.7 |
| 4 | per-channel | 9.15 | 0.6135 | 0.8215 | 0.8036 | 0.7238 | 2597.9 |
| 3 | 128 | 9.52 | 0.5939 | 0.8190 | 0.7916 | 0.7222 | 4003.5 |
| 3 | per-channel | 20.06 | 0.4531 | 0.7071 | 0.7508 | 0.6314 | 4466.6 |

Total elapsed time (including calibration data preparation): 19904.3 s (~5h 31m).

### JointQ (fixed, λ=0.01, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.89 | 0.6101 | 0.8291 | 0.7927 | 0.7332 | 3349.3 |
| 4 | per-channel | 9.16 | 0.6067 | 0.8266 | 0.8025 | 0.7348 | 3928.7 |
| 3 | 128 | 9.57 | 0.5776 | 0.8060 | 0.7894 | 0.7174 | 4324.9 |
| 3 | per-channel | 21.22 | 0.4480 | 0.7058 | 0.7514 | 0.6369 | 4953.1 |

Total elapsed time (including calibration data preparation): 22200.5 s (~6h 10m).

### JointQ (diagonal, λ=0.0)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.85 | 0.6049 | 0.8300 | 0.7982 | 0.7245 | 6761.7 |
| 4 | per-channel | 9.41 | 0.6152 | 0.8237 | 0.7982 | 0.7159 | 3211.5 |
| 3 | 128 | 9.93 | 0.5597 | 0.7774 | 0.7889 | 0.7198 | 5971.3 |
| 3 | per-channel | 20.25 | 0.5367 | 0.7677 | 0.7699 | 0.7032 | 4303.4 |

Total elapsed time (including calibration data preparation): 25905.1 s (~7h 11m).

### JointQ (diagonal, λ=0.001)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.85 | 0.6058 | 0.8262 | 0.7976 | 0.7309 | 2888.4 |
| 4 | per-channel | 9.52 | 0.6067 | 0.8220 | 0.7922 | 0.7174 | 1883.3 |
| 3 | 128 | 9.94 | 0.5503 | 0.7782 | 0.7867 | 0.7277 | 3875.0 |
| 3 | per-channel | 20.63 | 0.5119 | 0.7605 | 0.7688 | 0.7001 | 3717.5 |

Total elapsed time (including calibration data preparation): 17974.2 s (~4h 59m).

### JointQ (diagonal, λ=0.01)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.86 | 0.6058 | 0.8224 | 0.7982 | 0.7324 | 2760.6 |
| 4 | per-channel | 9.45 | 0.6075 | 0.8363 | 0.7911 | 0.7214 | 3116.7 |
| 3 | 128 | 10.00 | 0.5606 | 0.7828 | 0.7878 | 0.7214 | 3979.5 |
| 3 | per-channel | 22.19 | 0.4923 | 0.7601 | 0.7693 | 0.7009 | 4316.2 |

Total elapsed time (including calibration data preparation): 19638.2 s (~5h 27m).

### JointQ (diagonal, λ=0.05)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.83 | 0.6101 | 0.8178 | 0.7954 | 0.7182 | 2784.6 |
| 4 | per-channel | 9.63 | 0.6032 | 0.8220 | 0.7982 | 0.7206 | 4109.1 |
| 3 | 128 | 9.96 | 0.5341 | 0.7748 | 0.7873 | 0.7222 | 4136.6 |
| 3 | per-channel | 20.26 | 0.4949 | 0.7483 | 0.7758 | 0.7072 | 4909.5 |

Total elapsed time (including calibration data preparation): 21394.1 s (~5h 56m).

### JointQ (diagonal, λ=0.1)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.82 | 0.6152 | 0.8258 | 0.7933 | 0.7238 | 2595.8 |
| 4 | per-channel | 9.53 | 0.6049 | 0.8266 | 0.7976 | 0.7127 | 4495.6 |
| 3 | 128 | 9.93 | 0.5538 | 0.7959 | 0.7889 | 0.6993 | 4045.5 |
| 3 | per-channel | 20.94 | 0.4855 | 0.7471 | 0.7688 | 0.7096 | 5049.6 |

Total elapsed time (including calibration data preparation): 21618.0 s (~6h 0m).

### JointQ (diagonal, λ=0.15)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.86 | 0.6049 | 0.8232 | 0.7949 | 0.7182 | 2579.9 |
| 4 | per-channel | 9.55 | 0.5887 | 0.8215 | 0.7992 | 0.7064 | 4831.0 |
| 3 | 128 | 9.90 | 0.5478 | 0.7862 | 0.7856 | 0.6961 | 4104.9 |
| 3 | per-channel | 19.18 | 0.5017 | 0.7487 | 0.7661 | 0.7143 | 5228.3 |

Total elapsed time (including calibration data preparation): 22357.2 s (~6h 12m).

### JointQ (diagonal, λ=0.2)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.86 | 0.6126 | 0.8262 | 0.7954 | 0.7285 | 2519.8 |
| 4 | per-channel | 9.51 | 0.5879 | 0.8199 | 0.7954 | 0.7159 | 5044.3 |
| 3 | 128 | 9.88 | 0.5606 | 0.7875 | 0.7856 | 0.7080 | 4037.6 |
| 3 | per-channel | 19.32 | 0.5128 | 0.7500 | 0.7644 | 0.6993 | 5341.8 |

Total elapsed time (including calibration data preparation): 22583.3 s (~6h 16m).

### JointQ (diagonal, λ=0.3)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.88 | 0.6092 | 0.8249 | 0.7965 | 0.7269 | 2519.5 |
| 4 | per-channel | 9.53 | 0.5956 | 0.8144 | 0.7987 | 0.7135 | 5459.1 |
| 3 | 128 | 9.88 | 0.5444 | 0.7698 | 0.7862 | 0.7024 | 4072.6 |
| 3 | per-channel | 18.78 | 0.5068 | 0.7424 | 0.7633 | 0.6811 | 5580.4 |

Total elapsed time (including calibration data preparation): 23299.2 s (~6h 28m).

### JointQ (diagonal, λ=0.5)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.87 | 0.6101 | 0.8287 | 0.7987 | 0.7230 | 2316.2 |
| 4 | per-channel | 9.48 | 0.6049 | 0.8106 | 0.7960 | 0.7072 | 5693.7 |
| 3 | 128 | 9.89 | 0.5478 | 0.7807 | 0.7829 | 0.7001 | 3707.5 |
| 3 | per-channel | 18.69 | 0.5111 | 0.7529 | 0.7601 | 0.6867 | 5545.4 |

Total elapsed time (including calibration data preparation): 22865.8 s (~6h 21m).

### JointQ (diagonal, λ=0.01, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.91 | 0.6143 | 0.8295 | 0.7900 | 0.7309 | 3587.2 |
| 4 | per-channel | 9.14 | 0.6092 | 0.8220 | 0.8047 | 0.7324 | 3769.9 |
| 3 | 128 | 9.56 | 0.5785 | 0.8140 | 0.7884 | 0.7245 | 4514.6 |
| 3 | per-channel | 20.78 | 0.4531 | 0.6957 | 0.7568 | 0.6401 | 4951.3 |

Total elapsed time (including calibration data preparation): 22587.6 s (~6h 16m).

### JointQ (diagonal, λ=0.05, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.90 | 0.6067 | 0.8258 | 0.7933 | 0.7348 | 3779.8 |
| 4 | per-channel | 9.18 | 0.6177 | 0.8215 | 0.7949 | 0.7253 | 4613.8 |
| 3 | 128 | 9.61 | 0.5862 | 0.8072 | 0.7927 | 0.7151 | 4692.4 |
| 3 | per-channel | 17.58 | 0.4445 | 0.6940 | 0.7644 | 0.6693 | 5354.6 |

Total elapsed time (including calibration data preparation): 24179.5 s (~6h 42m).

### JointQ (diagonal, λ=0.1, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.95 | 0.6041 | 0.8270 | 0.7954 | 0.7395 | 3978.6 |
| 4 | per-channel | 9.15 | 0.5956 | 0.8241 | 0.7938 | 0.7277 | 5166.3 |
| 3 | 128 | 9.67 | 0.5768 | 0.8161 | 0.7873 | 0.7253 | 4933.7 |
| 3 | per-channel | 16.93 | 0.4684 | 0.7092 | 0.7650 | 0.6780 | 5633.6 |

Total elapsed time (including calibration data preparation): 25547.4 s (~7h 5m).

### JointQ (diagonal, λ=0.15, actorder+mse)

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.92 | 0.6049 | 0.8304 | 0.7938 | 0.7348 | 3720.5 |
| 4 | per-channel | 9.15 | 0.5939 | 0.8152 | 0.7916 | 0.7214 | 5347.1 |
| 3 | 128 | 9.74 | 0.5785 | 0.8173 | 0.7884 | 0.7245 | 4587.8 |
| 3 | per-channel | 16.65 | 0.4804 | 0.7247 | 0.7579 | 0.6890 | 5671.3 |

Total elapsed time (including calibration data preparation): 24994.3 s (~6h 56m).

### Cross-configuration comparison

Bold values indicate the top-3 results in each column (lower is better for PPL and Time; higher is better for accuracy).

#### 4-bit, group_size=128 — fixed (identity)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| incremental | **8.83** | 0.5981 | 0.8165 | 0.7916 | 0.7230 | 5872.2 |
| fixed λ=0.0 | 8.85 | 0.6049 | **0.8300** | **0.7982** | 0.7245 | 6828.8 |
| fixed λ=0.001 | 8.86 | 0.6058 | **0.8266** | **0.7965** | 0.7174 | **2277.8** |
| fixed λ=0.01 | **8.83** | 0.5973 | 0.8186 | 0.7949 | **0.7309** | 2549.3 |
| fixed λ=0.05 | 8.84 | 0.6118 | 0.8173 | 0.7905 | 0.7253 | 2621.1 |
| fixed λ=0.1 | **8.83** | **0.6152** | 0.8220 | 0.7900 | **0.7340** | 2396.5 |
| fixed λ=0.15 | 8.83 | 0.6058 | **0.8237** | 0.7894 | 0.7261 | 2444.3 |
| fixed λ=0.2 | 8.85 | **0.6135** | 0.8211 | 0.7960 | 0.7269 | 2331.2 |
| fixed λ=0.3 | 8.85 | **0.6135** | 0.8211 | **0.7992** | **0.7293** | **2203.4** |
| fixed λ=0.5 | 8.87 | 0.6101 | 0.8232 | 0.7949 | 0.7206 | **2116.1** |

#### 4-bit, group_size=128 — diagonal

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| diagonal λ=0.0 | **8.85** | 0.6049 | **0.8300** | **0.7982** | 0.7245 | 6761.7 |
| diagonal λ=0.001 | 8.85 | 0.6058 | **0.8262** | 0.7976 | **0.7309** | 2888.4 |
| diagonal λ=0.01 | 8.86 | 0.6058 | 0.8224 | **0.7982** | **0.7324** | 2760.6 |
| diagonal λ=0.05 | **8.83** | **0.6101** | 0.8178 | 0.7954 | 0.7182 | 2784.6 |
| diagonal λ=0.1 | **8.82** | **0.6152** | 0.8258 | 0.7933 | 0.7238 | 2595.8 |
| diagonal λ=0.15 | 8.86 | 0.6049 | 0.8232 | 0.7949 | 0.7182 | 2579.9 |
| diagonal λ=0.2 | 8.86 | **0.6126** | 0.8262 | 0.7954 | **0.7285** | **2519.8** |
| diagonal λ=0.3 | 8.88 | 0.6092 | 0.8249 | 0.7965 | 0.7269 | **2519.5** |
| diagonal λ=0.5 | 8.87 | 0.6101 | **0.8287** | **0.7987** | 0.7230 | **2316.2** |

#### 4-bit, group_size=128 — actorder+mse

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| incremental+actorder+mse | **8.90** | **0.6143** | 0.8237 | **0.7938** | 0.7301 | 4241.1 |
| fixed λ=0.0 actorder+mse | **8.89** | **0.6297** | **0.8300** | **0.7943** | 0.7269 | **2916.3** |
| fixed λ=0.001 actorder+mse | 8.90 | 0.6101 | 0.8253 | 0.7933 | 0.7285 | **3215.7** |
| fixed λ=0.01 actorder+mse | **8.89** | 0.6101 | 0.8291 | 0.7927 | 0.7332 | **3349.3** |
| diagonal λ=0.01 actorder+mse | 8.91 | **0.6143** | **0.8295** | 0.7900 | 0.7309 | 3587.2 |
| diagonal λ=0.05 actorder+mse | 8.90 | 0.6067 | 0.8258 | 0.7933 | **0.7348** | 3779.8 |
| diagonal λ=0.1 actorder+mse | 8.95 | 0.6041 | 0.8270 | **0.7954** | **0.7395** | 3978.6 |
| diagonal λ=0.15 actorder+mse | 8.92 | 0.6049 | **0.8304** | 0.7938 | **0.7348** | 3720.5 |

#### 4-bit, per-channel — fixed (identity)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| incremental | 9.57 | 0.5922 | 0.8051 | **0.7916** | 0.7080 | 17625.7 |
| fixed λ=0.0 | **9.41** | **0.6152** | **0.8237** | **0.7982** | 0.7159 | **3205.9** |
| fixed λ=0.001 | **9.53** | 0.5947 | **0.8211** | **0.7943** | **0.7293** | **1942.3** |
| fixed λ=0.01 | **9.53** | **0.6169** | **0.8245** | 0.7900 | 0.7182 | **3352.8** |
| fixed λ=0.05 | 9.68 | **0.6101** | 0.8178 | 0.7884 | **0.7206** | 4375.6 |
| fixed λ=0.1 | 9.99 | 0.5794 | 0.8081 | 0.7900 | 0.7072 | 4822.1 |
| fixed λ=0.15 | 9.82 | 0.5768 | 0.8018 | 0.7878 | **0.7190** | 5299.2 |
| fixed λ=0.2 | 9.91 | 0.5819 | 0.8152 | 0.7851 | 0.7167 | 5410.4 |
| fixed λ=0.3 | 9.89 | 0.5939 | 0.8085 | 0.7867 | 0.7088 | 5753.0 |
| fixed λ=0.5 | 9.92 | 0.5887 | 0.7984 | 0.7845 | 0.7167 | 6150.5 |

#### 4-bit, per-channel — diagonal

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| diagonal λ=0.0 | **9.41** | **0.6152** | **0.8237** | **0.7982** | 0.7159 | **3211.5** |
| diagonal λ=0.001 | 9.52 | **0.6067** | 0.8220 | 0.7922 | **0.7174** | **1883.3** |
| diagonal λ=0.01 | **9.45** | **0.6075** | **0.8363** | 0.7911 | **0.7214** | **3116.7** |
| diagonal λ=0.05 | 9.63 | 0.6032 | 0.8220 | 0.7982 | **0.7206** | 4109.1 |
| diagonal λ=0.1 | 9.53 | 0.6049 | **0.8266** | 0.7976 | 0.7127 | 4495.6 |
| diagonal λ=0.15 | 9.55 | 0.5887 | 0.8215 | **0.7992** | 0.7064 | 4831.0 |
| diagonal λ=0.2 | 9.51 | 0.5879 | 0.8199 | 0.7954 | 0.7159 | 5044.3 |
| diagonal λ=0.3 | 9.53 | 0.5956 | 0.8144 | **0.7987** | 0.7135 | 5459.1 |
| diagonal λ=0.5 | **9.48** | 0.6049 | 0.8106 | 0.7960 | 0.7072 | 5693.7 |

#### 4-bit, per-channel — actorder+mse

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| incremental+actorder+mse | **9.13** | 0.6126 | 0.8199 | 0.7998 | **0.7348** | 9589.0 |
| fixed λ=0.0 actorder+mse | **9.12** | **0.6229** | **0.8274** | 0.7998 | 0.7285 | 5073.1 |
| fixed λ=0.001 actorder+mse | 9.15 | **0.6135** | 0.8215 | **0.8036** | 0.7238 | **2597.9** |
| fixed λ=0.01 actorder+mse | 9.16 | 0.6067 | **0.8266** | **0.8025** | **0.7348** | **3928.7** |
| diagonal λ=0.01 actorder+mse | **9.14** | 0.6092 | 0.8220 | **0.8047** | **0.7324** | **3769.9** |
| diagonal λ=0.05 actorder+mse | 9.18 | **0.6177** | 0.8215 | 0.7949 | 0.7253 | 4613.8 |
| diagonal λ=0.1 actorder+mse | 9.15 | 0.5956 | **0.8241** | 0.7938 | 0.7277 | 5166.3 |
| diagonal λ=0.15 actorder+mse | 9.15 | 0.5939 | 0.8152 | 0.7916 | 0.7214 | 5347.1 |

#### 3-bit, group_size=128 — fixed (identity)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| incremental | 10.01 | 0.5486 | **0.7908** | 0.7851 | **0.7277** | 11589.2 |
| fixed λ=0.0 | **9.93** | **0.5597** | 0.7774 | **0.7889** | **0.7198** | 6000.1 |
| fixed λ=0.001 | **9.94** | **0.5563** | **0.7795** | **0.7911** | **0.7253** | **3309.5** |
| fixed λ=0.01 | **9.99** | **0.5734** | **0.8030** | **0.7933** | 0.7103 | 3713.7 |
| fixed λ=0.05 | 10.02 | 0.5299 | 0.7677 | 0.7884 | 0.7111 | 3897.4 |
| fixed λ=0.1 | 10.16 | 0.5461 | 0.7761 | 0.7807 | 0.7103 | 3794.4 |
| fixed λ=0.15 | 10.09 | 0.5452 | 0.7782 | 0.7813 | 0.7151 | 3939.3 |
| fixed λ=0.2 | 10.12 | 0.5350 | 0.7639 | 0.7813 | 0.7032 | 3722.8 |
| fixed λ=0.3 | 10.10 | 0.5427 | 0.7643 | 0.7824 | 0.7119 | **3612.5** |
| fixed λ=0.5 | 10.16 | 0.5222 | 0.7529 | 0.7813 | 0.7001 | **3396.7** |

#### 3-bit, group_size=128 — diagonal

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| diagonal λ=0.0 | 9.93 | **0.5597** | 0.7774 | **0.7889** | 0.7198 | 5971.3 |
| diagonal λ=0.001 | 9.94 | 0.5503 | 0.7782 | 0.7867 | **0.7277** | **3875.0** |
| diagonal λ=0.01 | 10.00 | **0.5606** | 0.7828 | **0.7878** | **0.7214** | **3979.5** |
| diagonal λ=0.05 | 9.96 | 0.5341 | 0.7748 | 0.7873 | **0.7222** | 4136.6 |
| diagonal λ=0.1 | 9.93 | 0.5538 | **0.7959** | **0.7889** | 0.6993 | 4045.5 |
| diagonal λ=0.15 | 9.90 | 0.5478 | **0.7862** | 0.7856 | 0.6961 | 4104.9 |
| diagonal λ=0.2 | **9.88** | **0.5606** | **0.7875** | 0.7856 | 0.7080 | 4037.6 |
| diagonal λ=0.3 | **9.88** | 0.5444 | 0.7698 | 0.7862 | 0.7024 | 4072.6 |
| diagonal λ=0.5 | **9.89** | 0.5478 | 0.7807 | 0.7829 | 0.7001 | **3707.5** |

#### 3-bit, group_size=128 — actorder+mse

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| incremental+actorder+mse | **9.53** | 0.5836 | **0.8178** | **0.7922** | 0.7222 | 5801.3 |
| fixed λ=0.0 actorder+mse | **9.50** | **0.5922** | **0.8228** | 0.7905 | 0.7222 | **3436.4** |
| fixed λ=0.001 actorder+mse | **9.52** | **0.5939** | **0.8190** | **0.7916** | 0.7222 | **4003.5** |
| fixed λ=0.01 actorder+mse | 9.57 | 0.5776 | 0.8060 | 0.7894 | 0.7174 | **4324.9** |
| diagonal λ=0.01 actorder+mse | 9.56 | 0.5785 | 0.8140 | 0.7884 | **0.7245** | 4514.6 |
| diagonal λ=0.05 actorder+mse | 9.61 | **0.5862** | 0.8072 | **0.7927** | 0.7151 | 4692.4 |
| diagonal λ=0.1 actorder+mse | 9.67 | 0.5768 | 0.8161 | 0.7873 | **0.7253** | 4933.7 |
| diagonal λ=0.15 actorder+mse | 9.74 | 0.5785 | 0.8173 | 0.7884 | **0.7245** | 4587.8 |

#### 3-bit, per-channel — fixed (identity)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| incremental | 22.89 | 0.4804 | 0.7243 | **0.7644** | 0.6796 | 22158.8 |
| fixed λ=0.0 | **20.25** | **0.5367** | **0.7677** | **0.7699** | **0.7032** | **4286.8** |
| fixed λ=0.001 | **20.45** | **0.5068** | **0.7412** | **0.7726** | **0.7103** | **3757.0** |
| fixed λ=0.01 | **22.15** | **0.4889** | **0.7635** | 0.7639 | **0.6953** | **4422.4** |
| fixed λ=0.05 | 22.47 | 0.4642 | 0.7302 | 0.7573 | 0.6646 | 5042.0 |
| fixed λ=0.1 | 23.21 | 0.4804 | 0.7252 | 0.7628 | 0.6827 | 5211.4 |
| fixed λ=0.15 | 24.36 | 0.4667 | 0.7045 | 0.7481 | 0.6543 | 5527.3 |
| fixed λ=0.2 | 24.50 | 0.4650 | 0.7142 | 0.7427 | 0.6511 | 5494.6 |
| fixed λ=0.3 | 25.25 | 0.4701 | 0.7109 | 0.7448 | 0.6559 | 5639.9 |
| fixed λ=0.5 | 24.45 | 0.4889 | 0.7109 | 0.7481 | 0.6448 | 5770.3 |

#### 3-bit, per-channel — diagonal

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| diagonal λ=0.0 | 20.25 | **0.5367** | **0.7677** | **0.7699** | 0.7032 | **4303.4** |
| diagonal λ=0.001 | 20.63 | **0.5119** | **0.7605** | 0.7688 | 0.7001 | **3717.5** |
| diagonal λ=0.01 | 22.19 | 0.4923 | **0.7601** | **0.7693** | 0.7009 | **4316.2** |
| diagonal λ=0.05 | 20.26 | 0.4949 | 0.7483 | **0.7758** | **0.7072** | 4909.5 |
| diagonal λ=0.1 | 20.94 | 0.4855 | 0.7471 | 0.7688 | **0.7096** | 5049.6 |
| diagonal λ=0.15 | **19.18** | 0.5017 | 0.7487 | 0.7661 | **0.7143** | 5228.3 |
| diagonal λ=0.2 | 19.32 | **0.5128** | 0.7500 | 0.7644 | 0.6993 | 5341.8 |
| diagonal λ=0.3 | **18.78** | 0.5068 | 0.7424 | 0.7633 | 0.6811 | 5580.4 |
| diagonal λ=0.5 | **18.69** | 0.5111 | 0.7529 | 0.7601 | 0.6867 | 5545.4 |

#### 3-bit, per-channel — actorder+mse

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| incremental+actorder+mse | 20.75 | **0.4727** | **0.7109** | 0.7573 | 0.6212 | 13840.5 |
| fixed λ=0.0 actorder+mse | 18.48 | **0.4701** | **0.7218** | 0.7573 | 0.6275 | 5495.4 |
| fixed λ=0.001 actorder+mse | 20.06 | 0.4531 | 0.7071 | 0.7508 | 0.6314 | **4466.6** |
| fixed λ=0.01 actorder+mse | 21.22 | 0.4480 | 0.7058 | 0.7514 | 0.6369 | **4953.1** |
| diagonal λ=0.01 actorder+mse | 20.78 | 0.4531 | 0.6957 | 0.7568 | 0.6401 | **4951.3** |
| diagonal λ=0.05 actorder+mse | **17.58** | 0.4445 | 0.6940 | **0.7644** | **0.6693** | 5354.6 |
| diagonal λ=0.1 actorder+mse | **16.93** | 0.4684 | 0.7092 | **0.7650** | **0.6780** | 5633.6 |
| diagonal λ=0.15 actorder+mse | **16.65** | **0.4804** | **0.7247** | **0.7579** | **0.6890** | 5671.3 |

## Environment

- GPU: NVIDIA B200 × 2

## Output

Results are saved under the `output_dir` directory (default: `qwen3-14b/`):

- `quantization_statistics_<quantizer_name>.json` — per-layer quantization error statistics
- Perplexity and accuracy results are printed to stdout

## Notes

- All lambda parameters (`lambda_mode`, `regularization_lambda`, `lambda_list`) are explicitly set in the benchmark script and YAML config. The benchmark does not depend on JointQ's default values for these parameters, ensuring reproducibility even if defaults change in the future.
- This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
