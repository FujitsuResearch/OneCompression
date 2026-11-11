# Qwen3-8B GPTQ Benchmark

GPTQ benchmark for [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) using OneComp v1.3.3.

All combinations of `bits × group_size` are run in a single pass, sharing calibration data accumulation across quantizers for efficiency.

Four configurations are benchmarked (the 2×2 grid of `actorder × mse`):

1. **GPTQ (default)** — `actorder=false`, `mse=false`
2. **GPTQ (actorder)** — `actorder=true`, `mse=false`
3. **GPTQ (mse)** — `actorder=false`, `mse=true`
4. **GPTQ (mse+actorder)** — `actorder=true`, `mse=true` (strongest GPTQ setting)

## Benchmark Configuration

### Common Parameters

| Parameter | Values |
|---|---|
| bits | 4, 3 |
| group_size | 128, per-channel |
| symmetric | true |
| num_calibration_samples | 1024 |
| calibration_strategy | drop_rand |
| max_length | 2048 |
| dtype | bfloat16 |

This produces **4 quantizers** (2 bits × 2 group sizes) per configuration.

### Configuration-Specific Parameters

| Parameter | default | actorder | mse | mse+actorder |
|---|---|---|---|---|
| actorder | false | true | false | true |
| mse | false | false | true | true |

### Evaluation

- Perplexity (WikiText-2)
- Accuracy (lm-eval-harness)

Both are computed for the original (unquantized) model and all dequantized models.

## Usage

Requires [Hydra](https://hydra.cc/) (see [benchmark/README.md](../README.md) for installation).

```bash
# default
python quant_benchmark.py model_path=/path/to/Qwen3-8B

# actorder
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    gptq.actorder=true output_dir=qwen3-8b-actorder

# mse
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    gptq.mse=true output_dir=qwen3-8b-mse

# mse+actorder
python quant_benchmark.py model_path=/path/to/Qwen3-8B \
    gptq.actorder=true gptq.mse=true output_dir=qwen3-8b-mse-actorder
```

### Hydra Overrides

You can override any parameter from the command line:

```bash
# Run only 4-bit
python quant_benchmark.py model_path=/path/to/model 'gptq.bits=[4]'

# Change calibration samples
python quant_benchmark.py model_path=/path/to/model num_calibration_samples=512
```

## Results

PPL = perplexity on WikiText-2 (lower is better). Accuracy = 0-shot `acc_norm` where available, `acc` otherwise (WinoGrande and BoolQ) (higher is better). `Accuracy Avg.` is the arithmetic mean of the six accuracy tasks.

### GPTQ (default)

`actorder=false`, `mse=false`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| - (Original) | - | 9.73 | 0.5640 | 0.8093 | 0.7758 | 0.6772 | 0.7489 | 0.8664 | 0.7403 | - |
| 4 | 128 | 10.33 | 0.5614 | 0.7879 | 0.7639 | 0.6693 | 0.7404 | 0.8709 | 0.7323 | 321.0 |
| 4 | per-channel | 11.04 | 0.5077 | 0.7441 | 0.7557 | 0.6835 | 0.7246 | 0.8529 | 0.7114 | 315.2 |
| 3 | 128 | 11.75 | 0.4881 | 0.7193 | 0.7584 | 0.6527 | 0.6914 | 0.8413 | 0.6919 | 318.5 |
| 3 | per-channel | 19.55 | 0.3217 | 0.4735 | 0.6850 | 0.5391 | 0.5814 | 0.6691 | 0.5449 | 310.9 |

Quantization phase elapsed time: 4305.35 s (~1h 11m).

### GPTQ (actorder)

`actorder=true`, `mse=false`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 128 | 9.99 | 0.5418 | 0.7849 | 0.7688 | 0.6867 | 0.7406 | 0.8593 | 0.7303 | 332.9 |
| 4 | per-channel | 10.39 | 0.5060 | 0.7378 | 0.7688 | 0.6519 | 0.7342 | 0.8599 | 0.7098 | 328.9 |
| 3 | 128 | 11.52 | 0.4761 | 0.6932 | 0.7546 | 0.6511 | 0.7043 | 0.8446 | 0.6873 | 332.6 |
| 3 | per-channel | 16.50 | 0.3191 | 0.4895 | 0.6980 | 0.5541 | 0.6081 | 0.7055 | 0.5624 | 328.5 |

Quantization phase elapsed time: 4350.07 s (~1h 12m).

### GPTQ (mse)

`actorder=false`, `mse=true`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 128 | 10.36 | 0.5435 | 0.7879 | 0.7715 | 0.6835 | 0.7402 | 0.8664 | 0.7322 | 1713.4 |
| 4 | per-channel | 12.78 | 0.5119 | 0.7454 | 0.7650 | 0.6511 | 0.7160 | 0.8609 | 0.7084 | 448.2 |
| 3 | 128 | 14.09 | 0.4625 | 0.7037 | 0.7448 | 0.6519 | 0.7025 | 0.8575 | 0.6872 | 2022.1 |
| 3 | per-channel | 51.11 | 0.2662 | 0.3636 | 0.6420 | 0.5517 | 0.4580 | 0.6887 | 0.4950 | 448.2 |

Quantization phase elapsed time: 7677.89 s (~2h 7m).

### GPTQ (mse+actorder)

`actorder=true`, `mse=true`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 128 | 9.93 | 0.5495 | 0.8005 | 0.7780 | 0.6748 | 0.7456 | 0.8648 | 0.7355 | 1546.3 |
| 4 | per-channel | 11.18 | 0.5316 | 0.7740 | 0.7666 | 0.6756 | 0.7257 | 0.8670 | 0.7234 | 414.3 |
| 3 | 128 | 11.42 | 0.5418 | 0.7841 | 0.7682 | 0.6906 | 0.7132 | 0.8554 | 0.7255 | 1824.0 |
| 3 | per-channel | 43.42 | 0.2986 | 0.4310 | 0.6790 | 0.5635 | 0.4958 | 0.7349 | 0.5338 | 417.3 |

Quantization phase elapsed time: 7231.67 s (~2h 0m).

### GPTQ (2bit/g32)

Additional benchmark for `bits=2`, `group_size=32` using OneComp v1.3.3.

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|
| default | 147.46 | 0.2577 | 0.3089 | 0.5566 | 0.5036 | 0.3677 | 0.4434 | 0.4063 | 321.2 |
| actorder | 52.80 | 0.2526 | 0.3354 | 0.6175 | 0.5178 | 0.4350 | 0.5691 | 0.4546 | 318.6 |
| mse | 2257102635.30 | 0.2730 | 0.2580 | 0.5158 | 0.4957 | 0.2615 | 0.6217 | 0.4043 | 6845.8 |
| mse+actorder | 2487632773.98 | 0.2679 | 0.2551 | 0.5218 | 0.4830 | 0.2611 | 0.6217 | 0.4018 | 6827.4 |

Quantization phase elapsed time: 17390.23 s (~4h 49m).

## Environment

- GPU: NVIDIA B200 × 1

## Notes

This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
