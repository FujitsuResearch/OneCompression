# Llama-3-8B GPTQ Benchmark

GPTQ benchmark for [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) using OneComp v1.1.1.

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
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B

# actorder
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    gptq.actorder=true output_dir=llama3-8b-actorder

# mse
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    gptq.mse=true output_dir=llama3-8b-mse

# mse+actorder
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B \
    gptq.actorder=true gptq.mse=true output_dir=llama3-8b-mse-actorder
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

PPL = perplexity on WikiText-2 (↓ lower is better). Accuracy = 0-shot `acc_norm` where available, `acc` otherwise (winogrande) (↑ higher is better).

### GPTQ (default)

`actorder=false`, `mse=false`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| — (Original) | — | 6.14 | 0.5384 | 0.7757 | 0.8058 | 0.7332 | — |
| 4 | 128 | 14.09 | 0.5060 | 0.7559 | 0.7818 | 0.7174 | 326.9 |
| 4 | per-channel | 399.02 | 0.3268 | 0.5206 | 0.6763 | 0.6464 | 318.7 |
| 3 | 128 | 57.69 | 0.2645 | 0.4482 | 0.6159 | 0.5904 | 323.3 |
| 3 | per-channel | 3381.46 | 0.2312 | 0.2753 | 0.5299 | 0.5114 | 322.7 |

Total elapsed time (including calibration data preparation): 3898.0 s (~65 min).

### GPTQ (actorder)

`actorder=true`, `mse=false`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.63 | 0.5162 | 0.7694 | 0.7911 | 0.7230 | 293.9 |
| 4 | per-channel | 7.65 | 0.4778 | 0.7391 | 0.7764 | 0.7048 | 287.2 |
| 3 | 128 | 9.58 | 0.4113 | 0.6507 | 0.7388 | 0.6953 | 294.6 |
| 3 | per-channel | 1115.93 | 0.2662 | 0.3931 | 0.6045 | 0.5501 | 286.7 |

Total elapsed time (including calibration data preparation): 3754.2 s (~63 min).

### GPTQ (mse)

`actorder=false`, `mse=true`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.84 | 0.5051 | 0.7652 | 0.7916 | 0.7269 | 1415.8 |
| 4 | per-channel | 7.79 | 0.4795 | 0.7226 | 0.7856 | 0.7238 | 368.5 |
| 3 | 128 | 56.76 | 0.4044 | 0.6414 | 0.7519 | 0.6993 | 1668.9 |
| 3 | per-channel | 58.55 | 0.2355 | 0.3763 | 0.5974 | 0.5580 | 373.1 |

Total elapsed time (including calibration data preparation): 6377.4 s (~106 min).

### GPTQ (mse+actorder)

`actorder=true`, `mse=true`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.55 | 0.5469 | 0.7992 | 0.7971 | 0.7316 | 1367.4 |
| 4 | per-channel | 8.19 | 0.4693 | 0.7542 | 0.7905 | 0.7309 | 360.7 |
| 3 | 128 | 8.37 | 0.4744 | 0.7327 | 0.7655 | 0.7364 | 1613.3 |
| 3 | per-channel | 27.55 | 0.3063 | 0.4802 | 0.6915 | 0.6732 | 366.9 |

Total elapsed time (including calibration data preparation): 6266.2 s (~104 min).

## Environment

- GPU: NVIDIA B200 × 1

## Notes

This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
