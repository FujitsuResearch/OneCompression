# Llama-3-8B GPTQ Benchmark

GPTQ benchmark for [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) using OneComp v1.3.3.

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

PPL = perplexity on WikiText-2 (lower is better). Accuracy = 0-shot `acc_norm` where available, `acc` otherwise (WinoGrande and BoolQ) (higher is better). `Accuracy Avg.` is the arithmetic mean of the six accuracy tasks.

### GPTQ (default)

`actorder=false`, `mse=false`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| - (Original) | - | 6.14 | 0.5384 | 0.7757 | 0.8058 | 0.7332 | 0.7921 | 0.8199 | 0.7442 | - |
| 4 | 128 | 14.09 | 0.5060 | 0.7559 | 0.7824 | 0.7167 | 0.7810 | 0.8070 | 0.7248 | 277.9 |
| 4 | per-channel | 399.02 | 0.3268 | 0.5206 | 0.6763 | 0.6464 | 0.6916 | 0.7144 | 0.5960 | 273.1 |
| 3 | 128 | 57.69 | 0.2628 | 0.4482 | 0.6153 | 0.5943 | 0.4888 | 0.6122 | 0.5036 | 274.8 |
| 3 | per-channel | 3381.46 | 0.2312 | 0.2753 | 0.5305 | 0.5107 | 0.2752 | 0.4165 | 0.3732 | 269.4 |

Quantization phase elapsed time: 3839.08 s (~1h 3m).

### GPTQ (actorder)

`actorder=true`, `mse=false`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.63 | 0.5162 | 0.7698 | 0.7922 | 0.7238 | 0.7850 | 0.7936 | 0.7301 | 314.0 |
| 4 | per-channel | 7.65 | 0.4778 | 0.7391 | 0.7764 | 0.7048 | 0.7691 | 0.7878 | 0.7092 | 308.8 |
| 3 | 128 | 9.58 | 0.4130 | 0.6503 | 0.7394 | 0.6969 | 0.7236 | 0.7612 | 0.6640 | 312.9 |
| 3 | per-channel | 1115.93 | 0.2662 | 0.3931 | 0.6039 | 0.5493 | 0.4965 | 0.6138 | 0.4871 | 306.6 |

Quantization phase elapsed time: 3999.66 s (~1h 6m).

### GPTQ (mse)

`actorder=false`, `mse=true`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 128 | 8.84 | 0.5043 | 0.7652 | 0.7916 | 0.7277 | 0.7762 | 0.8116 | 0.7294 | 1626.7 |
| 4 | per-channel | 7.79 | 0.4795 | 0.7226 | 0.7856 | 0.7245 | 0.7709 | 0.8076 | 0.7151 | 426.1 |
| 3 | 128 | 56.76 | 0.4027 | 0.6402 | 0.7530 | 0.6977 | 0.7041 | 0.7058 | 0.6506 | 1929.8 |
| 3 | per-channel | 58.55 | 0.2355 | 0.3763 | 0.5974 | 0.5580 | 0.4154 | 0.5872 | 0.4616 | 428.2 |

Quantization phase elapsed time: 7175.96 s (~1h 59m).

### GPTQ (mse+actorder)

`actorder=true`, `mse=true`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 128 | 6.55 | 0.5478 | 0.7988 | 0.7971 | 0.7316 | 0.7854 | 0.8260 | 0.7478 | 1454.3 |
| 4 | per-channel | 8.19 | 0.4693 | 0.7542 | 0.7905 | 0.7309 | 0.7493 | 0.8113 | 0.7176 | 397.6 |
| 3 | 128 | 8.37 | 0.4744 | 0.7323 | 0.7655 | 0.7372 | 0.7480 | 0.8086 | 0.7110 | 1705.5 |
| 3 | per-channel | 27.55 | 0.3063 | 0.4802 | 0.6915 | 0.6732 | 0.7033 | 0.7572 | 0.6020 | 399.2 |

Quantization phase elapsed time: 6708.12 s (~1h 51m).

### GPTQ (2bit/g32)

Additional benchmark for `bits=2`, `group_size=32` using OneComp v1.3.3.

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|
| default | 195282.65 | 0.2602 | 0.2601 | 0.5169 | 0.4933 | 0.2643 | 0.4147 | 0.3682 | 292.1 |
| actorder | 3883.28 | 0.2474 | 0.2576 | 0.5044 | 0.4759 | 0.2611 | 0.4049 | 0.3585 | 291.9 |
| mse | 2107615.56 | 0.2474 | 0.2500 | 0.5087 | 0.4964 | 0.2640 | 0.6217 | 0.3980 | 6225.5 |
| mse+actorder | 2150393.93 | 0.2474 | 0.2483 | 0.5109 | 0.4878 | 0.2644 | 0.6217 | 0.3968 | 6217.9 |

Quantization phase elapsed time: 15838.32 s (~4h 23m).

## Environment

- GPU: NVIDIA B200 × 1

## Notes

This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
