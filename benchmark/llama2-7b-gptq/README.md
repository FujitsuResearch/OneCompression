# Llama-2-7B GPTQ Benchmark

GPTQ benchmark for [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) using OneComp v1.3.3.

All combinations of `bits x group_size` are run in a single pass, sharing calibration data accumulation across quantizers for efficiency.

Four configurations are benchmarked (the 2x2 grid of `actorder x mse`):

1. **GPTQ (default)** - `actorder=false`, `mse=false`
2. **GPTQ (actorder)** - `actorder=true`, `mse=false`
3. **GPTQ (mse)** - `actorder=false`, `mse=true`
4. **GPTQ (mse+actorder)** - `actorder=true`, `mse=true`

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

This produces **4 quantizers** (2 bits x 2 group sizes) per configuration.

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
python quant_benchmark.py model_path=meta-llama/Llama-2-7b-hf

# actorder
python quant_benchmark.py model_path=meta-llama/Llama-2-7b-hf \
    gptq.actorder=true output_dir=llama2-7b-actorder

# mse
python quant_benchmark.py model_path=meta-llama/Llama-2-7b-hf \
    gptq.mse=true output_dir=llama2-7b-mse

# mse+actorder
python quant_benchmark.py model_path=meta-llama/Llama-2-7b-hf \
    gptq.actorder=true gptq.mse=true output_dir=llama2-7b-mse-actorder

# 2-bit, group size 32 (all four configurations)
python quant_benchmark_2bit_g32.py model_path=meta-llama/Llama-2-7b-hf
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
| - (Original) | - | 5.47 | 0.4480 | 0.7378 | 0.7845 | 0.6961 | 0.7617 | 0.7945 | 0.7038 | - |
| 4 | 128 | 5.89 | 0.4437 | 0.7176 | 0.7824 | 0.6867 | 0.7551 | 0.7654 | 0.6918 | 268.1 |
| 4 | per-channel | 6.17 | 0.4369 | 0.6982 | 0.7824 | 0.6827 | 0.7401 | 0.7492 | 0.6816 | 260.3 |
| 3 | 128 | 6.80 | 0.3857 | 0.6170 | 0.7563 | 0.6496 | 0.6906 | 0.7162 | 0.6359 | 263.1 |
| 3 | per-channel | 27.74 | 0.2782 | 0.4125 | 0.6572 | 0.5343 | 0.5154 | 0.6144 | 0.5020 | 257.6 |

Quantization phase elapsed time: 3676.58 s (~1h 1m).

### GPTQ (actorder)

`actorder=true`, `mse=false`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 128 | 5.68 | 0.4249 | 0.7083 | 0.7742 | 0.6882 | 0.7517 | 0.7936 | 0.6902 | 263.9 |
| 4 | per-channel | 5.96 | 0.4206 | 0.7163 | 0.7726 | 0.6788 | 0.7378 | 0.7578 | 0.6807 | 258.8 |
| 3 | 128 | 6.91 | 0.3763 | 0.6418 | 0.7503 | 0.6172 | 0.6656 | 0.7034 | 0.6258 | 260.4 |
| 3 | per-channel | 14.13 | 0.2500 | 0.3211 | 0.5615 | 0.5091 | 0.3728 | 0.5483 | 0.4271 | 256.4 |

Quantization phase elapsed time: 3659.05 s (~1h 0m).

### GPTQ (mse)

`actorder=false`, `mse=true`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 128 | 5.97 | 0.4394 | 0.7247 | 0.7807 | 0.6835 | 0.7483 | 0.7709 | 0.6913 | 1369.4 |
| 4 | per-channel | 27.21 | 0.3498 | 0.6448 | 0.7421 | 0.6811 | 0.5388 | 0.6003 | 0.5928 | 379.3 |
| 3 | 128 | 7.19 | 0.4215 | 0.7024 | 0.7704 | 0.6496 | 0.6850 | 0.7119 | 0.6568 | 1607.2 |
| 3 | per-channel | 855.39 | 0.2688 | 0.4470 | 0.6583 | 0.5588 | 0.3454 | 0.4193 | 0.4496 | 379.7 |

Quantization phase elapsed time: 6358.72 s (~1h 45m).

### GPTQ (mse+actorder)

`actorder=true`, `mse=true`

| bits | group_size | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 128 | 5.67 | 0.4420 | 0.7247 | 0.7807 | 0.6867 | 0.7517 | 0.7752 | 0.6935 | 1301.1 |
| 4 | per-channel | 16.53 | 0.3422 | 0.6389 | 0.7236 | 0.6654 | 0.4718 | 0.6101 | 0.5753 | 362.9 |
| 3 | 128 | 9.87 | 0.3601 | 0.6271 | 0.7492 | 0.6456 | 0.5615 | 0.7024 | 0.6077 | 1536.4 |
| 3 | per-channel | 334.30 | 0.2756 | 0.4701 | 0.6219 | 0.5556 | 0.3338 | 0.4505 | 0.4512 | 363.8 |

Quantization phase elapsed time: 6196.59 s (~1h 43m).

### GPTQ (2bit/g32)

Additional benchmark for `bits=2`, `group_size=32` using OneComp v1.3.3.

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|
| default | 583.21 | 0.2500 | 0.2500 | 0.5082 | 0.5036 | 0.2642 | 0.3884 | 0.3607 | 266.6 |
| actorder | 379.71 | 0.2671 | 0.2677 | 0.4940 | 0.5028 | 0.2580 | 0.3786 | 0.3614 | 268.6 |
| mse | 222675.44 | 0.2662 | 0.2584 | 0.5065 | 0.4870 | 0.2599 | 0.3783 | 0.3594 | 5568.9 |
| mse+actorder | 220964.12 | 0.2662 | 0.2588 | 0.5087 | 0.4901 | 0.2602 | 0.3783 | 0.3604 | 5579.6 |

Quantization phase elapsed time: 14363.03 s (~3h 59m).

## Environment

- GPU: NVIDIA B200 x 1

## Notes

This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.