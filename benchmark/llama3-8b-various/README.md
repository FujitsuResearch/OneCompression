# Llama-3-8B Various Quantizers Benchmark

Benchmark for [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) using OneComp v1.1.0.
Various quantizers are run with their **default parameters** (no QEP).

## Quantizers

All 9 quantizers share calibration data accumulation (X^T X) for efficiency.

| Quantizer | Key Defaults |
|---|---|
| GPTQ | wbits=4, groupsize=-1 (per-channel), sym=True |
| JointQ | bits=4, group_size=128, symmetric=False |
| DBF | target_bits=1.5 |
| QUIP | wbits=4 |
| Onebit | iters=10 |
| RTN | wbits=4, groupsize=-1 (per-channel), sym=False |
| CQ | each_row=True |
| ARB | arb_iters=15, split_points=2 |
| QBB | wbits=4 |

## Benchmark Configuration

| Parameter | Value |
|---|---|
| num_calibration_samples | 1024 |
| calibration_strategy | drop_rand |
| max_length | 2048 |

### Evaluation

- Perplexity (WikiText-2)
- Accuracy (lm-eval-harness)

Both are computed for the original (unquantized) model and all dequantized models.

## Usage

Requires [Hydra](https://hydra.cc/) (see [benchmark/README.md](../README.md) for installation).

Specify the path to the model via `model_path`:

```bash
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B
```

## Results

PPL = perplexity on WikiText-2 (↓ lower is better). Accuracy = 0-shot `acc_norm` where available, `acc` otherwise (winogrande) (↑ higher is better).

| Quantizer | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| — (Original) | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| JointQ | 6.58 | 0.5034 | 0.7546 | 0.7949 | 0.7316 | 875.3 |
| QUIP | 6.96 | 0.5017 | 0.7786 | 0.8003 | 0.7135 | 375.3 |
| RTN | 8.52 | 0.4872 | 0.7542 | 0.7655 | 0.7111 | 51.0 |
| GPTQ | 589.20 | 0.3157 | 0.5080 | 0.6817 | 0.6322 | 247.1 |
| CQ | 317,645.00 | 0.2730 | 0.2521 | 0.5098 | 0.5043 | 518.4 |
| DBF (1.5bit) | 186,493.09 | 0.2363 | 0.2748 | 0.5283 | 0.4830 | 7553.4 |
| QBB | 215,737.78 | 0.2577 | 0.2513 | 0.5316 | 0.4767 | 3845.1 |
| Onebit | 864,553.35 | 0.2526 | 0.2567 | 0.5136 | 0.5209 | 75.1 |
| ARB | 465,090.09 | 0.2671 | 0.2433 | 0.5131 | 0.5059 | 47.7 |

Total elapsed time (including calibration data preparation): 16233.0 s (~271 min).

## Environment

- GPU: NVIDIA B200 × 1

## Notes

GPTQ uses `groupsize=-1` (per-channel) by default, which significantly degrades quality compared to `groupsize=128` (see [llama3-8b-gptq](../llama3-8b-gptq/) for grouped results). DBF, Onebit, CQ, ARB, and QBB with default parameters produce near-random accuracy on this model.

This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
