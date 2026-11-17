# Calibration Stability Benchmark

This benchmark measures quantization stability when only a small number of
calibration samples is available. All five quantizers in one run share the same
calibration samples and seed.

## Configurations

Set `bits=3` or `bits=4` to select the quantization bit width. The default is
4 bits. All configurations use symmetric weight quantization with group size
128.

| Quantizer | Regularization | GPTQ MSE | Actorder |
|---|---|---|---|
| `JointQ_{bits}bit_g128` | Main benchmark layer-specific lambdas | false | false |
| `JointQ_{bits}bit_g128_unregularized` | Disabled | false | false |
| `JointQ_{bits}bit_g128_actorder_mse` | Main benchmark layer-specific lambdas | true | true |
| `GPTQ_{bits}bit_g128_default` | - | false | false |
| `GPTQ_{bits}bit_g128_mse_actorder` | - | true | true |

The regularized JointQ lambdas for q/k/v/o/gate/up/down are:

| Bits | q | k | v | o | gate | up | down |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0 | 0 | 0.01 | 0 | 0.01 | 0.01 | 0 |
| 3 | 0.01 | 0.01 | 0.01 | 0 | 0.01 | 0.01 | 0 |

The retry lambda is 0.01 with threshold 0.2. The unregularized configuration
uses zero for every layer and disables retry.

## Calibration

| Parameter | Default | Experiment values |
|---|---:|---|
| Samples | 64 | 8, 16, 32, 64, 128 |
| Seed | 0 | 0 through 19 |
| Strategy | `drop_rand` | `drop_rand` |
| Shuffle documents | `true` | `true` |
| Sequence length | 2048 | 512, 2048 |
| Batch size | 64 | 64 |

## Evaluation

Perplexity is evaluated on WikiText-2. Accuracy is the arithmetic mean of six
lm-eval tasks. Normalized accuracy is used for ARC Easy, ARC Challenge, PIQA,
and HellaSwag; raw accuracy is used for WinoGrande and BoolQ.

Only dequantized models are evaluated. The original model and packed quantized
models are not evaluated.

## Run

Run from this directory with any supported model path:

```bash
../../../.venv/bin/python quant_benchmark.py \
  model_path=meta-llama/Meta-Llama-3-8B \
  output_dir=llama3-8b/4bit/seqlen512/numsamples64_seed0 \
  bits=4 \
  max_length=512 \
  num_calibration_samples=64 \
  calibration_seed=0
```

## Results

Each task writes five `quantization_statistics_<quantizer>.json` files and one
summary named `numsamples{num_calibration_samples}_seed{calibration_seed}.json`
under its output directory. The summary has the following structure:

```json
{
  "bits": 4,
  "max_length": 512,
  "num_calibration_samples": 64,
  "calibration_seed": 0,
  "shuffle_documents": true,
  "results": {
    "JointQ_4bit_g128": {
      "ppl": 10.0,
      "mean_accuracy": 0.7
    }
  }
}
```

## Llama 3 8B Results

The following tables report the mean +/- sample standard deviation over 20
calibration seeds. All runs use a sequence length of 2048.

### 4-bit Results

| Samples | Quantizer | WikiText-2 PPL | Mean Accuracy (%) |
|---:|---|---:|---:|
| 8 | `JointQ_4bit_g128` | 6.9823 +/- 0.2164 | 72.9462 +/- 0.4710 |
| 8 | `JointQ_4bit_g128_unregularized` | 7.0707 +/- 0.2338 | 72.5754 +/- 0.3543 |
| 8 | `JointQ_4bit_g128_actorder_mse` | 6.8113 +/- 0.0384 | 73.3638 +/- 0.3928 |
| 8 | `GPTQ_4bit_g128_default` | 27.5218 +/- 39.9222 | 72.0631 +/- 0.5793 |
| 8 | `GPTQ_4bit_g128_mse_actorder` | 6.8500 +/- 0.0341 | 73.2241 +/- 0.4093 |
| 16 | `JointQ_4bit_g128` | 6.7973 +/- 0.0368 | 73.3571 +/- 0.3014 |
| 16 | `JointQ_4bit_g128_unregularized` | 6.8603 +/- 0.0953 | 73.2345 +/- 0.4440 |
| 16 | `JointQ_4bit_g128_actorder_mse` | 6.6823 +/- 0.0184 | 73.7215 +/- 0.2430 |
| 16 | `GPTQ_4bit_g128_default` | 21.5656 +/- 15.5674 | 72.3464 +/- 0.6049 |
| 16 | `GPTQ_4bit_g128_mse_actorder` | 6.7150 +/- 0.0266 | 73.6259 +/- 0.2623 |
| 32 | `JointQ_4bit_g128` | 6.7211 +/- 0.0199 | 73.7691 +/- 0.3238 |
| 32 | `JointQ_4bit_g128_unregularized` | 6.7575 +/- 0.0304 | 73.5862 +/- 0.3685 |
| 32 | `JointQ_4bit_g128_actorder_mse` | 6.6228 +/- 0.0147 | 73.8767 +/- 0.3901 |
| 32 | `GPTQ_4bit_g128_default` | 15.7321 +/- 3.4866 | 72.9703 +/- 0.4248 |
| 32 | `GPTQ_4bit_g128_mse_actorder` | 6.6435 +/- 0.0116 | 73.8282 +/- 0.4631 |
| 64 | `JointQ_4bit_g128` | 6.6843 +/- 0.0142 | 73.9090 +/- 0.2490 |
| 64 | `JointQ_4bit_g128_unregularized` | 6.7135 +/- 0.0151 | 73.6488 +/- 0.6245 |
| 64 | `JointQ_4bit_g128_actorder_mse` | 6.5860 +/- 0.0151 | 74.0526 +/- 0.2667 |
| 64 | `GPTQ_4bit_g128_default` | 15.4077 +/- 4.7969 | 72.8816 +/- 1.0621 |
| 64 | `GPTQ_4bit_g128_mse_actorder` | 6.6062 +/- 0.0105 | 74.0984 +/- 0.2658 |
| 128 | `JointQ_4bit_g128` | 6.6657 +/- 0.0137 | 73.9940 +/- 0.2862 |
| 128 | `JointQ_4bit_g128_unregularized` | 6.6892 +/- 0.0141 | 73.8175 +/- 0.2395 |
| 128 | `JointQ_4bit_g128_actorder_mse` | 6.5674 +/- 0.0130 | 74.1239 +/- 0.3174 |
| 128 | `GPTQ_4bit_g128_default` | 14.2128 +/- 4.1887 | 73.0534 +/- 0.4323 |
| 128 | `GPTQ_4bit_g128_mse_actorder` | 6.5885 +/- 0.0115 | 74.0526 +/- 0.3081 |

### 3-bit Results

| Samples | Quantizer | WikiText-2 PPL | Mean Accuracy (%) |
|---:|---|---:|---:|
| 8 | `JointQ_3bit_g128` | 11.0952 +/- 0.2984 | 63.0300 +/- 1.3054 |
| 8 | `JointQ_3bit_g128_unregularized` | 12.7257 +/- 1.0680 | 59.8454 +/- 2.3241 |
| 8 | `JointQ_3bit_g128_actorder_mse` | 13.3000 +/- 4.2346 | 65.6103 +/- 2.0197 |
| 8 | `GPTQ_3bit_g128_default` | 230.6170 +/- 478.4136 | 41.9834 +/- 2.6976 |
| 8 | `GPTQ_3bit_g128_mse_actorder` | 11.3662 +/- 1.6689 | 63.2237 +/- 2.9669 |
| 16 | `JointQ_3bit_g128` | 9.9750 +/- 0.2107 | 66.2912 +/- 0.9265 |
| 16 | `JointQ_3bit_g128_unregularized` | 10.5154 +/- 0.3841 | 64.5486 +/- 1.8982 |
| 16 | `JointQ_3bit_g128_actorder_mse` | 10.7588 +/- 1.2402 | 67.8096 +/- 1.7795 |
| 16 | `GPTQ_3bit_g128_default` | 114.8586 +/- 112.3029 | 46.0591 +/- 4.0093 |
| 16 | `GPTQ_3bit_g128_mse_actorder` | 10.0652 +/- 0.7537 | 65.6073 +/- 2.5674 |
| 32 | `JointQ_3bit_g128` | 9.3967 +/- 0.1010 | 67.5764 +/- 0.4242 |
| 32 | `JointQ_3bit_g128_unregularized` | 9.7241 +/- 0.1355 | 66.9076 +/- 0.8236 |
| 32 | `JointQ_3bit_g128_actorder_mse` | 10.8187 +/- 1.1602 | 69.3486 +/- 1.1853 |
| 32 | `GPTQ_3bit_g128_default` | 81.2483 +/- 51.4411 | 48.5335 +/- 4.4532 |
| 32 | `GPTQ_3bit_g128_mse_actorder` | 9.8167 +/- 0.7994 | 67.7166 +/- 1.7296 |
| 64 | `JointQ_3bit_g128` | 9.1449 +/- 0.0786 | 68.2766 +/- 0.5441 |
| 64 | `JointQ_3bit_g128_unregularized` | 9.3636 +/- 0.0827 | 67.8886 +/- 0.7838 |
| 64 | `JointQ_3bit_g128_actorder_mse` | 10.2288 +/- 0.9689 | 70.3128 +/- 1.0168 |
| 64 | `GPTQ_3bit_g128_default` | 74.8887 +/- 36.7259 | 52.1475 +/- 4.5000 |
| 64 | `GPTQ_3bit_g128_mse_actorder` | 9.0630 +/- 0.3937 | 68.2557 +/- 2.0076 |
| 128 | `JointQ_3bit_g128` | 8.9921 +/- 0.0838 | 68.9069 +/- 0.5844 |
| 128 | `JointQ_3bit_g128_unregularized` | 9.1804 +/- 0.0692 | 68.7015 +/- 0.8124 |
| 128 | `JointQ_3bit_g128_actorder_mse` | 10.2559 +/- 0.8503 | 70.5069 +/- 1.1213 |
| 128 | `GPTQ_3bit_g128_default` | 60.3864 +/- 31.9550 | 53.5217 +/- 5.0417 |
| 128 | `GPTQ_3bit_g128_mse_actorder` | 8.8243 +/- 0.4315 | 69.1487 +/- 1.8367 |