# GPTQ Benchmark Summary

This document summarizes GPTQ benchmark results for the following models:

- [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)

## Benchmark Configuration

### Common Parameters

| Parameter | Value |
|---|---|
| bits | 4, 3 |
| group_size | 128, null (per-channel) |
| symmetric | true |
| num_calibration_samples | 1024 |
| calibration_strategy | drop_rand |
| max_length | 2048 |
| dtype | bfloat16 |

### Evaluation

- Perplexity (WikiText-2)
- Accuracy (lm-eval-harness, 0-shot)

The accuracy columns use `acc_norm` where available and `acc` for WinoGrande and BoolQ. `Accuracy Avg.` is the arithmetic mean of ARC-c, ARC-e, PIQA, WinoGrande, HellaSwag, and BoolQ.

## GPTQ (Default)

This configuration uses `actorder=false` and `mse=false`.

### Llama-2-7B

| Setting | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|
| Original | 5.47 | 0.4480 | 0.7378 | 0.7845 | 0.6961 | 0.7617 | 0.7945 | 0.7038 | - |
| 4b/g128 | 5.89 | 0.4437 | 0.7176 | 0.7824 | 0.6867 | 0.7551 | 0.7654 | 0.6918 | 268.1 |
| 4b/pc | 6.17 | 0.4369 | 0.6982 | 0.7824 | 0.6827 | 0.7401 | 0.7492 | 0.6816 | 260.3 |
| 3b/g128 | 6.80 | 0.3857 | 0.6170 | 0.7563 | 0.6496 | 0.6906 | 0.7162 | 0.6359 | 263.1 |
| 3b/pc | 27.74 | 0.2782 | 0.4125 | 0.6572 | 0.5343 | 0.5154 | 0.6144 | 0.5020 | 257.6 |

### Llama-3-8B

| Setting | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|
| Original | 6.14 | 0.5384 | 0.7757 | 0.8058 | 0.7332 | 0.7921 | 0.8199 | 0.7442 | - |
| 4b/g128 | 14.09 | 0.5060 | 0.7559 | 0.7824 | 0.7167 | 0.7810 | 0.8070 | 0.7248 | 277.9 |
| 4b/pc | 399.02 | 0.3268 | 0.5206 | 0.6763 | 0.6464 | 0.6916 | 0.7144 | 0.5960 | 273.1 |
| 3b/g128 | 57.69 | 0.2628 | 0.4482 | 0.6153 | 0.5943 | 0.4888 | 0.6122 | 0.5036 | 274.8 |
| 3b/pc | 3381.46 | 0.2312 | 0.2753 | 0.5305 | 0.5107 | 0.2752 | 0.4165 | 0.3732 | 269.4 |

### Qwen3-8B

| Setting | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|
| Original | 9.73 | 0.5640 | 0.8093 | 0.7758 | 0.6772 | 0.7489 | 0.8664 | 0.7403 | - |
| 4b/g128 | 10.33 | 0.5614 | 0.7879 | 0.7639 | 0.6693 | 0.7404 | 0.8709 | 0.7323 | 321.0 |
| 4b/pc | 11.04 | 0.5077 | 0.7441 | 0.7557 | 0.6835 | 0.7246 | 0.8529 | 0.7114 | 315.2 |
| 3b/g128 | 11.75 | 0.4881 | 0.7193 | 0.7584 | 0.6527 | 0.6914 | 0.8413 | 0.6919 | 318.5 |
| 3b/pc | 19.55 | 0.3217 | 0.4735 | 0.6850 | 0.5391 | 0.5814 | 0.6691 | 0.5449 | 310.9 |

### Qwen3-14B

| Setting | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|
| Original | 8.65 | 0.6041 | 0.8279 | 0.7976 | 0.7309 | 0.7877 | 0.8936 | 0.7736 | - |
| 4b/g128 | 8.85 | 0.6015 | 0.8224 | 0.8003 | 0.7301 | 0.7813 | 0.8896 | 0.7709 | 473.2 |
| 4b/pc | 9.11 | 0.5887 | 0.8001 | 0.7840 | 0.7040 | 0.7722 | 0.8657 | 0.7525 | 463.9 |
| 3b/g128 | 10.07 | 0.5478 | 0.7862 | 0.7889 | 0.6914 | 0.7489 | 0.8508 | 0.7356 | 461.7 |
| 3b/pc | 13.84 | 0.3720 | 0.5669 | 0.7291 | 0.5746 | 0.6839 | 0.7266 | 0.6089 | 455.2 |

## GPTQ (MSE + Actorder)

This configuration uses `actorder=true` and `mse=true`.

### Llama-2-7B

| Setting | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|
| 4b/g128 | 5.67 | 0.4420 | 0.7247 | 0.7807 | 0.6867 | 0.7517 | 0.7752 | 0.6935 | 1301.1 |
| 4b/pc | 16.53 | 0.3422 | 0.6389 | 0.7236 | 0.6654 | 0.4718 | 0.6101 | 0.5753 | 362.9 |
| 3b/g128 | 9.87 | 0.3601 | 0.6271 | 0.7492 | 0.6456 | 0.5615 | 0.7024 | 0.6077 | 1536.4 |
| 3b/pc | 334.30 | 0.2756 | 0.4701 | 0.6219 | 0.5556 | 0.3338 | 0.4505 | 0.4512 | 363.8 |

### Llama-3-8B

| Setting | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|
| 4b/g128 | 6.55 | 0.5478 | 0.7988 | 0.7971 | 0.7316 | 0.7854 | 0.8260 | 0.7478 | 1454.3 |
| 4b/pc | 8.19 | 0.4693 | 0.7542 | 0.7905 | 0.7309 | 0.7493 | 0.8113 | 0.7176 | 397.6 |
| 3b/g128 | 8.37 | 0.4744 | 0.7323 | 0.7655 | 0.7372 | 0.7480 | 0.8086 | 0.7110 | 1705.5 |
| 3b/pc | 27.55 | 0.3063 | 0.4802 | 0.6915 | 0.6732 | 0.7033 | 0.7572 | 0.6020 | 399.2 |

### Qwen3-8B

| Setting | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|
| 4b/g128 | 9.93 | 0.5495 | 0.8005 | 0.7780 | 0.6748 | 0.7456 | 0.8648 | 0.7355 | 1546.3 |
| 4b/pc | 11.18 | 0.5316 | 0.7740 | 0.7666 | 0.6756 | 0.7257 | 0.8670 | 0.7234 | 414.3 |
| 3b/g128 | 11.42 | 0.5418 | 0.7841 | 0.7682 | 0.6906 | 0.7132 | 0.8554 | 0.7255 | 1824.0 |
| 3b/pc | 43.42 | 0.2986 | 0.4310 | 0.6790 | 0.5635 | 0.4958 | 0.7349 | 0.5338 | 417.3 |

### Qwen3-14B

| Setting | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|
| 4b/g128 | 8.92 | 0.6067 | 0.8279 | 0.7971 | 0.7261 | 0.7825 | 0.8829 | 0.7705 | 2610.6 |
| 4b/pc | 9.26 | 0.5913 | 0.8131 | 0.7954 | 0.7222 | 0.7690 | 0.8847 | 0.7626 | 698.2 |
| 3b/g128 | 9.52 | 0.5973 | 0.8169 | 0.7911 | 0.7174 | 0.7619 | 0.8823 | 0.7611 | 3104.6 |
| 3b/pc | 15.68 | 0.4753 | 0.7319 | 0.7677 | 0.6433 | 0.6773 | 0.8697 | 0.6942 | 694.0 |

## GPTQ (2bit/g32)

This additional experiment uses 2-bit quantization with a group size of 32.

### Llama-2-7B

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|
| default | 583.21 | 0.2500 | 0.2500 | 0.5082 | 0.5036 | 0.2642 | 0.3884 | 0.3607 | 266.6 |
| mse+actorder | 220964.12 | 0.2662 | 0.2588 | 0.5087 | 0.4901 | 0.2602 | 0.3783 | 0.3604 | 5579.6 |

### Llama-3-8B

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|
| default | 195282.65 | 0.2602 | 0.2601 | 0.5169 | 0.4933 | 0.2643 | 0.4147 | 0.3682 | 292.1 |
| mse+actorder | 2150393.93 | 0.2474 | 0.2483 | 0.5109 | 0.4878 | 0.2644 | 0.6217 | 0.3968 | 6217.9 |

### Qwen3-8B

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|
| default | 147.46 | 0.2577 | 0.3089 | 0.5566 | 0.5036 | 0.3677 | 0.4434 | 0.4063 | 321.2 |
| mse+actorder | 2487632773.98 | 0.2679 | 0.2551 | 0.5218 | 0.4830 | 0.2611 | 0.6217 | 0.4018 | 6827.4 |

### Qwen3-14B

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | HellaSwag | BoolQ | Accuracy Avg. | Quant time (s) |
|---|---|---|---|---|---|---|---|---|---|
| default | 23.93 | 0.2944 | 0.4465 | 0.6746 | 0.5351 | 0.5230 | 0.5829 | 0.5094 | 518.4 |
| mse+actorder | 45569397.23 | 0.2619 | 0.2546 | 0.5076 | 0.4941 | 0.2591 | 0.6217 | 0.3998 | 11619.3 |