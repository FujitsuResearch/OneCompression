# benchmark

Benchmark scripts for OneComp.
Configuration is managed with [Hydra](https://hydra.cc/).

> **Note:** Hydra is not a dependency of OneComp and must be installed separately.

## Installing Hydra

```bash
pip install hydra-core
```

Verify the installation:

```bash
python -c "import hydra; print(hydra.__version__)"
```

## Benchmarks

| Directory | Description |
|---|---|
| [llama3-8b-gptq/](llama3-8b-gptq/) | Llama-3-8B GPTQ (4bit/3bit × gs128/per-channel) |
| [llama3-8b-jointq/](llama3-8b-jointq/) | Llama-3-8B JointQ (4bit/3bit × gs128/per-channel) |
| [llama3-8b-qep-gptq/](llama3-8b-qep-gptq/) | Llama-3-8B QEP+GPTQ (4bit/3bit × gs128/per-channel) |
| [llama3-8b-various/](llama3-8b-various/) | Llama-3-8B Various quantizers with default parameters (no QEP) |
| [qwen3-8b-gptq/](qwen3-8b-gptq/) | Qwen3-8B GPTQ (4bit/3bit × gs128/per-channel) |
| [qwen3-8b-jointq/](qwen3-8b-jointq/) | Qwen3-8B JointQ (4bit/3bit × gs128/per-channel) |
| [qwen3-14b-gptq/](qwen3-14b-gptq/) | Qwen3-14B GPTQ (4bit/3bit × gs128/per-channel) |
| [qwen3-14b-jointq/](qwen3-14b-jointq/) | Qwen3-14B JointQ (4bit/3bit × gs128/per-channel) |

## Results Summary

### GPTQ vs. JointQ

- Referenced GPTQ directories:
  - [`llama3-8b-gptq/`](llama3-8b-gptq/)
  - [`qwen3-8b-gptq/`](qwen3-8b-gptq/)
  - [`qwen3-14b-gptq/`](qwen3-14b-gptq/)
- Referenced JointQ directories:
  - [`llama3-8b-jointq/`](llama3-8b-jointq/)
  - [`qwen3-8b-jointq/`](qwen3-8b-jointq/)
  - [`qwen3-14b-jointq/`](qwen3-14b-jointq/)
- GPTQ rows use the `num_calibration_samples=1024`, `max_length=2048` results from each GPTQ benchmark README.
- JointQ diagonal rows use `λ=0.05` (4-bit) / `λ=0.1` (3-bit) for gs128, and `λ=0.01` (4-bit) / `λ=0.1` (3-bit) for per-channel.
- JointQ diagonal+mse+actorder rows use the same `λ` values.
- Values are judged separately within the `gs128` and `per-channel` groups in each table.
- `PPL` / `Time` mark the best value in each group in bold.
- Accuracy columns mark the best value in each group in bold, and values that match or exceed `Original` are marked with `*`.

#### Llama-3-8B (4-bit)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| GPTQ (gs128) | 12.55 | 0.4974 | 0.7731 | 0.7938 | 0.7174 | **261.0** |
| GPTQ (gs128, mse+actorder) | **6.55** | **0.5427*** | 0.7891* | 0.7971 | 0.7348 | 1334.8 |
| JointQ (gs128, diagonal λ=0.05) | 6.66 | 0.5401 | 0.7870* | **0.7992** | 0.7293 | 908.4 |
| JointQ (gs128, diagonal λ=0.05, mse+actorder) | 6.59 | 0.5410* | **0.8001*** | 0.7933 | **0.7364** | 2088.5 |

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| GPTQ (per-channel) | 581.81 | 0.3174 | 0.5101 | 0.6828 | 0.6290 | **254.1** |
| GPTQ (per-channel, mse+actorder) | 8.19 | 0.4727 | 0.7391 | **0.7894** | **0.7435*** | 351.2 |
| JointQ (per-channel, diagonal λ=0.01) | 7.86 | **0.5051** | **0.7694** | 0.7889 | 0.7285 | 2119.6 |
| JointQ (per-channel, diagonal λ=0.01, mse+actorder) | **7.27** | 0.4761 | 0.7428 | 0.7797 | 0.7380* | 2348.7 |

#### Llama-3-8B (3-bit)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| GPTQ (gs128) | 47.74 | 0.3029 | 0.4886 | 0.6665 | 0.6369 | **259.0** |
| GPTQ (gs128, mse+actorder) | **8.06** | **0.4753** | 0.7176 | 0.7661 | **0.7340** | 1571.1 |
| JointQ (gs128, diagonal λ=0.1) | 8.82 | 0.4428 | 0.6974 | 0.7693 | 0.7222 | 1398.7 |
| JointQ (gs128, diagonal λ=0.1, mse+actorder) | 8.32 | 0.4565 | **0.7302** | **0.7709** | 0.7119 | 2774.2 |

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 6.14 | 0.5410 | 0.7757 | 0.8058 | 0.7372 | — |
| GPTQ (per-channel) | 1640.28 | 0.2312 | 0.2942 | 0.5365 | 0.5099 | **253.4** |
| GPTQ (per-channel, mse+actorder) | 22.53 | 0.3106 | 0.4819 | 0.6855 | 0.6780 | 356.1 |
| JointQ (per-channel, diagonal λ=0.1) | 14.84 | **0.3584** | **0.6351** | **0.7388** | **0.7111** | 2669.9 |
| JointQ (per-channel, diagonal λ=0.1, mse+actorder) | **14.48** | 0.3515 | 0.5715 | 0.7182 | 0.6969 | 3016.9 |

#### Qwen3-8B (4-bit)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| GPTQ (gs128) | 10.26 | **0.5580** | 0.7934 | 0.7671 | 0.6669 | **259.4** |
| GPTQ (gs128, mse+actorder) | **9.91** | 0.5384 | **0.8056** | **0.7791*** | 0.6835* | 1517.6 |
| JointQ (gs128, diagonal λ=0.05) | 10.17 | 0.5367 | 0.7795 | 0.7682 | **0.6985*** | 982.5 |
| JointQ (gs128, diagonal λ=0.05, mse+actorder) | 9.96 | 0.5461 | 0.7997 | 0.7742 | 0.6867* | 1960.1 |

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| GPTQ (per-channel) | 10.88 | 0.5119 | 0.7466 | 0.7622 | 0.6740 | **252.3** |
| GPTQ (per-channel, mse+actorder) | 11.22 | 0.5401 | **0.7828** | **0.7715** | 0.6693 | 395.0 |
| JointQ (per-channel, diagonal λ=0.01) | **10.62** | 0.5324 | 0.7614 | 0.7704 | **0.6843*** | 1262.4 |
| JointQ (per-channel, diagonal λ=0.01, mse+actorder) | 11.05 | **0.5478** | 0.7799 | 0.7688 | 0.6764* | 1672.5 |

#### Qwen3-8B (3-bit)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| GPTQ (gs128) | 11.75 | 0.4846 | 0.7222 | 0.7481 | 0.6488 | **257.2** |
| GPTQ (gs128, mse+actorder) | **11.24** | **0.5307** | 0.7597 | **0.7601** | **0.6867*** | 1794.6 |
| JointQ (gs128, diagonal λ=0.1) | 11.73 | 0.5154 | **0.7601** | 0.7563 | **0.6867*** | 1546.6 |
| JointQ (gs128, diagonal λ=0.1, mse+actorder) | 12.08 | 0.5196 | 0.7546 | 0.7579 | 0.6756* | 2455.2 |

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 9.72 | 0.5648 | 0.8093 | 0.7769 | 0.6756 | — |
| GPTQ (per-channel) | **20.02** | 0.3200 | 0.4217 | 0.6703 | 0.5391 | **251.9** |
| GPTQ (per-channel, mse+actorder) | 41.77 | 0.3259 | 0.4524 | 0.6801 | 0.5643 | 401.0 |
| JointQ (per-channel, diagonal λ=0.1) | 23.97 | **0.4420** | **0.6709** | **0.7383** | **0.6425** | 2227.4 |
| JointQ (per-channel, diagonal λ=0.1, mse+actorder) | 28.91 | 0.3942 | 0.5875 | 0.7301 | 0.6014 | 2406.3 |

#### Qwen3-14B (4-bit)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| GPTQ (gs128) | 8.84 | 0.5947 | 0.8228 | **0.8003*** | 0.7261 | **430.2** |
| GPTQ (gs128, mse+actorder) | 8.87 | **0.6195*** | 0.8237 | 0.7949 | 0.7285* | 2264.1 |
| JointQ (gs128, diagonal λ=0.05) | **8.83** | 0.6101* | 0.8178 | 0.7954 | 0.7182 | 2784.6 |
| JointQ (gs128, diagonal λ=0.05, mse+actorder) | 8.90 | 0.6067* | **0.8258** | 0.7933 | **0.7348*** | 3779.8 |

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| GPTQ (per-channel) | **9.11** | 0.5862 | 0.8098 | 0.7862 | 0.6953 | **419.9** |
| GPTQ (per-channel, mse+actorder) | 9.26 | 0.5990 | 0.8215 | 0.7960 | 0.7253 | 583.8 |
| JointQ (per-channel, diagonal λ=0.01) | 9.45 | 0.6075* | **0.8363*** | 0.7911 | 0.7214 | 3116.7 |
| JointQ (per-channel, diagonal λ=0.01, mse+actorder) | 9.14 | **0.6092*** | 0.8220 | **0.8047*** | **0.7324*** | 3769.9 |

#### Qwen3-14B (3-bit)

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| GPTQ (gs128) | 10.11 | 0.5384 | 0.7774 | 0.7878 | 0.6890 | **427.2** |
| GPTQ (gs128, mse+actorder) | **9.47** | **0.5853** | **0.8241** | **0.7927** | 0.7190 | 2672.7 |
| JointQ (gs128, diagonal λ=0.1) | 9.93 | 0.5538 | 0.7959 | 0.7889 | 0.6993 | 4045.5 |
| JointQ (gs128, diagonal λ=0.1, mse+actorder) | 9.67 | 0.5768 | 0.8161 | 0.7873 | **0.7253** | 4933.7 |

| Configuration | PPL | ARC-c | ARC-e | PIQA | WinoGrande | Time (s) |
|---|---|---|---|---|---|---|
| Original | 8.64 | 0.6024 | 0.8283 | 0.7982 | 0.7285 | — |
| GPTQ (per-channel) | **13.61** | 0.4138 | 0.5690 | 0.7296 | 0.6069 | **419.5** |
| GPTQ (per-channel, mse+actorder) | 15.73 | **0.4872** | 0.7336 | 0.7650 | 0.6551 | 589.7 |
| JointQ (per-channel, diagonal λ=0.1) | 20.94 | 0.4855 | **0.7471** | **0.7688** | **0.7096** | 5049.6 |
| JointQ (per-channel, diagonal λ=0.1, mse+actorder) | 16.93 | 0.4684 | 0.7092 | 0.7650 | 0.6780 | 5633.6 |
