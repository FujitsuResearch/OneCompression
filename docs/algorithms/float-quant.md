# FloatQuant (NVFP4 / MXFP4 / FP8)

The FloatQuant quantizer fake-quantizes weights onto microscaling floating-point formats:
**NVFP4**, **MXFP4** (OCP Microscaling), and **FP8 E4M3**. These formats are natively
supported by NVIDIA Blackwell-generation GPUs, whose tensor cores accelerate FP4/FP8
matrix multiplication with per-block scale factors. Quantizing weights to these formats
allows a model to be deployed on such hardware with minimal accuracy loss.

## Element formats

### FP4 E2M1

The 4-bit floating-point element format has 1 sign bit, 2 exponent bits, and 1 mantissa
bit. Its representable magnitudes form the grid

\[
\{0,\ \pm 0.5,\ \pm 1,\ \pm 1.5,\ \pm 2,\ \pm 3,\ \pm 4,\ \pm 6\}
\]

Quantization rounds each scaled weight to the nearest grid point.

### FP8 E4M3

The 8-bit floating-point element format has 1 sign bit, 4 exponent bits, and 3 mantissa
bits, with a maximum magnitude of 448. Rounding is implemented as a round-trip cast
through `torch.float8_e4m3fn`.

## Scaling schemes

### NVFP4: two-level scaling

NVFP4 splits each row into blocks of 16 elements. Every block stores an FP8 E4M3 scale,
and the whole tensor stores a single FP32 scale (NVIDIA definition):

\[
s_{\text{tensor}} = \frac{\max |W|}{448 \cdot 6},
\qquad
s_b = \mathrm{round}_{\text{E4M3}}\!\left(\frac{\max_{i \in b} |w_i|}{6\, s_{\text{tensor}}}\right)
\]

Each weight is then quantized as

\[
\hat{w}_i = \mathrm{round}_{\text{E2M1}}\!\left(\frac{w_i}{s_b\, s_{\text{tensor}}}\right)
\cdot s_b\, s_{\text{tensor}}
\]

The per-tensor scale maps the global maximum onto the largest representable product
\(448 \cdot 6\), so block scales use the full E4M3 range.

### MXFP4: E8M0 power-of-two scaling

MXFP4 follows the OCP Microscaling (MX) v1.0 specification: each block of 32 elements
stores an 8-bit exponent-only (E8M0) scale, i.e. a power of two:

\[
s_b = 2^{\left\lfloor \log_2 \max_{i \in b} |w_i| \right\rfloor - e_{\max}^{\text{elem}}},
\qquad e_{\max}^{\text{elem}} = 2 \text{ for E2M1}
\]

The exponent is clamped to \([-127, 127]\). This maps the block maximum into
\([4, 8)\); scaled magnitudes above the E2M1 maximum 6 saturate to \(\pm 6\) during
element rounding, as prescribed by the MX specification. Compared with NVFP4, the
coarser power-of-two scales and larger blocks typically yield a slightly higher
quantization error.

### FP8: per-channel scaling

FP8 mode uses one FP32 scale per output channel, \(s_r = \max_i |w_{ri}| / 448\), and
rounds \(w_{ri} / s_r\) onto the E4M3 grid.

## Hessian-based error compensation

With `use_hessian=True`, the quantizer runs a GPTQ-style column-sequential loop: each
column is rounded onto the target floating-point grid, and the rounding error is
propagated to the remaining columns through the inverse Hessian (computed from
calibration data). Block scales are computed from the error-compensated weights at each
block boundary. This usually lowers the layer output error compared with direct
rounding.

## Block-scale sweep (`scale_search`)

AbsMax-based block scales are not MSE-optimal: because the number of representable
scale values is small (E4M3 bytes for NVFP4, power-of-two exponents for MXFP4), the
optimal scale can be found by sweeping a small neighborhood of the AbsMax scale and
keeping, per block, the candidate minimizing the reconstruction error (cf. ScaleSweep,
arXiv:2606.07618). With `scale_search=True`:

- **NVFP4**: the E4M3 block scale is swept over `[-3, +7]` bit-pattern steps around
  the AbsMax scale (`[-8, +7]` under the Hessian-weighted objective).
- **MXFP4**: the E8M0 exponent is swept over `[-2, +1]` steps around the
  non-clipping ceil scale, trading saturation of the block maximum for finer
  resolution of the remaining elements.
- With `use_hessian=True`, the backward-compatible default is to run the sweep inside
  the GPTQ loop on the error-compensated weights, minimizing the Hessian-diagonal
  weighted error (WMSE), so the scale selection tracks the compensation.

For ablations and reviewer-facing comparisons, `scale_timing` can separate static and
in-loop scale selection under the same implementation:

- `scale_timing="static"` selects every physical block scale once before GPTQ
  compensation begins. This is the direct ScaleSweep-style baseline.
- `scale_timing="in_loop"` reselects each physical block scale when that block is
  committed by the GPTQ loop.
- `scale_objective="mse"` uses unweighted reconstruction error.
- `scale_objective="diag_wmse"` uses the Hessian diagonal.
- `scale_objective="conditional"` uses the full block metric
  `((H^{-1})_BB)^{-1}`, the Schur-complement cost induced by GPTQ's future
  compensation. This mode currently applies to in-loop Hessian quantization.

`scale_candidate_strategy` controls how aggressively each sweep searches the discrete
scale grid:

- `local` preserves the paper-comparable ScaleSweep windows: NVFP4 uses `[-3,+7]`
  E4M3 neighbors for MSE and `[-8,+7]` for Hessian-weighted objectives; MXFP4 uses
  `[-2,+1]` E8M0 exponent offsets.
- `full` evaluates the full positive E4M3 / E8M0 scale grid. It is slower, but it is
  the clean ablation for proving that a local window is not hiding a better scale.
- `adaptive` starts from the local window and expands only when the selected scale lies
  on a search-window boundary, giving a stronger default candidate set without paying
  full-grid cost on every block.

The sweep never increases the per-block error by construction, adds no inference-time
cost (the checkpoint layout is unchanged), and costs almost nothing at quantization
time for the RTN path.

## Parameters

| Parameter      | Type    | Description                                                        | Default   |
|----------------|---------|--------------------------------------------------------------------|-----------|
| `fmt`          | `str`   | Target format: `"nvfp4"`, `"mxfp4"`, or `"fp8"`                   | `"nvfp4"` |
| `block_size`   | `int`   | Scale-block size (`None` resolves to 16 / 32 / -1 per format)     | `None`    |
| `use_hessian`  | `bool`  | Enable GPTQ-style error-compensated rounding                       | `False`   |
| `scale_search` | `bool`  | Enable the MSE/WMSE block-scale sweep (nvfp4 / mxfp4)              | `False`   |
| `scale_timing` | `str`   | Sweep timing: `"auto"`, `"none"`, `"static"`, or `"in_loop"`       | `"auto"`  |
| `scale_objective` | `str` | Sweep objective: `"auto"`, `"mse"`, `"diag_wmse"`, or `"conditional"` | `"auto"` |
| `scale_candidate_strategy` | `str` | Candidate set: `"local"`, `"full"`, or `"adaptive"` | `"local"` |
| `blocksize`    | `int`   | Columns per outer loop block (only when `use_hessian=True`)        | `128`     |
| `percdamp`     | `float` | Hessian dampening fraction (only when `use_hessian=True`)          | `0.01`    |

## Usage

```python
from onecomp import ModelConfig, Runner
from onecomp.quantizer.floatquant import FloatQuant

model_config = ModelConfig(
    model_id="meta-llama/Llama-2-7b-hf",
    device="cuda:0",
)

floatquant = FloatQuant(fmt="nvfp4", use_hessian=True)

runner = Runner(model_config=model_config, quantizer=floatquant)
runner.run()
```

### Save / load / generate

All three formats save fake-quant checkpoints (dequantized FP16 weights plus a
`quantization_config` that records the format) and reload with
`load_quantized_model`:

```python
runner.save_quantized_model("./qwen_nvfp4")

from onecomp import load_quantized_model
model, tokenizer = load_quantized_model("./qwen_nvfp4")
output = model.generate(**inputs)
```

### vLLM inference

Two paths are supported:

1. **Native export (real low-precision storage, no plugin) — recommended.**
   `runner.save_vllm_native_model` writes a `compressed-tensors` checkpoint
   in the layout vLLM executes with real quantized kernels, so all three
   formats get actual memory savings and faster inference:

    ```python
    runner.run()
    runner.save_vllm_native_model("./qwen_nvfp4_vllm")
    # stock vLLM, no plugin: LLM(model="./qwen_nvfp4_vllm")
    ```

   Per-format storage and kernels:

    | Format | Checkpoint layout | Stored tensors per Linear | vLLM kernel |
    |---|---|---|---|
    | `nvfp4` | `nvfp4-pack-quantized` | `weight_packed` (uint8, 2 FP4/byte), `weight_scale` (E4M3, group 16), `weight_global_scale` (FP32) | FP4 Marlin (W4A16) |
    | `nvfp4` + activation scales | `nvfp4-pack-quantized` | same + `input_global_scale` (FP32) | **FP4 tensor cores (W4A4)**, activation block scales computed at runtime |
    | `mxfp4` | `mxfp4-pack-quantized` | `weight_packed` (uint8), `weight_scale` (E8M0 exponent bytes, group 32) | FP4 Marlin (W4A16) |
    | `fp8`   | `float-quantized` | `weight` (E4M3), `weight_scale` (FP32 per-channel) | FP8 tensor cores (W8A8, dynamic per-token activations) |

   The FP8 path preserves FloatQuant's per-channel scales and
   Hessian-compensated weights bit-exactly. For NVFP4, layers that vLLM
   fuses into one matrix (`q/k/v`, `gate/up`) must share a single global
   scale; the exporter unifies each fused group on the maximum scale and
   re-quantizes the affected shards from their (Hessian-compensated)
   dequantized weights. MXFP4 checkpoints are saved with
   `torch_dtype=bfloat16` because vLLM's MXFP4 kernel only supports
   bfloat16 activations (E8M0 scales exceed the float16 exponent range).
   The standalone function
   `onecomp.quantizer.floatquant.save_vllm_native_model(model, results, dir)`
   is also available when a Runner is not around.

   **NVFP4 W4A4 (FP4 tensor cores).** NVFP4's two-level scaling applies to
   activations exactly as to weights: a static per-layer FP32 global scale
   plus per-block-16 E4M3 scales that vLLM computes at runtime
   (`dynamic="local"`). Only the global scale needs calibration — one
   forward pass recording the per-layer input magnitude statistic:

    ```python
    from onecomp.quantizer.floatquant import (
        collect_input_global_scales, save_vllm_native_model,
    )

    scales = collect_input_global_scales(
        model, tokenizer, quantizer.results.keys(), calibration_texts,
        percentile=100.0,       # AbsMax; lower values trade clipping for resolution
        scale_multiplier=1.0,   # log-scale local-search knob
    )
    save_vllm_native_model(
        model, quantizer.results, "./qwen_nvfp4_w4a4",
        tokenizer=tokenizer, input_global_scales=scales,
    )
    # or: runner.save_vllm_native_model(dir, input_global_scales=scales)
    ```

   The exported checkpoint replaces the W4A16 Marlin kernel with real
   FP4 x FP4 matmuls on Blackwell tensor cores. Because q/k/v (and
   gate/up) receive the same input tensor, the fused-layer shards get
   identical activation scales by construction — no unification step is
   needed on the activation side.

   For W4A4 ablations, `percentile < 100` and `scale_multiplier != 1`
   expose the standard activation-scale tradeoff: a smaller calibration
   percentile increases the divisor scale and improves resolution for
   typical activations, at the cost of clipping rare outliers.

   **Mixed NVFP4 / FP8 per layer (`mixed-precision`).** The two formats
   can be combined in one checkpoint, spending FP8's extra bits only on
   the layers where NVFP4 hurts most:

    ```python
    from onecomp.quantizer.floatquant import (
        select_mixed_formats, save_vllm_mixed_model,
    )

    mixed = select_mixed_formats(
        model, nvfp4_quantizer.results, fp8_quantizer.results,
        fp8_fraction=0.25,  # fraction of the extra FP8 budget to spend
    )
    save_vllm_mixed_model(model, mixed, "./qwen_mixed", tokenizer=tokenizer)
    ```

   `select_mixed_formats` measures each layer's squared reconstruction
   error under both formats and upgrades layers to FP8 by solving the
   resulting 0-1 budgeted assignment with sparse dynamic programming.
   `assignment="greedy"` keeps the older error-reduction-per-byte
   heuristic for ablations. vLLM-fused groups (q/k/v, gate/up) are
   decided as one unit so every fused matrix keeps a single format. The
   checkpoint uses the compressed-tensors `mixed-precision` format (one
   config group per format with explicit layer-name targets); vLLM runs
   NVFP4 layers on FP4 Marlin and FP8 layers on W8A8 tensor cores in the
   same model.

2. **Fake-quant checkpoints (accuracy evaluation).** `save_quantized_model`
   stores dequantized FP16 weights with the dedicated
   `quant_method="onecomp_fake_quant"` (the microscaling format is recorded
   in `fmt`). With the OneComp repository on `PYTHONPATH` (or the package
   installed), the `vllm_plugins.floatquant` plugin registers this method
   and maps every Linear layer to vLLM's unquantized method — the weights
   are already the dequantized FP16 values, so generations match HF exactly,
   but there is no memory or speed benefit:

    ```python
    from vllm import LLM
    llm = LLM(model="./qwen_nvfp4", enforce_eager=True)
    ```

   The dedicated name deliberately does not collide with vLLM's built-in
   `fp8` / `mxfp4` handlers, so native checkpoints of those formats keep
   loading through vLLM's built-ins even while the plugin is installed.
   Legacy fake-quant checkpoints that still carry
   `quant_method="nvfp4"|"mxfp4"|"fp8"` should have their `config.json`
   updated to the dedicated name (the weights are unchanged plain FP16).

There is also `save_vllm_fp8_model(model, dir)`, a legacy FP8 exporter that
re-quantizes any model with *per-tensor* scales into vLLM's plain `fp8`
layout; prefer `save_vllm_native_model` with `fmt="fp8"` results, which
keeps the per-channel scales.

## Format support and measured quality

Deployment-path support per format (verified on Qwen2.5-0.5B-Instruct,
B200):

| Format | quantize | save/load (fake_quant) | HF generate | vLLM (fake_quant + plugin) | vLLM (native kernels) |
|---|---|---|---|---|---|
| `nvfp4` | ✓ | ✓ | ✓ | ✓ | ✓ (`save_vllm_native_model`, FP4 Marlin W4A16) |
| `mxfp4` | ✓ | ✓ | ✓ | ✓ | ✓ (`save_vllm_native_model`, FP4 Marlin W4A16) |
| `fp8`   | ✓ | ✓ | ✓ | ✓ | ✓ (`save_vllm_native_model` per-channel W8A8; `save_vllm_fp8_model` per-tensor) |

Perplexity on a fixed English text (Qwen2.5-0.5B-Instruct, weight-only
fake-quant, `use_hessian=False`, lower is better):

| Model | PPL |
|---|---|
| FP16 baseline | 1.4925 |
| `fp8` (per-channel E4M3) | 1.4951 |
| `mxfp4` (blocks of 32, E8M0 scales) | 1.5141 |
| `nvfp4` (blocks of 16, E4M3 + per-tensor scale) | 1.5227 |

FP8 is nearly lossless. Both FP4 formats cost ~1.5–2% PPL on this model; on
this small model MXFP4 happens to edge out NVFP4, so measure per model
rather than assuming NVFP4's finer blocks always win. Greedy generations of
the reloaded checkpoints match the corresponding HF generations in vLLM
(fake-quant path) for all three formats.

The block-scale sweep improves every configuration at no inference cost
(Qwen2.5-0.5B-Instruct; `heldout` is a text unrelated to the fixed
evaluation text; FP16 PPL 1.1192):

| Configuration | PPL (fixed) | PPL (heldout) | Mean rel. weight error |
|---|---|---|---|
| `nvfp4` RTN AbsMax | 1.1298 | 1.1222 | 0.0944 |
| `nvfp4` RTN + sweep | **1.1255** | **1.1042** | **0.0815** |
| `nvfp4` Hessian AbsMax | 1.1287 | 1.1192 | 0.1256 |
| `nvfp4` Hessian + sweep (WMSE) | **1.1253** | **1.1046** | 0.1096 |
| `mxfp4` RTN ceil | 1.1343 | 1.1086 | 0.1212 |
| `mxfp4` RTN + sweep | **1.1269** | **1.1026** | 0.1145 |
| `mxfp4` Hessian ceil | 1.1557 | 1.1221 | 0.1656 |
| `mxfp4` Hessian + sweep (WMSE) | **1.1331** | **1.1174** | 0.1574 |

Composed with QEP (cross-layer error propagation), the three-stage pipeline
QEP -> WMSE sweep -> GPTQ compensation gives the best NVFP4 quality measured
on this model (PPL 1.1213 fixed / 1.1029 heldout), with each stage
contributing additively.

The gain survives the real vLLM kernels: swept native checkpoints measured
through vLLM `prompt_logprobs` give MXFP4 PPL 1.1477 (vs 1.1549 for the
ceil-rule checkpoint — on par with NVFP4), while swept native NVFP4 is
neutral (1.1492 vs 1.1476; the local MSE surrogate does not always track
downstream PPL through the fused-scale unification, so validate per model).

Native checkpoints measured in vLLM 0.20 on one B200 (Qwen2.5-0.5B-Instruct,
`enforce_eager`, Triton attention backend, batch 8 x 128 new tokens; the
FP16 safetensors weigh ~988 MB):

| Checkpoint | Weight storage | Decode throughput |
|---|---|---|
| FP16 baseline | 988 MB | ~620–770 tok/s |
| `nvfp4` native | 474 MB | ~720–780 tok/s |
| `nvfp4` + Hessian native | 474 MB | ~740–780 tok/s |
| `mxfp4` native | 463 MB | ~700–780 tok/s |
| `fp8` native (per-channel) | 632 MB | ~630–650 tok/s |

Throughput on this 0.5B model is on par with (to modestly above) FP16 —
run-to-run variance on a shared node is ~10% — while weight storage
halves; on this small model the FP16 embedding table (~272 MB) dominates
the checkpoint, so the Linear-weight compression (~4x for FP4, 2x for FP8)
is partially masked. Larger, bandwidth-bound models benefit proportionally
more. All native checkpoints produce coherent greedy generations through
vLLM's quantized kernels (FP4 Marlin / FP8 tensor cores).

Perplexity computed *through vLLM* (`prompt_logprobs`, same fixed text) for
the native checkpoints confirms the real kernels preserve the fake-quant
accuracy (Qwen2.5-0.5B-Instruct; degradation is relative to the vLLM FP16
baseline PPL of 1.1403 on 2048 tokens):

| Native checkpoint | vLLM PPL | Degradation vs FP16 |
|---|---|---|
| `fp8` | 1.1409 | +0.05% |
| `nvfp4` | 1.1476 | +0.64% |
| `nvfp4` + Hessian | 1.1492 | +0.78% |
| `mxfp4` | 1.1549 | +1.28% |

Mixed NVFP4 / FP8 checkpoints trace a monotone accuracy/memory Pareto
front between the pure formats (Qwen2.5-0.5B-Instruct, vLLM
`prompt_logprobs`, real kernels; `fp8_fraction` is the share of the
extra FP8 byte budget spent by `select_mixed_formats`):

| Checkpoint | FP8 layers | Weight storage | vLLM PPL |
|---|---|---|---|
| pure `nvfp4` | 0/168 | 452 MB | 1.1492 |
| mixed 10% | 75/168 | 467 MB | 1.1482 |
| mixed 25% | 90/168 | 489 MB | 1.1449 |
| mixed 50% | 113/168 | 527 MB | 1.1424 |
| pure `fp8` | 168/168 | 602 MB | 1.1409 |

At 25% of the extra budget the mixed checkpoint recovers about half of
the NVFP4-to-FP8 quality gap; earlier results used the ratio-greedy
heuristic, while the current implementation also exposes the exact
budgeted assignment needed for clean ablations.

On a bandwidth-bound model the speed advantage is clear. Qwen2.5-7B
quantized with FloatQuant (RTN) and exported natively, measured on one B200
(vLLM 0.20, CUDA graphs, Triton attention, median of 3 runs, 256 new
tokens; FP16 safetensors weigh ~15.2 GB):

| Checkpoint | Weight storage | Decode bs=1 | Decode bs=8 |
|---|---|---|---|
| FP16 baseline | 15.2 GB | 139.5 tok/s | 1123 tok/s |
| `nvfp4` native | 5.85 GB | 165.7 tok/s (**+19%**) | 1330 tok/s (**+18%**) |
| `mxfp4` native | 5.65 GB | 161.4 tok/s (**+16%**) | 1223 tok/s (+9%) |
| `fp8` native | 8.71 GB | 163.8 tok/s (**+17%**) | 1383 tok/s (**+23%**) |

All 7B native checkpoints also produce coherent greedy generations.

W4A16 vs W4A4 (Qwen2.5-7B, one *idle* B200, CUDA graphs, CUTLASS FP4
GEMM for W4A4, median of 3 runs, 256 new tokens per request; absolute
numbers are higher than the table above because that node was shared):

| Checkpoint | bs=1 | bs=8 | bs=32 |
|---|---|---|---|
| FP16 baseline | 277.6 tok/s | 2164 tok/s | 7929 tok/s |
| `nvfp4` W4A16 (Marlin) | 323.4 (**+16%**) | 2447 (**+13%**) | 7949 (+0.2%) |
| `nvfp4` W4A4 (FP4 tensor cores) | 327.8 (**+18%**) | 2525 (**+17%**) | 8602 (**+8.5%**) |

The two paths are complementary, matching the roofline expectation: at
small batch both are memory-bound and equally fast, but as the batch
grows the W4A16 advantage evaporates (bs=32: +0.2%; the Marlin kernel
still pays FP16 compute plus dequantization) while W4A4 keeps a solid
lead by *computing* in FP4. The accuracy cost of quantizing activations
was +1.0% PPL over W4A16 on the 0.5B proxy (1.1612 vs 1.1492, coherent
generations).

## Characteristics

- **Hardware-aligned formats** -- matches the FP4/FP8 tensor-core formats of NVIDIA
  Blackwell GPUs and the OCP MX specification
- **No calibration required by default** -- direct rounding works without data;
  `use_hessian=True` adds calibration-based error compensation
- **Two deployment modes** -- fake-quant checkpoints (FP16 weights, exact HF parity,
  for accuracy evaluation) and vLLM-native compressed-tensors checkpoints (packed
  low-precision weights, real memory/speed benefits)
- **NVFP4 vs MXFP4** -- NVFP4's finer blocks (16) and E4M3 scales typically achieve a
  lower quantization error than MXFP4's power-of-two scales with blocks of 32

## Implementation Notes

### Encoding summary

| Format  | Element grid                                        | Scale format            | Block size | Extra scale        |
|---------|-----------------------------------------------------|-------------------------|------------|--------------------|
| `nvfp4` | E2M1 \(\{0, \pm 0.5, \pm 1, \pm 1.5, \pm 2, \pm 3, \pm 4, \pm 6\}\) | FP8 E4M3 (max 448)      | 16         | per-tensor FP32    |
| `mxfp4` | E2M1 (same grid)                                    | E8M0 (power of two)     | 32         | none               |
| `fp8`   | E4M3 (max 448)                                      | FP32                    | per-channel| none               |

Blocks run along the input dimension of the `(out_features, in_features)` weight, so
each block shares a scale with elements that are contracted together in the matrix
product. `in_features` must be divisible by the block size (validated; no padding is
performed). Codes are stored as indices into the 15-entry symmetric E2M1 grid
(`torch.int8`), and block scales are stored decoded as FP32 tensors.

### Element rounding

Rounding onto the E2M1 grid is sign-symmetric round-to-nearest. Exact midpoint ties
(e.g. 0.25, 2.5, 5.0 after scaling) resolve to the candidate with an even mantissa
(round-half-to-even), matching IEEE-style casts, so
\(\mathrm{round}(-x) = -\mathrm{round}(x)\) holds everywhere. Values outside the grid
range saturate to \(\pm 6\). E4M3 rounding is a round-trip cast through
`torch.float8_e4m3fn` with saturation at \(\pm 448\); it runs on CPU and GPU alike.

### Two-level scaling derivation (NVFP4)

The per-tensor scale is chosen so that the largest weight magnitude maps to the largest
representable product of a block scale and an element:

\[
\frac{\max |W|}{s_{\text{tensor}}} = 448 \cdot 6
\quad\Longrightarrow\quad
s_{\text{tensor}} = \frac{\max |W|}{448 \cdot 6}
\]

With this choice, \(\max_b |w|_{\max} / (6\, s_{\text{tensor}}) \le 448\) for every
block, so the E4M3 block scales never saturate and use the full E4M3 dynamic range.
All-zero blocks (and all-zero tensors) get scale 1 as a division guard; their codes map
to zero regardless of the scale, so reconstruction is exactly zero.

### Hessian compensation loop

With `use_hessian=True`, the quantizer follows the GPTQ column-sequential structure
(`onecomp.quantizer.gptq`), replacing integer rounding with rounding onto the target
floating-point grid:

```text
Hinv = upper Cholesky factor of (H + damp I)^-1
for each column block [i1, i2):            # aligned to scale-block boundaries
    for col in [i1, i2):
        if col is a scale-block boundary:   # nvfp4 / mxfp4
            recompute the block scale from the current
            (error-compensated) weights of that block
        q    = round_to_grid(w_col / scale) * scale
        err  = (w_col - q) / Hinv[col, col]
        W[:, col:] -= err * Hinv[col, col:]   # propagate within the block
    W[:, i2:] -= Err_block @ Hinv[i1:i2, i2:] # propagate to later columns
```

The outer loop width is aligned to a multiple of the scale-block size so a scale block
never straddles two outer blocks. Recomputing each block scale at its boundary lets the
scale reflect the error-compensated weights rather than the original ones. For `fp8`,
per-channel scales are fixed from the original weights before the loop (there are no
block boundaries in the column direction). The per-tensor NVFP4 scale is likewise fixed
upfront from the original weights.

### Native checkpoint packing

`save_vllm_native_model` stores FP4 codes in the IEEE-style E2M1 bit layout
(`sign | exponent(2) | mantissa(1)`) packed two per byte with the
even-indexed element in the low nibble, and MXFP4 scales as biased E8M0
exponent bytes (`exponent + 127`) — the exact conventions of
`compressed-tensors` `*-pack-quantized` checkpoints. NVFP4 global scales
are stored as `1 / s_tensor` (the multiplier convention used by
compressed-tensors). Reconstruction from the packed tensors is covered by
unit tests that compare against `FloatQuantResult.compute_dequantized_weight`.

### Known limitations

- **MXFP4 native path is weight-only (W4A16)** -- NVFP4 supports both
  W4A16 (default) and W4A4 (with `input_global_scales`); MXFP4 runs
  through the FP4 Marlin kernel only, as vLLM's compressed-tensors MXFP4
  scheme does not quantize activations.
- **W4A4 activation scales are static per layer** -- the activation
  *global* scale is calibrated offline from AbsMax or percentile
  calibration texts; inputs exceeding the calibrated range saturate. The
  per-block-16 activation scales are computed at runtime and adapt to
  each token.
- **No padding** -- layers whose `in_features` is not divisible by the block size are
  rejected rather than padded.
- **Activations are not quantized during evaluation** -- only weights are
  fake-quantized in the HF evaluation path (the native FP8 export enables
  dynamic per-token activation quantization inside vLLM).
