# AutoBit

AutoBit is a mixed-precision quantization method that automatically assigns optimal per-layer
bit-widths under a memory budget using Integer Linear Programming (ILP).

## Algorithm

Given a target average bit-width \(b^*\) (estimated from available VRAM or specified manually),
AutoBit solves for the assignment of each layer \(\ell\) to one of \(K\) candidate quantizers:

\[
\min_{x_{\ell,k}} \sum_{\ell} \sum_{k} c_{\ell,k} \cdot x_{\ell,k}
\quad \text{s.t.} \quad
\sum_{\ell} \sum_{k} \text{bpw}_{\ell,k} \cdot n_\ell \cdot x_{\ell,k}
\le b^* \cdot \sum_\ell n_\ell
\]

where \(x_{\ell,k} \in \{0, 1\}\) indicates whether layer \(\ell\) is assigned to quantizer \(k\),
\(c_{\ell,k}\) is the quantization error cost, \(\text{bpw}_{\ell,k}\) is the effective bits-per-weight,
and \(n_\ell\) is the number of parameters in layer \(\ell\).

Two error metrics are supported:

- **RTN error**: \(c_{\ell,k} = \| W_\ell - \hat{W}_{\ell,k} \|_F^2\)
- **Activation-aware error**: \(c_{\ell,k} = \sum_{q,p} b_q \cdot a_p \cdot (\Delta W_{qp})^2\), where \(a_p\) and \(b_q\) are input and output curvature statistics collected from calibration data

When `enable_fused_groups=True` (the default), equality constraints ensure that vLLM fused layers
(e.g., q/k/v projections, gate/up projections) receive the same quantizer assignment.

For ultra-low-bit targets (\(\le 2\) bpw), AutoBit can optionally inject DBF (Double Binary Factorization)
as a fallback quantizer for the layers where GPTQ candidates would incur excessive error.

## Parameters

| Parameter              | Type    | Description                                                        | Default              |
|------------------------|---------|--------------------------------------------------------------------|----------------------|
| `quantizers`           | `list`  | List of candidate quantizers (e.g., GPTQ at different bit-widths)  | (required)           |
| `target_bit`           | `float` | Target average bit-width                                           | `None` (auto)        |
| `assignment_strategy`  | `str`   | `"activation_aware"`, `"ilp"`, or `"manual"`                       | `"activation_aware"` |
| `calibration_config`   | `CalibrationConfig` | Calibration settings for activation statistics              | auto                 |
| `enable_fused_groups`  | `bool`  | Enforce same quantizer for vLLM fused layers                       | `True`               |
| `auto_dbf`             | `bool`  | Enable DBF fallback for ultra-low-bit targets                      | `True`               |
| `dbf_threshold`        | `float` | Target bit-width threshold below which DBF is injected             | `2.0`                |
| `save_path`            | `str`   | Path to save assignment heatmap visualization                      | `None`               |

## Usage

### VRAM-based automatic bit-width estimation

```python
from onecomp import Runner

runner = Runner.auto_run(
    model_id="meta-llama/Llama-2-7b-hf",
    total_vram_gb=8,
)
```

### Explicit target bit-width with custom candidates

```python
from onecomp import GPTQ, ModelConfig, Runner
from onecomp.quantizer.autobit import AutoBitQuantizer

model_config = ModelConfig(model_id="meta-llama/Llama-2-7b-hf", device="cuda:0")

autobit = AutoBitQuantizer(
    target_bit=3.0,
    quantizers=[GPTQ(wbits=2), GPTQ(wbits=3), GPTQ(wbits=4)],
)

runner = Runner(model_config=model_config, quantizer=autobit)
runner.run()
```

### Mixed bit-width and group size

```python
autobit = AutoBitQuantizer(
    target_bit=3.0,
    quantizers=[
        GPTQ(wbits=2, groupsize=32),
        GPTQ(wbits=4, groupsize=128),
        GPTQ(wbits=4, groupsize=32),
    ],
)
```

## VRAM Estimation

When `target_bit` is not specified, `Runner.auto_run()` uses `estimate_wbits_from_vram()` to
derive the target from available GPU memory. This accounts for model size, KV cache, and
inference overhead to find the largest model that fits in the VRAM budget.

## vLLM Compatibility

AutoBit emits a `mixed_gptq`-compatible `quantization_config`, allowing quantized models to
be served directly with vLLM via the built-in Mixed-GPTQ plugin. The `enable_fused_groups`
constraint ensures that fused layers (qkv_proj, gate_up_proj) have matching bit-widths,
which is required by vLLM.
