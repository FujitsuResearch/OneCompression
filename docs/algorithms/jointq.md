# JointQ

JointQ is a post-training quantization method that jointly optimizes integer weight assignments
and scale parameters to minimize the layer-wise reconstruction error.

## Algorithm

For each linear layer, JointQ minimizes:

\[
\min_{\hat{W}} \| Y - \hat{W} X^T \|_F^2
\]

where \(Y = WX^T\) is the full-precision output. Unlike GPTQ which quantizes column-by-column,
JointQ optimizes weight assignments and scale parameters simultaneously using local search.
For asymmetric quantization, the zero-point is determined during initialization and remains
fixed during local search.

The weight is decomposed as:

\[
\hat{W}_{i, g} = s_{i,g} \cdot (a_{i,g} - z_{i,g})
\]

where \(s\) is the scale, \(z\) is the zero-point, and \(a\) is the integer assignment,
with group index \(g\) for group-wise quantization.

### Initialization Strategies

JointQ supports multiple initialization strategies for the local search.
Each strategy can be independently enabled or disabled:

1. **Clip-Optimize** (`enable_clip_optimize`): Finds optimal clipping range, then quantizes
2. **Clip-Optimize with Error Propagation** (`enable_clip_optimize_ep`): Adds GPTQ-style error propagation to initialization
3. **GPTQ** (`enable_gptq`): Uses OneComp GPTQ solution as the starting point for joint optimization

By default, Clip-Optimize and GPTQ are enabled, while Clip-Optimize with Error
Propagation is disabled. The best solution among all enabled strategies is
selected row-by-row based on the reconstruction error.

### Regularization

To prevent overfitting to calibration data, JointQ applies Tikhonov regularization:

\[
X^T X + n \lambda R
\]

where \(\lambda\) controls the regularization strength and \(R\) is the
regularization matrix controlled by `regularization_mode`:

- **`"identity"`**: \(R = I\). Standard Tikhonov — all input
  dimensions are protected equally.
- **`"diagonal"`** (default): \(R = \mathrm{diag}(a)\) where
  \(a_i = \bigl(\mathrm{diag}(X^TX)_i \;/\; \mathrm{mean}(\mathrm{diag}(X^TX))\bigr)^{\gamma}\).
  Columns with larger activations receive stronger regularization while
  less important columns are given more freedom, reducing over-protection.

JointQ uses a single fixed \(\lambda\) for all layers when
`regularization_lambda` is set. By default, `regularization_lambda=None`
uses an unregularized first pass.

For finer control, set `regularization_lambda_by_layer` to override the
lambda per linear layer type. Keys may be full module names or suffixes such
as `"self_attn.q_proj"` and `"mlp.down_proj"`. Full-name matches take
priority; otherwise the longest matching suffix is used. An explicit per-layer
`0.0` disables regularization for the first pass and remains eligible for the
fallback retry described below.

When `regularization_lambda` is disabled (`None` or `0.0`), JointQ can
automatically retry a layer with a small fallback lambda if the unregularized
quantized weight appears unstable. The retry is triggered when
\(\|W - \hat{W}\|_F^2 / \|W\|_F^2\) is greater than or equal to
`regularization_retry_threshold`. Set `regularization_retry_lambda=None` to
disable this fallback.

## Parameters

| Parameter               | Type           | Description                                                | Default  |
|-------------------------|----------------|------------------------------------------------------------|----------|
| `bits`                  | `int`          | Quantization bit-width (1--4)                              | `4`      |
| `symmetric`             | `bool`         | Symmetric quantization                                     | `False`  |
| `group_size`            | `int` or `None`| Group size for group-wise quantization; must divide the layer's input dimension (`None` = per-channel) | `128`    |
| `regularization_lambda` | `float` or `None` | Tikhonov regularization strength                       | `None`   |
| `regularization_lambda_by_layer` | `dict[str, float]` or `None` | Per-layer lambda overrides by full module name or suffix | `None` |
| `regularization_mode`  | `str`              | `"identity"` (λI) or `"diagonal"` (λ·diag(a))          | `"diagonal"` |
| `regularization_gamma` | `float`            | Exponent for diagonal weights (`"diagonal"` mode)      | `0.5`    |
| `regularization_retry_lambda` | `float` or `None` | Fallback lambda used only when regularization is disabled and the weight-error threshold is exceeded | `0.01` |
| `regularization_retry_threshold` | `float` or `None` | Retry threshold for \(\|W - \hat{W}\|_F^2 / \|W\|_F^2\) | `0.2` |
| `actorder`              | `bool`         | Reorder columns by activation magnitude                    | `False`  |
| `device`                | `torch.device` | Device for computation                                     | `None`   |
| `enable_clip_optimize`  | `bool`         | Enable Clip-Optimize initialization                        | `True`   |
| `enable_clip_optimize_ep`| `bool`        | Enable Clip-Optimize with Error Propagation initialization | `False`  |
| `enable_gptq`           | `bool`         | Enable GPTQ initialization                                 | `True`   |
| `gptq_use_regularized_hessian` | `bool` | Use the regularized Hessian for GPTQ initialization when regularization is active | `True` |
| `gptq`                  | `GPTQ` or `None` | Custom GPTQ instance for initial solution generation    | `None`   |

## Usage

### Basic 4-bit quantization

```python
from onecomp import JointQ, ModelConfig, Runner

model_config = ModelConfig(
    model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    device="cuda:0",
)
jointq = JointQ(bits=4, group_size=128)

runner = Runner(model_config=model_config, quantizer=jointq, qep=False)
runner.run()
```

### With identity regularization

```python
jointq = JointQ(
    bits=4,
    group_size=128,
  regularization_lambda=0.1,
    regularization_mode="identity",
)
```

### With per-layer regularization

Use `regularization_lambda_by_layer` when attention and MLP projections need
different regularization strengths:

```python
jointq = JointQ(
    bits=4,
    group_size=128,
    regularization_lambda_by_layer={
        "self_attn.q_proj": 0.0,
        "self_attn.k_proj": 0.0,
        "self_attn.v_proj": 0.0,
        "self_attn.o_proj": 0.0,
        "mlp.gate_proj": 0.05,
        "mlp.up_proj": 0.05,
        "mlp.down_proj": 0.01,
    },
)
```

Suffix keys match all corresponding layers, for example
`"mlp.up_proj"` matches `"model.layers.3.mlp.up_proj"`. If you need to
override only one layer, use its full module name.

### With activation ordering

```python
jointq = JointQ(bits=4, group_size=128, actorder=True)
```

### Symmetric quantization

```python
jointq = JointQ(bits=4, symmetric=True, group_size=128)
```

### With all initialization strategies

```python
jointq = JointQ(
    bits=4,
    group_size=128,
    enable_clip_optimize=True,
    enable_clip_optimize_ep=True,
    enable_gptq=True,
)
```

### Custom GPTQ parameters

```python
from onecomp.quantizer.gptq import GPTQ

jointq = JointQ(
    bits=4,
    group_size=128,
    gptq=GPTQ(wbits=4, groupsize=128, sym=False, mse=True, percdamp=0.05),
)
```

JointQ uses this nested GPTQ quantizer only for its initial solution and always
sets its `bitpack_on_quantize` value to `False`. This overwrites any value
specified on the supplied `GPTQ` instance.

## Save and Load

JointQ reuses GPTQ's scale/zero/assignment structure, so it emits
`quant_method="gptq"` and can be saved with the standard OneComp save API,
reloaded with `load_quantized_model()`, and served with vLLM's built-in GPTQ
plugin:

```python
runner.save_quantized_model("./output/jointq_model")

# Load later with OneComp
from onecomp import load_quantized_model
model, tokenizer = load_quantized_model("./output/jointq_model")
```

!!! note "Bit-width and vLLM"
    Use `bits` in {2, 3, 4} for JointQ models served with vLLM. `bits=1`
    cannot be bit-packed by `GPTQLinear`; if you need to save/load a 1-bit
    JointQ model with OneComp, call `runner.save_quantized_model(..., pack_weights=False)`.
    Such 1-bit models are not vLLM-servable.

See [vLLM Inference](../user-guide/vllm-inference.md) for serving details, and the
[`example/vllm_inference/example_jointq_vllm_inference.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/example/vllm_inference/example_jointq_vllm_inference.py)
script for an end-to-end quantize → save → serve example.

## Notes

- JointQ requires GPU for computation (CUDA-based local search).
- Group-wise quantization (`group_size > 0`) is recommended for accuracy.
  Set `group_size=None` for per-channel quantization.
- JointQ does not support QEP; create the `Runner` with `qep=False`.
