# JointQ

JointQ is a post-training quantization method that jointly optimizes integer weight assignments
and scale parameters to minimize the layer-wise reconstruction error.

## Algorithm

For each linear layer, JointQ minimizes:

\[
\min_{\hat{W}} \| Y - \hat{W} X^T \|_F^2
\]

where \(Y = WX^T\) is the full-precision output. Unlike GPTQ which quantizes column-by-column,
JointQ optimizes weight assignments and scale/zero-point parameters simultaneously using
local search.

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
  Only supported with `lambda_mode="fixed_lambda"`.

JointQ provides two lambda selection modes, selected by `lambda_mode`.

#### Fixed lambda mode (default)

Uses a single fixed \(\lambda\) for all layers (`lambda_mode="fixed_lambda"`).
The default strength is `regularization_lambda=0.1`.

#### Incremental lambda mode

`lambda_mode="incremental_lambda"` tries increasing \(\lambda\) values from
`lambda_list` for each layer, keeping the solution as long as it improves
weight error without substantially degrading output error.

For each layer, the algorithm:

1. Quantizes with the first (smallest) \(\lambda\).
   - If the first candidate uses `lambda=0.0` and its relative weight error
     is extremely large, JointQ can skip that candidate and move to the next
     lambda via `incremental_initial_skip_ew_threshold`.
2. For each subsequent \(\lambda\), re-quantizes using the previous accepted
   solution as a warm start.
3. Accepts the candidate if the relative weight error \(E_w\) decreases
   without the relative output error \(E_y\) worsening beyond tolerance.
4. Stops at the first rejection and returns the last accepted solution.

The acceptance criteria are:

- Both \(E_w\) and \(E_y\) decreased → accept.
- \(E_w\) increased → reject.
- \(E_y\) worsened within `incremental_eps_y` and \(E_w\) improved by at
  least `incremental_eps_w` → accept.
- Otherwise → reject.

Where:

- \(E_w = \|W_q - W\|_F^2 \;/\; \|W\|_F^2\)
- \(E_y = \|(W_q - W) X^T\|_F^2 \;/\; \|W X^T\|_F^2\)

## Parameters

| Parameter               | Type           | Description                                                | Default  |
|-------------------------|----------------|------------------------------------------------------------|----------|
| `bits`                  | `int`          | Quantization bit-width (1--4)                              | `4`      |
| `symmetric`             | `bool`         | Symmetric quantization                                     | `False`  |
| `group_size`            | `int` or `None`| Group size for group-wise quantization (None = per-channel) | `128`    |
| `lambda_mode`           | `str`          | `"fixed_lambda"` or `"incremental_lambda"`                 | `"fixed_lambda"` |
| `regularization_lambda` | `float` or `None` | Tikhonov regularization strength (fixed mode)          | `0.1`    |
| `regularization_mode`  | `str`              | `"identity"` (λI) or `"diagonal"` (λ·diag(a), fixed mode only) | `"diagonal"` |
| `regularization_gamma` | `float`            | Exponent for diagonal weights (`"diagonal"` mode)      | `0.5`    |
| `lambda_list`           | `list[float]` or `None` | Lambda values to try (incremental mode)         | `[0.001, 0.01, ..., 0.5]` |
| `incremental_eps_y`     | `float`        | Max tolerated relative output-error increase               | `0.03`   |
| `incremental_eps_w`     | `float`        | Min required relative weight-error decrease                | `0.10`   |
| `incremental_initial_skip_ew_threshold` | `float` or `None` | Skip initial `lambda=0.0` candidate when Ew is too large | `0.3` |
| `actorder`              | `bool`         | Reorder columns by activation magnitude                    | `False`  |
| `device`                | `torch.device` | Device for computation                                     | `None`   |
| `enable_clip_optimize`  | `bool`         | Enable Clip-Optimize initialization                        | `True`   |
| `enable_clip_optimize_ep`| `bool`        | Enable Clip-Optimize with Error Propagation initialization | `False`  |
| `enable_gptq`           | `bool`         | Enable GPTQ initialization                                 | `True`   |
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

### With incremental lambda

```python
jointq = JointQ(
    bits=4,
    group_size=128,
    lambda_mode="incremental_lambda",
)
```

### With identity regularization

```python
jointq = JointQ(
    bits=4,
    group_size=128,
    regularization_mode="identity",
)
```

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

## Notes

- JointQ requires GPU for computation (CUDA-based local search).
- Group-wise quantization (`group_size > 0`) is recommended for accuracy.
  Set `group_size=None` for per-channel quantization.
- JointQ currently supports dequantized-model evaluation only (not packed quantized inference).
- Incremental lambda mode runs `quantize()` multiple times per layer (once per
  lambda value until rejection), so quantization time increases compared to
  fixed lambda mode.
