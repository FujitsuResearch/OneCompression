# QEP (Quantization Error Propagation)

QEP is a meta-algorithm that improves any layer-wise quantization method by compensating for
the error that propagates from previously quantized layers to subsequent ones.

!!! abstract "Reference"
    Yamato Arai and Yuma Ichikawa, "Quantization Error Propagation: Revisiting Layer-Wise
    Post-Training Quantization," NeurIPS 2025.
    [OpenReview](https://openreview.net/forum?id=a3l3K9khbL)

## Motivation

Standard layer-wise PTQ quantizes each layer independently using the **original** input
activations. However, after quantizing layer \(l\), the input to layer \(l+1\) is no longer
the original activation -- it is the output of the quantized layer \(l\), which contains
quantization error. This accumulated error degrades quantization quality, especially at low
bit-widths.

## How QEP Works

QEP addresses this by adjusting the weights of each layer **before** quantization to account
for the activation error introduced by previously quantized layers.

For a layer with weight \(W\), original input activations \(X\), and quantized-model input
activations \(\hat{X}\):

1. Compute the activation difference: \(\Delta = X - \hat{X}\)
2. Compute the cross-term: \(\Delta^T \hat{X}\)
3. Solve for a weight correction \(\Delta W\) via the Hessian:

\[
\Delta W = \alpha \cdot (\Delta^T \hat{X}) \cdot H^{-1}
\]

where \(H = \hat{X}^T \hat{X}\) is the Hessian matrix and \(\alpha\) is the correction
strength (`perccorr`).

4. Quantize the adjusted weight \(W + \Delta W\) using the base quantizer (e.g., GPTQ).

## Two Implementations

OneComp provides two QEP implementations, controlled by the `QEPConfig.general` parameter:

### Architecture-aware (default, `general=False`)

- Exploits the structure of transformer blocks (e.g., QKV layers sharing the same input)
- Groups layers that share input activations for efficient Hessian computation
- Processes one transformer block at a time to minimize GPU memory usage
- **Recommended** for Llama-like architectures

### Generic (`general=True`)

- Architecture-independent implementation
- Captures input activations for each layer individually
- Works with any model architecture
- Higher memory consumption and more forward passes

## Usage

### Basic QEP

```python
from onecomp import ModelConfig, Runner, GPTQ

model_config = ModelConfig(model_id="meta-llama/Llama-2-7b-hf", device="cuda:0")
gptq = GPTQ(wbits=3)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    qep=True,
)
runner.run()
```

### Custom QEP Configuration

```python
from onecomp import QEPConfig

qep_config = QEPConfig(
    general=False,              # Architecture-aware (default)
    percdamp=0.01,              # Hessian damping
    perccorr=0.5,               # Correction strength
    device="cuda:0",            # GPU for QEP computation
    exclude_layer_keywords=["mlp.down_proj"],
)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    qep=True,
    qep_config=qep_config,
)
runner.run()
```

### Generic QEP (for non-Llama architectures)

```python
qep_config = QEPConfig(general=True)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    qep=True,
    qep_config=qep_config,
)
runner.run()
```

## Parameters

| Parameter                 | Type        | Description                                      | Default              |
|---------------------------|-------------|--------------------------------------------------|----------------------|
| `general`                 | `bool`      | Use generic (architecture-independent) QEP       | `False`              |
| `percdamp`                | `float`     | Damping percentage for Hessian regularization     | `0.01`               |
| `perccorr`                | `float`     | Correction strength (0 = no correction, 1 = full)| `0.5`                |
| `device`                  | `str`       | GPU device for QEP computation                    | `"cuda:0"`           |
| `exclude_layer_keywords`  | `list[str]` | Layer keywords excluded from error propagation    | `["mlp.down_proj"]`  |

!!! note
    The default `exclude_layer_keywords` is designed for Llama-like architectures. You may need
    to adjust this for other model families.
