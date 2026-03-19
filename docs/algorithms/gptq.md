# GPTQ

GPTQ is a Hessian-based post-training quantization method that finds optimal quantized weights
by minimizing the layer-wise output error.

!!! abstract "Reference"
    Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh,
    "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers,"
    ICLR 2023.

## Algorithm

GPTQ formulates quantization as a per-layer optimization problem:

\[
\min_{\hat{W}} \| W X - \hat{W} X \|_F^2
\]

It solves this column-by-column using the inverse Hessian \(H^{-1} = (2 X X^T)^{-1}\).
For each column \(i\):

1. Quantize column \(i\) of \(W\) to the nearest quantization level
2. Compensate the remaining unquantized columns using the Hessian information

This produces significantly better results than simple round-to-nearest (RTN) quantization,
especially at lower bit-widths.

## Parameters

| Parameter    | Type   | Description                                          | Default  |
|-------------|--------|------------------------------------------------------|----------|
| `wbits`      | `int`  | Quantization bit-width                              | —        |
| `groupsize`  | `int`  | Group size for group-wise quantization (-1 = none)  | `-1`     |
| `sym`        | `bool` | Symmetric quantization                              | `True`   |
| `actorder`   | `bool` | Reorder columns by activation magnitude             | `False`  |
| `percdamp`   | `float`| Hessian damping percentage                          | `0.01`   |

## Usage

### Basic 4-bit quantization

```python
from onecomp import GPTQ

gptq = GPTQ(wbits=4, groupsize=128)
```

### 3-bit with activation ordering

```python
gptq = GPTQ(wbits=3, groupsize=128, actorder=True)
```

### Asymmetric quantization

```python
gptq = GPTQ(wbits=4, sym=False)
```

### With QEP for improved quality

```python
from onecomp import Runner, ModelConfig

model_config = ModelConfig(model_id="meta-llama/Llama-2-7b-hf", device="cuda:0")
gptq = GPTQ(wbits=3, groupsize=128)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    qep=True,
)
runner.run()
```

## Group-wise Quantization

When `groupsize > 0`, weights are divided into groups of consecutive columns, and each group
has its own scale and zero-point. This improves quantization accuracy at the cost of slightly
more storage for the quantization parameters.

Typical values:

- `groupsize=128` -- good balance of accuracy and compression
- `groupsize=-1` -- per-channel quantization (no grouping)

## Activation Ordering

When `actorder=True`, columns are reordered by their activation magnitude (Hessian diagonal)
before quantization. Columns with higher activation influence are quantized first, which can
improve accuracy. The permutation is stored so weights can be reconstructed correctly.
