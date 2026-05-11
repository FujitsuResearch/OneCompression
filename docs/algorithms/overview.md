# Algorithm Overview

Fujitsu One Compression (OneComp) provides a collection of post-training quantization (PTQ) algorithms for LLMs.
Each algorithm represents a different approach to compressing model weights while preserving
model quality.

## What is Post-Training Quantization?

Post-training quantization converts model weights from high-precision floating-point (e.g., FP16)
to lower-precision representations (e.g., INT4, INT3) after training is complete. This reduces
model size and can accelerate inference without requiring retraining.

The core problem is to find quantized weights \(\hat{W}\) that minimize the error:

\[
\min_{\hat{W}} \| W X - \hat{W} X \|_F^2
\]

where \(W\) is the original weight matrix and \(X\) is the input activation matrix.

## Available Algorithms

| Algorithm | Bit-width | Calibration | Description |
|-----------|-----------|-------------|-------------|
| [**GPTQ**](gptq.md)  | Arbitrary (typically 2--4) | Required | Hessian-based optimal rounding with column-by-column processing |
| [**DBF**](dbf.md)     | ~1.5 (binary) | Required | Double Binary Factorization: \(W \approx A \cdot \text{diag}(d) \cdot B\) |
| [**RTN**](rtn.md)     | Arbitrary | Not required | Round-To-Nearest baseline |
| [**AutoBit**](autobit.md) | Mixed-precision | Required | ILP-based per-layer bit-width assignment under a VRAM budget |
| [**JointQ**](jointq.md) | Arbitrary | Required | Joint optimization of assignments and scale parameters |
| **QuIP**   | Arbitrary | Required | Quantization with Incoherence Processing |
| **ARB**    | Arbitrary | Required | Adaptive Rounding with Binary search |
| **CQ**     | Arbitrary | Required | Combinatorial quantization |
| **QBB**    | Arbitrary | Required | Quantization with Block-wise Balancing |
| **Onebit** | 1-bit     | Required | Extreme 1-bit quantization |

## Quantization Error Propagation (QEP)

[**QEP**](qep.md) is not a standalone quantizer but a **meta-algorithm** that works on top
of any layer-wise quantizer. It compensates for the error that propagates from one layer to
the next during sequential quantization.

QEP can be combined with any quantizer:

```python
runner = Runner(
    model_config=model_config,
    quantizer=GPTQ(wbits=3),
    qep=True,
)
```

## Layer-Projected Coordinate Descent (LPCD)

[**LPCD**](lpcd.md) is a **submodule-level refinement framework** built on top of
layer-wise PTQ. Instead of treating each linear layer independently, LPCD jointly
optimizes related module groups such as Q/K, V/O, MLP up/down, or residual paths,
then projects the refined solution back through the underlying quantizer.

LPCD can be used with or without QEP:

```python
from onecomp import GPTQ, LPCDConfig, Runner

runner = Runner(
    model_config=model_config,
    quantizer=GPTQ(wbits=3, groupsize=128),
    qep=True,
    lpcd=True,
    lpcd_config=LPCDConfig(),
)
runner.run()
```

## Choosing an Algorithm

- **GPTQ** is the recommended default for most use cases (4-bit or 3-bit quantization)
- **GPTQ + QEP** provides the best quality at low bit-widths (3-bit or lower)
- **GPTQ + QEP + LPCD** is useful when you want additional submodule refinement beyond layer-wise PTQ
- **RTN** is useful as a fast baseline or when calibration data is not available
- **DBF** targets extreme compression (~1.5-bit) with binary factorization
