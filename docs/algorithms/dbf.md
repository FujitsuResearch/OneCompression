# DBF (Double Binary Factorization)

DBF is an extreme compression method that approximates weight matrices using binary factors,
achieving approximately 1.5-bit quantization.

## Algorithm

DBF decomposes each weight matrix \(W\) as:

\[
W \approx A \cdot \text{diag}(d) \cdot B
\]

where:

- \(A \in \{-1, +1\}^{m \times k}\) is a binary matrix
- \(d \in \mathbb{R}^k\) is a scaling vector
- \(B \in \{-1, +1\}^{k \times n}\) is a binary matrix

This factorization drastically reduces the storage requirement since the binary matrices
only need 1 bit per element. The scaling vector \(d\) provides the necessary dynamic range.

The optimization is performed using ADMM (Alternating Direction Method of Multipliers)
with optional weight balancing.

## Parameters

| Parameter       | Type    | Description                                  | Default    |
|----------------|---------|----------------------------------------------|------------|
| `target_bits`   | `float` | Target bit-width (e.g., 1.5)               | —          |
| `iters`         | `int`   | Number of ADMM optimization iterations       | `100`      |
| `reg`           | `float` | Regularization coefficient                   | `1.0`      |
| `use_balancing` | `bool`  | Apply weight balancing before factorization   | `True`     |
| `balance_iters` | `int`   | Number of balancing iterations                | `20`       |
| `balance_alpha` | `float` | Balancing alpha parameter                     | `0.5`      |

## Usage

```python
from onecomp import ModelConfig, Runner
from onecomp.quantizer.dbf import DBF

model_config = ModelConfig(
    model_id="meta-llama/Llama-2-7b-hf",
    device="cuda:0",
)

dbf = DBF(target_bits=1.5, iters=100, use_balancing=True)

runner = Runner(model_config=model_config, quantizer=dbf)
runner.run()
```

## Save and Load

DBF models can be saved in a format compatible with the OneComp loader:

```python
runner.save_quantized_model("./output/dbf_model")

# Load later
from onecomp import load_quantized_model
model, tokenizer = load_quantized_model("./output/dbf_model")
```

The `DoubleBinaryLinear` inference layer implements a 5-stage pipeline:
scaling0 -> binary_B -> scaling2 -> binary_A -> scaling4.
