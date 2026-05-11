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

| Parameter            | Type                       | Description                                                                       | Default |
|---------------------|----------------------------|-----------------------------------------------------------------------------------|---------|
| `target_bits`        | `float`                    | Target bit-width (e.g., 1.5)                                                      | `1.5`   |
| `iters`              | `int`                      | Number of ADMM optimization iterations                                            | `600`   |
| `reg`                | `float`                    | Regularization coefficient                                                        | `3e-2`  |
| `use_balancing`      | `bool`                     | Apply weight balancing before factorization                                       | `True`  |
| `balance_iters`      | `int`                      | Number of balancing iterations                                                    | `40`    |
| `balance_alpha`      | `float`                    | Balancing alpha parameter                                                         | `1.0`   |
| `balance_mode`       | `str`                      | Balancing mode (`"l1"` or `"l2"`)                                                 | `"l1"`  |
| `use_adaptive_rho`   | `bool`                     | Adapt the ADMM penalty parameter ρ during optimization                            | `True`  |
| `mlp_target_bits`    | `Optional[float]`          | Override `target_bits` for layers whose name contains `"mlp"`                     | `None`  |
| `module_target_bits` | `Optional[dict[str,float]]`| Per-layer override of `target_bits`, keyed by exact layer name (highest priority) | `None`  |

## Usage

### Quick Start

For a first run, use a small model (TinyLlama) and a lightweight calibration
configuration. This combination fits in a few GB of GPU memory and is the
recommended way to verify the pipeline end-to-end.

```python
from onecomp import CalibrationConfig, ModelConfig, Runner
from onecomp.quantizer.dbf import DBF

model_config = ModelConfig(
    model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    device="cuda:0",
)
calib_config = CalibrationConfig(
    max_length=512,
    num_calibration_samples=128,
)
dbf = DBF(target_bits=1.5)
runner = Runner(
    model_config=model_config,
    quantizer=dbf,
    calibration_config=calib_config,
)
runner.run()
```

### Recommended Configuration

For production use, run with longer sequences and more calibration samples to
improve quantization quality. DBF holds fp32 ADMM buffers in addition to the
model weights, so the per-forward GPU memory consumption is higher than GPTQ.
To avoid `CUDA out of memory` with the default calibration settings on larger
models such as Llama-2-7B, set `CalibrationConfig.batch_size` to enable
chunked calibration.

```python
from onecomp import CalibrationConfig, ModelConfig, Runner
from onecomp.quantizer.dbf import DBF

model_config = ModelConfig(
    model_id="meta-llama/Llama-2-7b-hf",
    device="cuda:0",
)
calib_config = CalibrationConfig(
    max_length=2048,
    num_calibration_samples=128,  # Increase to 256-512 for higher accuracy
    batch_size=32,                # Tune to GPU free memory (8-32)
)
dbf = DBF(target_bits=1.5)
runner = Runner(
    model_config=model_config,
    quantizer=dbf,
    calibration_config=calib_config,
)
runner.run()
```

!!! note "Tuning `batch_size` to your GPU"
    `CalibrationConfig.batch_size` controls the number of calibration sequences
    forwarded through the model at once, and is the main knob for peak GPU
    memory. Rough guideline:

    - H100 (80 GB): `batch_size=32`
    - A100 (40 GB): `batch_size=16`
    - When sharing the GPU with other processes: `batch_size=8`

    If you still hit `CUDA out of memory`, halve the value until the run
    succeeds.

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
