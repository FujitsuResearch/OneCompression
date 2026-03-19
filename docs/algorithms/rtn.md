# RTN (Round-To-Nearest)

RTN is the simplest quantization method. It rounds each weight to the nearest quantization
level without using calibration data or Hessian information.

## Algorithm

For each weight element \(w\):

\[
\hat{w} = \text{clamp}\left(\left\lfloor \frac{w - z}{s} \right\rceil, 0, 2^b - 1\right) \cdot s + z
\]

where:

- \(s\) is the scale factor
- \(z\) is the zero point
- \(b\) is the bit-width
- \(\lfloor \cdot \rceil\) denotes rounding to the nearest integer

RTN serves as a **baseline** for comparing more sophisticated quantization algorithms.

## Parameters

| Parameter    | Type   | Description                                          | Default  |
|-------------|--------|------------------------------------------------------|----------|
| `wbits`      | `int`  | Quantization bit-width                              | —        |
| `groupsize`  | `int`  | Group size for group-wise quantization (-1 = none)  | `-1`     |
| `sym`        | `bool` | Symmetric quantization                              | `True`   |

## Usage

```python
from onecomp import ModelConfig, Runner
from onecomp.quantizer.rtn import RTN

model_config = ModelConfig(
    model_id="meta-llama/Llama-2-7b-hf",
    device="cuda:0",
)

rtn = RTN(wbits=4, groupsize=128)

runner = Runner(model_config=model_config, quantizer=rtn)
runner.run()
```

## Characteristics

- **No calibration data required** -- quantization is performed directly on the model weights
- **Very fast** -- no optimization or iterative processing
- **Lower quality** -- compared to GPTQ or other Hessian-based methods, RTN produces higher quantization error
- **Useful as a baseline** -- provides a lower bound on expected quantization quality

## When to Use RTN

- Quick experiments where calibration data is not available
- Comparing against more advanced methods as a baseline
- High bit-width quantization (e.g., 8-bit) where the difference from optimal is small
