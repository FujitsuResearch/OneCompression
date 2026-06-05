# RTN (Round-To-Nearest)

RTN is the simplest quantization method. It rounds each weight to the nearest quantization
level without using calibration data or Hessian information.

## Algorithm

For each weight element \(w\):

\[
\hat{w} = \text{clamp}\left(\left\lfloor \frac{w}{s} \right\rceil + z,\ 0,\ 2^b - 1\right) \cdot s - z \cdot s
\]

where:

- \(s\) is the scale factor
- \(z\) is the zero point
- \(b\) is the bit-width
- \(\lfloor \cdot \rceil\) denotes rounding to the nearest integer

The integer level range is always \([0, 2^b - 1]\) regardless of `sym`.

- **Symmetric** (`sym=True`): max-abs symmetrisation \(x_{\max} = \max(|x_{\min}|, x_{\max})\), with zero point at \((2^b - 1 + 1) / 2\). This aligns with `GPTQExcecutor`.
- **Asymmetric** (`sym=False`): range includes zero (\(x_{\min} \le 0 \le x_{\max}\)), zero point = \(\lfloor -x_{\min} / s \rceil\).

When `mse=True`, an MSE grid search is performed to find the optimal clipping range
that minimises the Lp-norm reconstruction error.

RTN serves as a **baseline** for comparing more sophisticated quantization algorithms.

## Parameters

| Parameter    | Type    | Description                                          | Default  |
|-------------|---------|------------------------------------------------------|----------|
| `wbits`      | `int`   | Quantization bit-width                              | `4`      |
| `groupsize`  | `int`   | Group size for group-wise quantization (-1 = none)  | `-1`     |
| `sym`        | `bool`  | Symmetric quantization                              | `False`  |
| `mse`        | `bool`  | Enable MSE grid search for optimal clipping         | `False`  |
| `norm`       | `float` | Lp norm exponent for MSE search                     | `2.4`    |
| `grid`       | `int`   | Number of candidate shrink levels for MSE search    | `100`    |

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

## Save and Load

RTN uses the same tensor format as GPTQ (`qweight`/`scales`/`qzeros`), so it emits
`quant_method="gptq"` and can be saved with the standard OneComp save API, reloaded
with `load_quantized_model()`, and served with vLLM's built-in GPTQ plugin:

```python
runner.save_quantized_model("./output/rtn_model")

# Load later with OneComp
from onecomp import load_quantized_model
model, tokenizer = load_quantized_model("./output/rtn_model")
```

See [vLLM Inference](../user-guide/vllm-inference.md) for serving details.

!!! note "Bit-width and vLLM"
    Use `wbits` in {2, 3, 4, 8} for vLLM serving. RTN itself accepts a wider
    range of `wbits`, but GPTQ-compatible bit packing and vLLM serving are limited
    to these bit-widths.

!!! warning "Rotation-preprocessed RTN models cannot be served with vLLM"
    The SpinQuant-style [Rotation Preprocessing + RTN](../user-guide/examples.md#rotation-preprocessing-rtn)
    flow produces models that vLLM cannot serve, because vLLM does not apply the online
    Hadamard transform on `down_proj` inputs. Such models remain loadable with
    `load_quantized_model()`, which auto-registers the Hadamard hooks.

## Characteristics

- **No calibration data required** -- quantization is performed directly on the model weights
- **Very fast** -- no optimization or iterative processing
- **Lower quality** -- compared to GPTQ or other Hessian-based methods, RTN produces higher quantization error
- **Useful as a baseline** -- provides a lower bound on expected quantization quality

## When to Use RTN

- Quick experiments where calibration data is not available
- Comparing against more advanced methods as a baseline
- High bit-width quantization (e.g., 8-bit) where the difference from optimal is small
