# macOS / Apple Silicon (MPS)

OneComp supports GPTQ quantization and Hugging Face `generate()` inference on macOS
with Apple Silicon GPUs via PyTorch MPS (`device="mps"`).

vLLM serving and GemLite kernels require Linux with an NVIDIA GPU. On Mac, use
[Transformers inference](#inference-with-transformers) after saving a quantized model.

## Supported Features

With `device="mps"`, calibration and model forward passes can use MPS, but several
quantization steps intentionally run on CPU for performance. See
[MPS Device Placement (GPTQ vs QEP)](#mps-device-placement-gptq-vs-qep) for details.

| Feature | Supported | Where it runs on MPS |
|---------|:---------:|----------------------|
| GPTQ quantization | Yes | Forwards on MPS; **column-wise GPTQ loop** (incl. inverse-Hessian Cholesky) on **CPU** |
| AutoBit (GPTQ-only candidates) | Yes | Same as GPTQ for per-layer quantization; bitwidth assignment (ILP) on CPU |
| QEP (Quantization Error Propagation) | Yes | Per-layer correction (e.g. `weight @ delta_hatX`) on MPS; **Cholesky solve** on **CPU** (once per layer) |
| Calibration forward passes | Yes | MPS |
| `load_quantized_model` + `generate()` | Yes | MPS (when available) |
| vLLM / GemLite serving | No | Linux + CUDA only |
| DBF, RTN, JointQ, and other quantizers | No | — |
| AutoBit DBF fallback | No | — |
| Multi-GPU quantization | No | — |

## Installation

See [Installation](../getting-started/installation.md#step-1-install-pytorch) for
pip and `uv` setup. On macOS:

- **pip users:** install PyTorch from PyPI (default wheels include MPS).
- **uv users:** run `uv sync --extra mps --extra dev` (do not use CUDA extras or `--extra cpu`).

Verify MPS:

```python
import torch
print(torch.backends.mps.is_available())
```

## Quick Start

`Runner.auto_run` and the `onecomp` CLI default to `device="cuda:0"`. On Mac you must
set `device="mps"` explicitly.

VRAM auto-detection (`wbits=None` without `total_vram_gb`) queries CUDA only. On MPS,
pass `total_vram_gb` with your unified memory budget (e.g. 16 or 32):

=== "Python"

    ```python
    from onecomp import Runner

    Runner.auto_run(
        model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        device="mps",
        total_vram_gb=16,
    )
    ```

=== "CLI"

    ```bash
    onecomp TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
        --device mps --total-vram-gb 16
    ```

For manual configuration:

```python
from onecomp import ModelConfig, Runner, GPTQ, setup_logger

setup_logger()

model_config = ModelConfig(
    model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    device="mps",
)
gptq = GPTQ(wbits=4, groupsize=128)

runner = Runner(model_config=model_config, quantizer=gptq, qep=True)
runner.run()
```

## MPS Device Placement (GPTQ vs QEP)

With `device="mps"`, calibration and model forward passes can run on the GPU.
The implementation splits work as follows:

- **GPTQ (`run_gptq`):** The Hessian and weights are moved to **CPU** for the full
  column-wise loop (including inverse-Hessian Cholesky). If that loop stayed on MPS,
  `quantize()` would call `maxq.item()` once per column; each call triggers
  **per-column host sync** (wait for pending MPS ops, then read one scalar—not a full
  Hessian/weight copy every column)—often several times slower than CPU on Apple Silicon.
  Keeping GPTQ on CPU avoids that overhead. With `mse=True`, `find_params` also calls
  `quantize()` in a grid loop and benefits from the same CPU placement.

- **QEP weight correction (`adjust_weight`, when QEP runs—typically `qep=True`):**
  Per-layer work stays on **MPS** (e.g. `weight @ delta_hatX`, diagonal damping).
  Only the Cholesky solve uses CPU via `_safe_cholesky_and_solve` (one solve per layer,
  not per column). The subsequent GPTQ step still uses the CPU path above.

## Inference with Transformers

After quantization, load the saved model and run text generation with Transformers.
`load_quantized_model()` places the model on MPS automatically when available
(via `get_default_device()`).

```python
from onecomp import load_quantized_model

model, tokenizer = load_quantized_model("./output/quantized_model")

inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

For high-throughput serving or Open WebUI integration, use vLLM on a Linux machine
with an NVIDIA GPU. See the [vLLM Inference guide](vllm-inference.md).

## Limitations and Validation

`Runner.check()` enforces MPS constraints before quantization:

- Only **GPTQ** quantizers are allowed (or **AutoBitQuantizer** whose candidates are all GPTQ).
- **AutoBit DBF fallback** is rejected when the target bitwidth would require DBF-only assignment.
- **`multi_gpu=True`** is not supported.

To avoid DBF fallback on MPS, either set an explicit `wbits` within the GPTQ candidate range
or ensure VRAM estimation yields a bitwidth that does not trigger DBF-only paths.

## See Also

- [Installation](../getting-started/installation.md) — macOS pip / `uv` setup
- [CLI Reference](cli.md) — `--device mps` and `--total-vram-gb`
- [Configuration](configuration.md) — `ModelConfig.device`
- [Examples](examples.md#load-a-saved-quantized-model) — save / load workflow
