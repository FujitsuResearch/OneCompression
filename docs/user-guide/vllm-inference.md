# vLLM Inference

OneComp provides vLLM plugins for serving quantized models.
The plugins are automatically registered via Python entry points when `onecomp` is installed — no extra configuration is needed.

## Supported Quantization Methods

| Plugin | `quant_method` | Description |
|--------|---------------|-------------|
| DBF | `dbf` | 1-bit Double Binary Factorization. Uses GemLite kernels by default; set `ONECOMP_DBF_NAIVE_LINEAR=1` to use the naive fallback. |
| Mixed-GPTQ | `mixed_gptq` | Per-layer mixed-bitwidth GPTQ. Automatically dispatches to Marlin or Exllama kernels based on bit-width and symmetry. |

## Installation

vLLM is available as an optional dependency:

=== "uv (recommended)"

    ```bash
    uv sync --extra cu128 --extra vllm
    ```

    Replace `cu128` with your CUDA variant (`cu118`, `cu121`, `cu124`, `cu126`, or `cu128`).

=== "pip"

    ```bash
    pip install vllm
    ```

!!! note
    vLLM requires CUDA and a compatible GPU. See the [vLLM documentation](https://docs.vllm.ai/) for detailed installation instructions and system requirements.

!!! warning
    **uv users:** Do not install vLLM with `uv pip install vllm`. Packages installed via `uv pip` are not tracked by the lockfile and will be removed by subsequent `uv sync` or `uv run` commands. Always use `--extra vllm` instead.

## Usage

### 1. Quantize and save a model with OneComp

```python
from onecomp import Runner, ModelConfig
from onecomp.quantizer.gptq import GPTQ

model_config = ModelConfig(model_id="meta-llama/Llama-3.1-8B-Instruct")
quantizer = GPTQ(wbits=4, groupsize=128)
runner = Runner(model_config=model_config, quantizer=quantizer, qep=True)
runner.run()
runner.save_quantized_model("./Llama-3.1-8B-Instruct-gptq-4bit")
```

### 2. Serve with vLLM

There are two ways to use the quantized model with vLLM.

#### Option A: API Server (`vllm serve`)

Launch an OpenAI-compatible HTTP server:

```bash
vllm serve ./Llama-3.1-8B-Instruct-gptq-4bit
```

The server starts at `http://localhost:8000` by default. You can send requests using `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./Llama-3.1-8B-Instruct-gptq-4bit",
    "messages": [{"role": "user", "content": "What is post-training quantization?"}],
    "max_tokens": 128
  }'
```

Or use the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="./Llama-3.1-8B-Instruct-gptq-4bit",
    messages=[{"role": "user", "content": "What is post-training quantization?"}],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

#### Option B: Offline Inference (`vllm.LLM`)

For batch inference without launching a server:

```python
from vllm import LLM, SamplingParams

model_path = "./Llama-3.1-8B-Instruct-gptq-4bit"

llm = LLM(
    model=model_path,
    max_model_len=2048,
    dtype="float16",
    enforce_eager=True,
)

outputs = llm.generate(
    ["What is post-training quantization?"],
    SamplingParams(max_tokens=128, temperature=0.0),
)
print(outputs[0].outputs[0].text)
```

!!! tip
    You do not need to pass `quantization=` explicitly. vLLM reads the `quant_method` from the model's `config.json` and automatically selects the correct OneComp plugin.

!!! warning
    When combining quantization and vLLM inference in a single script, you **must** wrap your code in `if __name__ == "__main__":`. vLLM spawns worker processes that re-import the script, so without this guard the quantization step will run again in each child process.

A complete working example (quantization + vLLM inference) is available at
[`example/example_vllm_inference.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/example/example_vllm_inference.py).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ONECOMP_DBF_NAIVE_LINEAR` | `0` | Set to `1` to force the naive (non-GemLite) kernel for DBF inference. Useful for debugging or when GemLite is unavailable. |
