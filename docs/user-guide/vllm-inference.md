# vLLM Inference

OneComp provides vLLM plugins for serving quantized models.
The plugins are automatically registered via Python entry points when `onecomp` is installed — no extra configuration is needed.

## Supported Quantization Methods

| Plugin | `quant_method` | Description |
|--------|---------------|-------------|
| DBF | `dbf` | 1-bit Double Binary Factorization. Uses GemLite kernels by default; set `ONECOMP_DBF_NAIVE_LINEAR=1` to use the naive fallback. |
| Mixed-GPTQ | `mixed_gptq` | Per-layer mixed-bitwidth GPTQ. Automatically dispatches to Marlin or Exllama kernels based on bit-width and symmetry. |

!!! warning "Rotation-preprocessed models are not supported"
    Models quantized after rotation preprocessing (`prepare_rotated_model`) cannot be served with vLLM. vLLM kernels do not apply the online Hadamard transform on `down_proj` inputs that rotation-preprocessed models require for correct inference.

## Installation

vLLM is available as an optional dependency:

=== "uv (recommended)"

    ```bash
    uv sync --extra cu130 --extra vllm
    ```

    !!! note "Use `cu130`; older CUDA extras are rejected"
        Recent vLLM releases depend on `torch>=2.10`, whose wheels are only published for the `cu130` PyTorch index. `pyproject.toml` therefore declares `--extra vllm` as conflicting with `cpu`, `cu118`, `cu121`, `cu124`, `cu126`, and `cu128`; combining any of those with `--extra vllm` will fail at lock time. Use `--extra cu130` for vLLM workflows.

=== "pip"

    ```bash
    pip install vllm
    ```

!!! note
    vLLM requires CUDA and a compatible GPU. See the [vLLM documentation](https://docs.vllm.ai/) for detailed installation instructions and system requirements.

!!! warning
    **uv users:** Do not install vLLM with `uv pip install vllm`. Packages installed via `uv pip` are not tracked by the lockfile and will be removed by subsequent `uv sync` or `uv run` commands. Always use `--extra vllm` instead.

## AutoBit + vLLM

When using `AutoBitQuantizer` with mixed-precision candidates (different `wbits` or `groupsize`),
the `enable_fused_groups` parameter must be `True` (the default since v0.5.1) to ensure vLLM compatibility.

vLLM fuses certain layers into a single linear module during inference:

- **qkv_proj**: `q_proj` + `k_proj` + `v_proj`
- **gate_up_proj**: `gate_proj` + `up_proj`

A fused module can only have **one** quantization configuration (one bit-width, one group size).
When `enable_fused_groups=True`, the ILP solver constrains fused-layer constituents to share the same quantizer.

!!! warning "`enable_fused_groups=False` causes vLLM load failures"
    Setting `enable_fused_groups=False` allows the ILP to assign different quantizers
    (different bits or group sizes) to layers within a fused group. The resulting model
    will **fail to load in vLLM** with an error like:
    *"Detected some but not all shards of ... are quantized. All shards of fused layers to have the same precision."*

    Only set `enable_fused_groups=False` if you do **not** intend to serve the model with vLLM.

`Runner.auto_run()` always sets `enable_fused_groups=True`, so models quantized via `auto_run` or the CLI are always vLLM-compatible.

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
[`example/vllm_inference/example_gptq_vllm_inference.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/example/vllm_inference/example_gptq_vllm_inference.py).

### 3. Chat with Open WebUI (optional)

[Open WebUI](https://github.com/open-webui/open-webui) provides a ChatGPT-like browser interface.
Because vLLM exposes an OpenAI-compatible API, Open WebUI can connect to it directly.

#### 3-1. Start the vLLM server

```bash
vllm serve ./Llama-3.1-8B-Instruct-gptq-4bit
```

Keep this terminal open. The server listens on `http://localhost:8000` by default.

#### 3-2. Launch Open WebUI

=== "Docker (recommended)"

    ```bash
    docker run -d -p 3000:8080 \
      --add-host=host.docker.internal:host-gateway \
      --name open-webui \
      ghcr.io/open-webui/open-webui:latest
    ```

    `--add-host` allows the container to reach the vLLM server running on the host (required on Linux; macOS/Windows Docker Desktop resolves it automatically).

=== "pip"

    Open WebUI requires **Python 3.11 or 3.12** (3.13+ is not supported).
    To avoid dependency conflicts with OneComp/vLLM, create a separate virtual environment:

    ```bash
    # Create a dedicated venv (uv auto-downloads Python 3.12 if needed)
    uv venv ~/open-webui-env --python 3.12
    source ~/open-webui-env/bin/activate

    # Install and launch
    uv pip install open-webui
    open-webui serve --port 3000
    ```

    When done, run `deactivate` to leave the venv.
    To uninstall completely, remove the directory: `rm -rf ~/open-webui-env`.

!!! note
    The first launch takes several minutes while Open WebUI runs database migrations and downloads an embedding model (~80 MB). Subsequent launches start in seconds.

#### 3-3. Connect to vLLM

1. Open `http://localhost:3000` in your browser.
2. Create an admin account on first launch.
3. Go to **Admin Panel** → **Settings** → **Connections**.
4. Under **OpenAI API**, set the URL:

    | Setting | Value |
    |---------|-------|
    | URL | `http://host.docker.internal:8000/v1` (Docker) or `http://localhost:8000/v1` (pip) |
    | API Key | `dummy` (any non-empty string) |

5. Click **Save**. The quantized model appears in the model selector.

#### 3-4. Start chatting

Select the model from the dropdown at the top of the chat screen and start a conversation.

!!! tip
    Open WebUI persists chat history, supports multiple conversations, and provides features like system prompt customization and temperature control out of the box.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ONECOMP_DBF_NAIVE_LINEAR` | `0` | Set to `1` to force the naive (non-GemLite) kernel for DBF inference. Useful for debugging or when GemLite is unavailable. |

## Troubleshooting

### `RuntimeError: DeepGEMM backend is not available or outdated`

vLLM unconditionally runs a DeepGEMM (FP8) kernel warmup at engine startup, even for non-FP8 quantization such as GPTQ, DBF, or Mixed-GPTQ. When the optional [`deep_gemm`](https://github.com/deepseek-ai/DeepGEMM) package is not installed, the warmup fails with:

```
RuntimeError: DeepGEMM backend is not available or outdated. Please install or update the `deep_gemm` to a newer version to enable FP8 kernels.
```

OneComp-quantized models do not require DeepGEMM. Disable the FP8 kernel path before launching vLLM:

```bash
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip

# Then launch vllm as usual
vllm serve ./your-quantized-model
# or
python your_vllm_script.py
```

Both variables are read directly by vLLM; OneComp does not interpret them.
