# Evaluation (`onecomp-eval`)

The `onecomp.eval` package runs **serving-time benchmarks** on models loaded through vLLM:
MT-Bench (default: English; answer generation + OpenAI judge scoring) and a Chat Completions throughput benchmark (TTFT / decode tokens per second).

The pipeline starts **one vLLM server**, runs enabled evaluators **serially** as subprocesses, and writes a combined `summary.json` / `summary.csv` under the output directory.

!!! note "Different from `Runner` evaluation"
    `onecomp` / `Runner` can compute **perplexity and zero-shot accuracy** during quantization (`evaluate=True`).
    `onecomp-eval` is a separate harness for **vLLM-served** models. See [Basic Usage](basic-usage.md) for in-runner metrics.

## Installation

Evaluation requires a **source checkout** of this repository: the MT-Bench download scripts and default data directories (`onecomp/eval/data/mt_bench_en/`, etc.) are not published on PyPI (only `conf/*.yaml` is packaged).

=== "uv (recommended)"

    ```bash
    git clone https://github.com/FujitsuResearch/OneCompression.git
    cd OneCompression
    uv sync --extra cu130 --extra eval --extra vllm
    ```

    Use `--extra cu130` with `--extra vllm` (see [vLLM Inference](vllm-inference.md#installation)).

=== "pip (editable)"

    ```bash
    git clone https://github.com/FujitsuResearch/OneCompression.git
    cd OneCompression
    pip install torch --index-url https://download.pytorch.org/whl/cu130
    pip install -e ".[eval,vllm,cu130]"
    ```

Optional extras from `pyproject.toml`:

| Extra   | Purpose                                                         |
|---------|-----------------------------------------------------------------|
| `eval`  | Hydra, OmegaConf, OpenAI client, matplotlib, japanize-matplotlib |
| `vllm`  | vLLM server for answer generation / throughput                  |
| `cu130` | PyTorch built for CUDA 13.0 (use with `vllm` on supported GPUs) |

## Quick start

```bash
bash onecomp/eval/scripts/download_mt_bench_data_en.sh
export OPENAI_API_KEY="sk-..."
onecomp-eval model.path=/path/to/model
```

Results land in `./output/eval/` by default (`summary.json`, per-eval `result.json`, MT-Bench radar chart under `charts/`).

```bash
# Common overrides
onecomp-eval model.path=/path/to/model output_dir=./output/run01
onecomp-eval model.path=/path/to/model \
  evals.mt_bench.enabled=false evals.throughput.enabled=true
```

All settings live in [`onecomp/eval/conf/eval_config.yaml`](https://github.com/FujitsuResearch/OneCompression/blob/main/onecomp/eval/conf/eval_config.yaml).
Defaults and field descriptions are defined in [`onecomp.eval.schema`](../api/eval.md#configuration-schema).

You can also invoke the same entry point as a module:

```bash
python -m onecomp.eval model.path=/path/to/model
```

## Evaluators

| Name         | Default | Description                                                                 |
|--------------|---------|-----------------------------------------------------------------------------|
| `mt_bench`   | on      | MT-Bench (default: English): answer generation → judge scoring (`gpt-4o-2024-08-06` by default) → category scores → radar chart |
| `throughput` | off     | vLLM Chat Completions streaming benchmark (TTFT, ITL, decode tok/s)       |

Disable or enable evaluators with Hydra overrides, for example `evals.mt_bench.enabled=false` or `evals.throughput.enabled=true`.

Quantized models must be loadable by vLLM. See [vLLM Inference](vllm-inference.md) for supported `quant_method` values and installation notes.

## MT-Bench data

MT-Bench data is **not bundled** in OSS releases (licensing). Download once before running `mt_bench`.

**Default locale is English.** The eval pipeline is language-agnostic; switch locales by pointing `evals.mt_bench.data_dir` at a different dataset directory.

| Locale | Script | Default output directory |
|--------|--------|--------------------------|
| English (default) | `download_mt_bench_data_en.sh` | `onecomp/eval/data/mt_bench_en/` |
| Japanese | `download_mt_bench_data_jp.sh` | `onecomp/eval/data/mt_bench_jp/` |

```bash
# English (default — no data_dir override needed if mt_bench_en/ exists)
bash onecomp/eval/scripts/download_mt_bench_data_en.sh
onecomp-eval model.path=/path/to/model

# Japanese — set data_dir when both locales are installed (English is preferred)
bash onecomp/eval/scripts/download_mt_bench_data_jp.sh
onecomp-eval model.path=/path/to/model evals.mt_bench.data_dir=onecomp/eval/data/mt_bench_jp

# Japanese only — if mt_bench_en/ is absent, mt_bench_jp/ is auto-discovered; data_dir optional
bash onecomp/eval/scripts/download_mt_bench_data_jp.sh
onecomp-eval model.path=/path/to/model

# Custom directory
bash onecomp/eval/scripts/download_mt_bench_data_en.sh /custom/english
bash onecomp/eval/scripts/download_mt_bench_data_jp.sh /custom/japanese
```

!!! note "Locale selection vs auto-discovery"
    Bundled lookup prefers `mt_bench_en/` over `mt_bench_jp/`. If you download **both** English and Japanese datasets, pass `evals.mt_bench.data_dir` (or `MT_BENCH_DATA_DIR`) explicitly for Japanese runs. If only `mt_bench_jp/` exists on disk, it is picked up without an override.

Expected layout under `<data_dir>/`:

```
question.jsonl
judge_prompts.jsonl
reference_answer/gpt-4.jsonl    # required; gpt-4o.jsonl optional for gpt-4o-* judges
```

**Data directory resolution** (first match wins):

1. `evals.mt_bench.data_dir=...` (Hydra override)
2. `MT_BENCH_DATA_DIR` environment variable (ignored with a warning if the path lacks `question.jsonl`)
3. Bundled directories under `onecomp/eval/data/`, in order: `mt_bench_en/` → `mt_bench_jp/` → legacy flat `data/` (and repo walk-ups for the same relative paths)

Steps 2–3 apply only when step 1 leaves `data_dir` empty.

**Upstream sources**

- **English:** [lm-sys/FastChat](https://github.com/lm-sys/FastChat) `main` — questions, reference answers, and judge prompts (original 80-question MT-Bench)
- **Japanese:** [Stability-AI/FastChat](https://github.com/Stability-AI/FastChat) `jp-stable` — questions (`question_full.jsonl`, 80 items) and reference answers; [lm-sys/FastChat](https://github.com/lm-sys/FastChat) `main` — judge prompts

## MT-Bench answer post-processing

After each chat completion, [`gen_answer.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/onecomp/eval/evals/mt_bench/gen_answer.py) runs `_strip_thinking_tokens()` on `message.content` before saving `model_answer/*.jsonl` and before appending the turn to the multi-turn conversation.

This is a **string fallback** when control tokens appear in the visible content field. It does **not** replace server-side parsing.

### Recommended: vLLM reasoning parser

When the vLLM server supports it, prefer separating reasoning on the server.

With `onecomp-eval`, pass extra vLLM CLI flags via `inference.extra_args` (see [`eval_config.yaml`](https://github.com/FujitsuResearch/OneCompression/blob/main/onecomp/eval/conf/eval_config.yaml)):

```bash
# Parser name depends on model; see vLLM docs
onecomp-eval model.path=/path/to/model \
  inference.extra_args=[--reasoning-parser,deepseek_r1]
# or: qwen3, etc.
```

For standalone serving outside the harness:

```bash
vllm serve <model> --reasoning-parser deepseek_r1
```

With a reasoning parser, chain-of-thought is exposed as `message.reasoning_content` and `message.content` stays clean. Client-side stripping is **insurance** when parsers are unavailable or the deployment still leaks tokens into `content`.

### Markers handled today (client fallback)

| Format | Markers | Typical models / stacks | Behaviour |
|--------|---------|---------------------------|-----------|
| OpenAI Harmony | `<\|channel\|>`, `<\|message\|>`, `<\|start\|>`, `<\|end\|>`, `<\|return\|>` | GPT-OSS / harmony-style outputs via vLLM | Last `<\|channel\|>` segment; if `<\|message\|>` present, text after the **last** `<\|message\|>`; then remove leftover control tokens |
| Reasoning tag | `<think>…</think>` | DeepSeek-R1, Qwen3 (thinking), QwQ, similar | Remove paired block; keep surrounding answer text |
| Cohere-style | `<\|START_THINKING\|>…<\|END_THINKING\|>`, `<START_THINKING>…<END_THINKING>` | Cohere Command-R+ style | Remove paired block |

**Harmony example**

```text
Input:  <|channel|>analysis<|message|>…<|channel|>final<|message|>Answer text
Output: Answer text
```

Without the `<|message|>` step, the tail would incorrectly stay as `final<|message|>Answer text`.

**Reasoning-tag example**

```text
Input:  <think>internal…</think>Answer text
Output: Answer text
```

### Not covered (client fallback)

- Unpaired markers (e.g. `<|START_THINKING|>` only because of `max_tokens`)
- Non-Harmony asymmetric channel syntax (e.g. `<|channel>thought\n` without symmetric `<|channel|>`)
- Vendor-specific formats not listed above

Enable the appropriate **vLLM `--reasoning-parser`**, adjust the chat template, or extend `_strip_thinking_tokens` in `gen_answer.py` when a format becomes common.

### Adding a new marker

1. Add a paired `(start, end)` tuple to `thinking_block_patterns` inside `_strip_thinking_tokens` in `gen_answer.py`, **or** extend the Harmony branch if it is channel-based.
2. Add a test in [`test_strip_thinking_tokens.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/tests/onecomp/eval/evals/mt_bench/test_strip_thinking_tokens.py) with **Before / After** in the docstring.
3. Update the marker table in this section.

## Judge API key

MT-Bench judging calls the OpenAI API. Provide a key via any of:

- `export OPENAI_API_KEY="sk-..."` (read before the orchestrator sets `OPENAI_API_KEY=EMPTY` for vLLM)
- `export ONECOMP_JUDGE_OPENAI_API_KEY="sk-..."` (explicit judge-only key; useful in SLURM jobs)
- `$LAB_DIR/secrets/openai_api_key.txt`
- `<repo>/secrets/openai_api_key.txt`

The judge subprocess also checks `ONECOMP_JUDGE_OPENAI_API_KEY` first, then the secrets files above.

!!! warning "Never put API keys in Hydra overrides or YAML"
    The orchestrator forwards the judge key to the judge subprocess only.
    Local answer generation uses `OPENAI_API_KEY=EMPTY` for the vLLM HTTP client.

If no key is found, MT-Bench is recorded as `status="skipped"`. The default judge model is `gpt-4o-2024-08-06` (`evals.mt_bench.judge_model`); override as needed. The judge client uses `https://api.openai.com/v1` by default, or set `evals.mt_bench.judge_api_base` for Azure OpenAI and other compatible endpoints.

## Output layout

```
<output_dir>/
├── summary.json / summary.csv
├── mt_bench/result.json          # scores + artifact paths
├── mt_bench/subprocess.log       # generation / judge progress
├── _logs/vllm_server.log
└── charts/mt_bench_radar_*.png
```

Each evaluator writes a [`TaskResult`](../api/eval.md#taskresult) JSON to `<output_dir>/<eval_name>/result.json`.

```bash
tail -f output/eval/mt_bench/subprocess.log   # default output_dir=./output/eval
```

## Python API

```python
from omegaconf import OmegaConf
from onecomp.eval import EvalConfig, ModelConfig, run_pipeline

summary = run_pipeline(OmegaConf.structured(EvalConfig(
    model=ModelConfig(path="/path/to/model"),
    output_dir="./output/eval",
)))
```

More examples: [`onecomp/eval/conf/eval_config.yaml`](https://github.com/FujitsuResearch/OneCompression/blob/main/onecomp/eval/conf/eval_config.yaml).
Full API documentation: [Evaluation API](../api/eval.md).

## Adding an evaluator

To extend the harness:

1. Add a config dataclass in `onecomp/eval/schema.py` and register it on `EvalsConfig`.
2. Add `<name>.enabled: false` under `evals` in `conf/eval_config.yaml`.
3. Create `evals/<name>/adapter.py` and `evals/<name>/run.py` (follow `mt_bench` or `throughput`).
4. Register the adapter in `evals/__init__.py`.

Evaluators run serially while the vLLM server holds the GPU.

## See also

- [vLLM Inference](vllm-inference.md) — quant methods, serving, and `cu130` / `vllm` install
- [Evaluation API](../api/eval.md) — `EvalConfig`, `run_pipeline`, orchestrator classes
