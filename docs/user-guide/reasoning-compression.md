# Reasoning Compression (Experimental)

`onecomp.reasoning` shortens the *thinking* phase of reasoning models such as
Qwen3 at inference time. It is training-free and opt-in: with the default
`ReasoningConfig()` (`policy="none"`), generation is token-for-token identical
to plain greedy decoding. It is independent of weight quantization and works on
original or OneComp-quantized models.

!!! warning "Research feature"
    This release provides the policy/controller API, the budget baselines,
    ThinkBrake (reads only output logits) and the answer-probe policies DEER and
    stability (Transformers reference backend only). Attention-based policies
    are not included yet.

## Policies

| `policy` | Stops when | Extra model compute | Backends | Source |
|---|---|---|---|---|
| `none` | never | none | Transformers, vLLM | vanilla |
| `fixed` | reasoning-token count ≥ `budget` | none | Transformers, vLLM | budget baseline |
| `prompt_budget` | reasoning-token count ≥ a per-prompt `budget` | none | Transformers, vLLM | TALE-like heuristic (not TALE) |
| `thinkbrake` | at a boundary, the best non-end token's log-probability minus `</think>`'s ≤ `threshold` | none (reads logits decoding already produced) | Transformers, vLLM | [ThinkBrake](../reasoning-compression/papers.md#thinkbrake) |
| `deer` | a trial answer's geometric-mean confidence > `confidence_threshold` | answer probe per boundary | Transformers | [DEER](../reasoning-compression/papers.md#deer) |
| `stability` | `stability_window` identical trial answers in a row | answer probe per boundary | Transformers | [Dynasor-inspired](../reasoning-compression/papers.md#certaindex-dynasor) |

When a policy stops, the backend emits the model's own `</think>` marker through
normal decoding and the model then writes its final answer. Budgets stop as soon
as the budget is reached; they do not wait for a sentence boundary.

ThinkBrake checks at each `boundary` (sentence ends by default) and stops when
`</think>` is nearly the most likely next token:

```python
config = ReasoningConfig(policy="thinkbrake", threshold=0.1, temperature=0.6)
```

`temperature` only scales the logit gap (the paper uses Qwen3's sampling
temperature, 0.6); it does not change decoding, which stays greedy. ThinkBrake
needs a single-token `</think>`. With greedy decoding the gap at sentence ends
can stay far above the paper's `threshold`, so ThinkBrake may never trigger;
choose `threshold` on a development set for your model.

DEER and stability run an *answer probe* at each boundary: the backend appends
the cue `\n**Final Answer**\n\boxed` to the current reasoning, greedily decodes a
short trial answer on a temporary branch, and discards it. DEER stops when the
answer's confidence is high; stability stops when the last `stability_window`
answers agree. A probe cut off before the closing brace never counts as
confident or stable. After a stop, the model closes its reasoning with
`</think>` and writes a fresh final answer; the trial answer is not reused.

```python
config = ReasoningConfig(policy="deer", confidence_threshold=0.95, boundary="transition")
```

Probes are extra model work, and every probe token is counted in
`result.metrics["model_evaluated_tokens"]` (`probe_model_evaluated_tokens`,
`probe_history`). By default a probe extends the request's own KV cache and
crops it back afterwards, so it skips re-reading the prefix; pass
`probe_prefix_reuse=False` to the adapter to re-prefill every probe instead (the
conservative cost reference). Reuse lowers model compute, but on a serial
reference backend wall-clock latency is dominated by token-by-token decoding.

`prompt_budget` reads its budget from the config like `fixed`. Substitute the
per-prompt value from `estimate_prompt_budget` for each request:

```python
from dataclasses import replace

from onecomp.reasoning import ReasoningConfig, estimate_prompt_budget

config = replace(ReasoningConfig(policy="prompt_budget"), budget=estimate_prompt_budget(question))
```

## Python API (Transformers reference)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from onecomp.reasoning import ReasoningConfig, ReasoningFormat
from onecomp.runtime.transformers import TransformersReasoningAdapter

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
adapter = TransformersReasoningAdapter(model, tokenizer)

prompt_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 17 * 24? Put the answer in \\boxed{}."}],
    tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False,
)
result = adapter.generate(
    prompt_ids,
    reasoning_format=ReasoningFormat.from_tokenizer(tokenizer),  # <think> ... </think>
    config=ReasoningConfig(policy="fixed", budget=1024),
    max_new_tokens=16000,
)
print(result.final_answer, result.metrics["reasoning_tokens"], result.metrics["latency_s"])
```

The reference backend decodes greedily with batch size one. `result.metrics`
reports reasoning, answer and marker token counts, model-evaluated tokens and
wall-clock latencies; `result.state.interventions` records each stop.

## vLLM

Register the logits processor and attach per-request metadata:

```python
from vllm import LLM, SamplingParams
from onecomp.reasoning import ReasoningConfig, ReasoningFormat
from onecomp.runtime.vllm import VLLMReasoningAdapter

llm = LLM(model="Qwen/Qwen3-8B", logits_processors=[VLLMReasoningAdapter.processor_path])
fmt = ReasoningFormat.from_tokenizer(llm.get_tokenizer())
params = SamplingParams(
    temperature=0,
    max_tokens=16000,
    extra_args=VLLMReasoningAdapter.extra_args(
        ReasoningConfig(policy="fixed", budget=1024), fmt, "req-0"
    ),
)
```

Requests without the metadata are untouched, and each request in a batch keeps
its own state. `none`, `fixed`, `prompt_budget` and `thinkbrake` are supported;
probe policies raise `NotImplementedError`. The processor supports vLLM 0.21.x only (other versions raise
`RuntimeError` at construction, although the `vllm` extra allows
`>=0.10,<0.22`) and rejects speculative decoding and `n > 1`.

## Configuration

| Field | Default | Used by | Meaning |
|---|---|---|---|
| `policy` | `"none"` | all | One of the policies above |
| `budget` | `2048` | fixed, prompt_budget | Reasoning-token budget |
| `threshold` | `0.1` | thinkbrake | Stop when the logit gap ≤ threshold |
| `temperature` | `1.0` | thinkbrake | Divides the logit gap (ThinkBrake uses the sampling temperature, 0.6 for Qwen3) |
| `confidence_threshold` | `0.95` | deer | Stop when probe confidence > threshold |
| `relative_confidence` | `False` | deer | Non-paper variant: also stop on a new confidence maximum |
| `stability_window` | `3` | stability | Number of identical consecutive answers |
| `probe_max_tokens` | `128` | deer, stability | Trial-answer length limit; truncated probes are never confident |
| `boundary` | `"sentence"` | adaptive policies | `sentence`, `transition` or `tokens` (budgets ignore it) |
| `interval` | `128` | `boundary="tokens"` | Check every N reasoning tokens |
| `min_reasoning_tokens` | `0` | adaptive policies | No adaptive check before this many reasoning tokens |

Boundaries select where adaptive policies look: `sentence` after a token ending
in `.`, `!`, `?` or a blank line (token-level approximation), `transition`
before the model emits `Wait` / `Alternatively`, and `tokens` every `interval`
reasoning tokens.

## Examples

- [`example/example_reasoning_compression.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/example/example_reasoning_compression.py) -- Transformers reference: accuracy, reasoning tokens and latency of none / fixed / prompt_budget / ThinkBrake / DEER / stability on a few questions
- [`example/vllm_inference/example_reasoning_vllm_inference.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/example/vllm_inference/example_reasoning_vllm_inference.py) -- vLLM batch with per-request policies

## Further reading

- [Design](../reasoning-compression/design.md): package structure and contracts
- [Runtime and vLLM integration](../reasoning-compression/runtime.md)
- [Sources](../reasoning-compression/papers.md): the papers behind each policy and how this implementation differs
