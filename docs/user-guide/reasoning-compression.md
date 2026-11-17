# Reasoning Compression (Experimental)

`onecomp.reasoning` shortens the *thinking* phase of reasoning models such as
Qwen3 at inference time. It is training-free and opt-in: with the default
`ReasoningConfig()` (`policy="none"`), generation is token-for-token identical
to plain greedy decoding. It is independent of weight quantization and works on
original or OneComp-quantized models.

!!! warning "Research feature"
    This release provides the policy/controller API, the budget baselines and
    ThinkBrake, an adaptive policy that reads only output logits. Probe- and
    attention-based policies are not included yet.

## Policies

| `policy` | Stops when | Extra model compute | Backends | Source |
|---|---|---|---|---|
| `none` | never | none | Transformers, vLLM | vanilla |
| `fixed` | reasoning-token count ≥ `budget` | none | Transformers, vLLM | budget baseline |
| `prompt_budget` | reasoning-token count ≥ a per-prompt `budget` | none | Transformers, vLLM | TALE-like heuristic (not TALE) |
| `thinkbrake` | at a boundary, the best non-end token's log-probability minus `</think>`'s ≤ `threshold` | none (reads logits decoding already produced) | Transformers, vLLM | [ThinkBrake](../reasoning-compression/papers.md#thinkbrake) |

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
its own state. `none`, `fixed`, `prompt_budget` and `thinkbrake` are supported.
The processor supports vLLM 0.21.x only (other versions raise
`RuntimeError` at construction, although the `vllm` extra allows
`>=0.10,<0.22`) and rejects speculative decoding and `n > 1`.

## Configuration

| Field | Default | Used by | Meaning |
|---|---|---|---|
| `policy` | `"none"` | all | One of the policies above |
| `budget` | `2048` | fixed, prompt_budget | Reasoning-token budget |
| `threshold` | `0.1` | thinkbrake | Stop when the logit gap ≤ threshold |
| `temperature` | `1.0` | thinkbrake | Divides the logit gap (ThinkBrake uses the sampling temperature, 0.6 for Qwen3) |
| `boundary` | `"sentence"` | thinkbrake | `sentence`, `transition` or `tokens` (budgets ignore it) |
| `interval` | `128` | `boundary="tokens"` | Check every N reasoning tokens |
| `min_reasoning_tokens` | `0` | thinkbrake | No adaptive check before this many reasoning tokens |

Boundaries select where adaptive policies look: `sentence` after a token ending
in `.`, `!`, `?` or a blank line (token-level approximation), `transition`
before the model emits `Wait` / `Alternatively`, and `tokens` every `interval`
reasoning tokens.

## Examples

- [`example/example_reasoning_compression.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/example/example_reasoning_compression.py) -- Transformers reference: accuracy, reasoning tokens and latency of none / fixed / prompt_budget / ThinkBrake on a few questions
- [`example/vllm_inference/example_reasoning_vllm_inference.py`](https://github.com/FujitsuResearch/OneCompression/blob/main/example/vllm_inference/example_reasoning_vllm_inference.py) -- vLLM batch with per-request policies

## Further reading

- [Design](../reasoning-compression/design.md): package structure and contracts
- [Runtime and vLLM integration](../reasoning-compression/runtime.md)
- [Sources](../reasoning-compression/papers.md): the papers behind each policy and how this implementation differs
