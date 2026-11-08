"""

Example: Runtime reasoning compression with the Transformers reference backend

A small experiment: runs a few math questions with policies that shorten a
thinking model's reasoning, then reports per policy the accuracy, reasoning
tokens, total model-token evaluations and wall-clock latency. Every policy
decodes greedily from the same prompts.

  - none:          vanilla generation (identical tokens to disabled compression)
  - fixed:         force </think> after a fixed reasoning-token budget
  - prompt_budget: a per-question budget from a TALE-like length heuristic
  - thinkbrake:    stop at a sentence boundary when </think> nearly wins (logits only)

Edit QUESTIONS and POLICIES to try other settings. The reference backend is
serial and batch-size one. For serving, use the vLLM logits processor (see
vllm_inference/example_reasoning_vllm_inference.py).

A OneComp-quantized model works the same way, e.g.::

    model, tokenizer = QuantizedModelLoader.load_quantized_model("./qwen3-gptq-4bit")

Copyright 2025-2026 Fujitsu Ltd.

"""

from dataclasses import replace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from onecomp.reasoning import ReasoningConfig, ReasoningFormat, estimate_prompt_budget
from onecomp.reasoning.data import PROMPT_TEMPLATE, extract_answer, grade_answer
from onecomp.runtime.transformers import TransformersReasoningAdapter

MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 4096
# (question, expected answer)
QUESTIONS = [
    (
        "What is the smallest positive integer n such that 3n is a perfect square "
        "and 2n is a perfect cube?",
        "108",
    ),
    ("How many positive divisors does 360 have?", "24"),
    ("What is the sum of the interior angles of a convex hexagon, in degrees?", "720"),
    ("If 3x + 7 = 25, what is x?", "6"),
]
POLICIES = [
    ReasoningConfig(),  # policy="none"
    ReasoningConfig(policy="fixed", budget=256),
    ReasoningConfig(policy="fixed", budget=1024),
    # The budget is replaced per question by estimate_prompt_budget() below.
    ReasoningConfig(policy="prompt_budget"),
    # ThinkBrake's Qwen3 setting: gap threshold 0.1 at the paper's temperature 0.6.
    ReasoningConfig(policy="thinkbrake", threshold=0.1, temperature=0.6),
]

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
adapter = TransformersReasoningAdapter(model, tokenizer)

# Qwen3 opens reasoning with <think> and closes it with </think>.
reasoning_format = ReasoningFormat.from_tokenizer(tokenizer)

summary = []
for config in POLICIES:
    name = f"fixed_{config.budget}" if config.policy == "fixed" else config.policy
    correct = reasoning_tokens = model_evaluated_tokens = 0
    latency = 0.0
    for question, expected in QUESTIONS:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            return_dict=False,
        )
        if config.policy == "prompt_budget":
            request_config = replace(config, budget=estimate_prompt_budget(question))
        else:
            request_config = config
        budget = request_config.budget if config.policy in {"fixed", "prompt_budget"} else None
        result = adapter.generate(
            prompt_ids,
            reasoning_format=reasoning_format,
            config=request_config,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        metrics = result.metrics
        is_correct = grade_answer(result.final_answer, expected)
        correct += is_correct
        reasoning_tokens += metrics["reasoning_tokens"]
        model_evaluated_tokens += metrics["model_evaluated_tokens"]
        latency += metrics["latency_s"]
        print(
            f"{name:>13}: answer={extract_answer(result.final_answer)!r} "
            f"expected={expected!r} correct={is_correct} budget={budget} "
            f"reasoning_tokens={metrics['reasoning_tokens']} "
            f"answer_tokens={metrics['answer_tokens']} "
            f"finish_reason={metrics['finish_reason']} latency={metrics['latency_s']:.1f}s"
        )
    summary.append((name, correct, reasoning_tokens, model_evaluated_tokens, latency))

count = len(QUESTIONS)
print(f"\n{'policy':>13}  accuracy  reasoning_tokens  model_evaluated_tokens  latency")
for name, correct, reasoning_tokens, model_evaluated_tokens, latency in summary:
    print(
        f"{name:>13}  {correct}/{count:<6}  {reasoning_tokens / count:>16.0f}  "
        f"{model_evaluated_tokens / count:>22.0f}  {latency / count:>6.1f}s"
    )
print("(token counts and latency are per-question means)")
