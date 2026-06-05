"""MT-Bench answer generation via the vLLM OpenAI-compatible HTTP API.

The orchestrator injects OPENAI_BASE_URL / OPENAI_API_KEY before launching
this evaluator. Generation results go to
<output_dir>/mt_bench/model_answer/<model_name>.jsonl.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
import os
import re
import time
from logging import getLogger
from pathlib import Path
from typing import Any

from .data import load_questions

logger = getLogger(__name__)

CATEGORIES = [
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
]

TEMPERATURE_CONFIG = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}


def generate_answers(
    *,
    data_dir: Path,
    output_dir: Path,
    model_name: str,
    max_new_tokens: int = 1024,
    request_timeout_sec: int = 600,
) -> Path:
    """Generate MT-Bench answers and return the JSONL output path."""
    import shortuuid
    from openai import OpenAI

    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not base_url:
        raise RuntimeError(
            "OPENAI_BASE_URL is not set; the orchestrator should inject it "
            "before launching this evaluator."
        )
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

    answer_dir = output_dir / "mt_bench" / "model_answer"
    answer_dir.mkdir(parents=True, exist_ok=True)
    answer_file = answer_dir / f"{model_name}.jsonl"

    questions = load_questions(data_dir / "question.jsonl")
    logger.info("[GEN] Loaded %d questions", len(questions))

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout_sec)
    served_model = _resolve_served_model(client, fallback=model_name)
    logger.info("[GEN] Using base_url=%s served_model=%s", base_url, served_model)

    results: list[dict] = []
    for q in questions:
        qid = q["question_id"]
        category = q.get("category", "")
        turns = q["turns"]
        temperature = q.get(
            "required_temperature",
            TEMPERATURE_CONFIG.get(category, 0.7),
        )

        conversation: list[dict[str, str]] = []
        output_turns: list[str] = []
        for turn_text in turns:
            conversation.append({"role": "user", "content": turn_text})
            generated = _chat(
                client,
                served_model,
                conversation,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            generated = _strip_thinking_tokens(generated)
            output_turns.append(generated)
            conversation.append({"role": "assistant", "content": generated})

        results.append(
            {
                "question_id": qid,
                "answer_id": shortuuid.uuid(),
                "model_id": model_name,
                "choices": [{"index": 0, "turns": output_turns}],
                "tstamp": time.time(),
            }
        )

    _save_answers(results, answer_file)
    return answer_file


def _chat(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_new_tokens: int,
    max_retries: int = 3,
) -> str:
    """Single chat call with exponential backoff retry."""
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 2**attempt
            logger.warning(
                "[GEN] chat API error (attempt %d/%d): %s; retrying in %ds",
                attempt + 1,
                max_retries,
                e,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"chat API call failed after retries: {last_err}")


def _resolve_served_model(client: Any, fallback: str) -> str:
    """Pick a model id from /v1/models (vLLM exposes the local path)."""
    try:
        ids = [m.id for m in client.models.list().data]
        if ids:
            return ids[0]
    except Exception as e:  # noqa: BLE001
        logger.warning("[GEN] /v1/models lookup failed (%s); falling back to %r", e, fallback)
    return fallback


def _strip_thinking_tokens(text: str) -> str:
    """Strip model-internal reasoning markers from the visible answer.

    Client-side fallback when vLLM leaves control tokens in message.content.
    Prefer --reasoning-parser on the vLLM server when supported.

    Processing order and typical models:

    1. Harmony channels (<|channel|>, <|message|>)
       OpenAI Harmony template outputs served without a reasoning parser, e.g.
       GPT-OSS (gpt-oss-20b, gpt-oss-120b) and other models using the
       openai-harmony chat format via vLLM. Takes the final channel and,
       if present, only the text after the last <|message|>.

    2. Paired reasoning blocks (removed when both ends appear in content)
       - <think>…</think> — DeepSeek-R1, Qwen3
         (thinking mode / enable_thinking), and similar stacks when the
         tag is not split into reasoning_content.
       - <|START_THINKING|>…<|END_THINKING|> — Cohere Command-R / Command-R+,
         which emit these tokens in decoded text.
       - <START_THINKING>…<END_THINKING> — same families when the visible
         string omits pipe delimiters around the marker names.

    3. Residual Harmony control tokens (<|channel|>, <|message|>, <|start|>, <|end|>, <|return|>)
      — stray tokens left on harmony-format outputs after steps 1–2.

    Unpaired markers (e.g. <|START_THINKING|> only after max_tokens cut-off)
    are not handled here.
    """
    # Harmony format: GPT-OSS / openai-harmony via vLLM (see docstring above).
    if "<|channel|>" in text:
        parts = text.split("<|channel|>")
        if len(parts) >= 2:
            last_segment = parts[-1]
            if "<|message|>" in last_segment:
                last_segment = last_segment.split("<|message|>")[-1]
            text = last_segment.strip()

    # Paired reasoning blocks
    thinking_block_patterns: list[tuple[str, str]] = [
        ("<think>", "</think>"),  # DeepSeek-R1, Qwen3,
        ("<|START_THINKING|>", "<|END_THINKING|>"),  # Cohere Command-R+
        ("<START_THINKING>", "<END_THINKING>"),  # Cohere-style
    ]
    for start, end in thinking_block_patterns:
        pattern = re.escape(start) + r".*?" + re.escape(end)
        text = re.sub(pattern, "", text, flags=re.DOTALL).strip()

    # Residual Harmony tokens — GPT-OSS / harmony-format outputs.
    harmony_control_tokens = re.compile(r"<\|(?:channel|message|start|end|return)\|>")
    return harmony_control_tokens.sub("", text).strip()


def _save_answers(results: list[dict], answer_file: Path) -> None:
    results.sort(key=lambda x: x["question_id"])
    with open(answer_file, "w", encoding="utf-8") as f:
        for ans in results:
            f.write(json.dumps(ans, ensure_ascii=False) + "\n")

    empty = sum(1 for a in results for t in a["choices"][0]["turns"] if not t.strip())
    total = sum(len(a["choices"][0]["turns"]) for a in results)
    logger.info("[GEN] Generated %d answers (%d/%d empty turns)", len(results), empty, total)
    logger.info("[GEN] Saved to %s", answer_file)
