"""MT-Bench judge scoring via the OpenAI API (judge model is independent of
the model under test, e.g. GPT-4o).

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
import os
import re
import time
from logging import getLogger
from pathlib import Path

from .data import (
    load_answers,
    load_judge_prompts,
    load_questions,
    load_reference_answers,
)

logger = getLogger(__name__)

NEED_REF_CATS = {"math", "reasoning", "coding"}
SCORE_PATTERN = re.compile(r"\[\[(\d+\.?\d*)\]\]")
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

_SECRETS_REL_PATH = Path("secrets") / "openai_api_key.txt"
_REPO_ROOT_MARKER = "pyproject.toml"


def _find_repo_root(start: Path) -> Path | None:
    for parent in (start, *start.parents):
        if (parent / _REPO_ROOT_MARKER).is_file():
            return parent
    return None


def _candidate_secrets_paths() -> list[Path]:
    """$LAB_DIR/secrets/... first, then the repo-root fallback."""
    candidates: list[Path] = []
    lab_dir = os.environ.get("LAB_DIR", "").strip()
    if lab_dir:
        candidates.append(Path(lab_dir) / _SECRETS_REL_PATH)
    repo_root = _find_repo_root(Path(__file__).resolve())
    if repo_root is not None:
        candidates.append(repo_root / _SECRETS_REL_PATH)
    return candidates


def _resolve_api_key() -> str:
    """Priority: ONECOMP_JUDGE_OPENAI_API_KEY > secrets file."""
    env_key = os.environ.get("ONECOMP_JUDGE_OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    for secrets_file in _candidate_secrets_paths():
        if secrets_file.is_file():
            key = secrets_file.read_text().strip()
            if key:
                logger.info("[JUDGE] API key loaded from %s", secrets_file)
                return key
    return ""


def _resolve_judge_base_url(judge_api_base: str = "") -> str:
    """Return the judge API base URL, ignoring OPENAI_BASE_URL from env.

    The orchestrator injects OPENAI_BASE_URL for vLLM answer generation;
    the judge must not inherit that endpoint.
    """
    if judge_api_base.strip():
        return judge_api_base.strip()
    return _DEFAULT_OPENAI_BASE_URL


def judge_answers(
    *,
    data_dir: Path,
    output_dir: Path,
    model_name: str,
    judge_model: str,
    judge_api_base: str = "",
    bench_subdir: str = "mt_bench",
) -> Path:
    """Score model answers using a judge model via OpenAI API.

    Raises:
        RuntimeError: when no API key is available.
        FileNotFoundError: when no answers were generated.
    """
    from openai import OpenAI

    api_key = _resolve_api_key()
    if not api_key:
        probed = (
            "\n".join(f"     - {p}" for p in _candidate_secrets_paths())
            or "     (none; could not locate $LAB_DIR or repository root)"
        )
        raise RuntimeError(
            "Judge model API key required. Set one of:\n"
            "  1. export OPENAI_API_KEY=<your-key> before running onecomp-eval\n"
            "  2. write the key to one of:\n"
            f"{probed}"
        )

    client_kwargs: dict = {
        "api_key": api_key,
        "base_url": _resolve_judge_base_url(judge_api_base),
    }
    logger.info("[JUDGE] Using base_url=%s", client_kwargs["base_url"])
    client = OpenAI(**client_kwargs)

    question_file = data_dir / "question.jsonl"
    questions = {q["question_id"]: q for q in load_questions(question_file)}

    answer_file = output_dir / bench_subdir / "model_answer" / f"{model_name}.jsonl"
    if not answer_file.exists():
        raise FileNotFoundError(f"Answer file not found: {answer_file}")
    answers = load_answers(answer_file)

    judge_prompts = load_judge_prompts(data_dir / "judge_prompts.jsonl")
    ref_answers = load_reference_answers(data_dir / "reference_answer", judge_model)

    judgment_dir = output_dir / bench_subdir / "model_judgment"
    judgment_dir.mkdir(parents=True, exist_ok=True)
    judgment_file = judgment_dir / f"{judge_model}_single.jsonl"

    _clear_previous_judgments(judgment_file, model_name)

    judged = 0
    for qid, question in sorted(questions.items()):
        answer = answers.get(qid)
        if not answer:
            continue

        category = question.get("category", "")
        answer_turns = answer["choices"][0]["turns"]

        for turn_idx in range(len(answer_turns)):
            turn_num = turn_idx + 1

            user_prompt, judge_prompt = _build_judge_prompt(
                question,
                answer_turns,
                turn_num,
                category,
                judge_prompts,
                ref_answers,
            )
            if user_prompt is None:
                continue

            score, judgment_text = _call_judge(
                client,
                judge_model,
                judge_prompt,
                user_prompt,
            )

            result = {
                "question_id": qid,
                "model": model_name,
                "judge": [judge_model, judge_prompt["name"]],
                "user_prompt": user_prompt[:500],
                "judgment": judgment_text,
                "score": score,
                "turn": turn_num,
                "tstamp": time.time(),
            }
            with open(judgment_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            judged += 1
            logger.info("[JUDGE] Q%d turn%d: score=%s", qid, turn_num, score)

    logger.info("[JUDGE] Done: %d judgments saved to %s", judged, judgment_file)
    return judgment_file


def _clear_previous_judgments(judgment_file: Path, model_name: str) -> None:
    if not judgment_file.exists():
        return
    kept_lines: list[str] = []
    removed = 0
    with open(judgment_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if obj.get("model") == model_name:
                    removed += 1
                else:
                    kept_lines.append(line)
    if removed > 0:
        with open(judgment_file, "w", encoding="utf-8") as f:
            f.writelines(kept_lines)
        logger.info("[JUDGE] Cleared %d previous judgments for %s", removed, model_name)


def _build_judge_prompt(
    question: dict,
    answer_turns: list[str],
    turn_num: int,
    category: str,
    judge_prompts: dict[str, dict],
    ref_answers: dict[int, dict],
) -> tuple[str | None, dict | None]:
    qid = question["question_id"]
    need_ref = category in NEED_REF_CATS
    if turn_num == 1:
        prompt_name = "single-math-v1" if need_ref else "single-v1"
    else:
        prompt_name = "single-math-v1-multi-turn" if need_ref else "single-v1-multi-turn"

    judge_prompt = judge_prompts.get(prompt_name)
    if not judge_prompt:
        return None, None

    if turn_num == 1:
        kwargs = {
            "question": question["turns"][0],
            "answer": answer_turns[0],
        }
        if need_ref and qid in ref_answers:
            kwargs["ref_answer_1"] = ref_answers[qid]["choices"][0]["turns"][0]
    else:
        kwargs = {
            "question_1": question["turns"][0],
            "question_2": question["turns"][1],
            "answer_1": answer_turns[0],
            "answer_2": answer_turns[1] if len(answer_turns) > 1 else "",
        }
        if need_ref and qid in ref_answers:
            ref_turns = ref_answers[qid]["choices"][0]["turns"]
            kwargs["ref_answer_1"] = ref_turns[0]
            kwargs["ref_answer_2"] = ref_turns[1] if len(ref_turns) > 1 else ""

    user_prompt = judge_prompt["prompt_template"].format(**kwargs)
    return user_prompt, judge_prompt


def _call_judge(
    client,
    judge_model: str,
    judge_prompt: dict,
    user_prompt: str,
    *,
    max_retries: int = 3,
) -> tuple[float, str]:
    score = -1.0
    judgment_text = ""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": judge_prompt["system_prompt"]},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=2048,
            )
            judgment_text = response.choices[0].message.content
            match = SCORE_PATTERN.search(judgment_text or "")
            if match:
                score = float(match.group(1))
            break
        except Exception as e:  # noqa: BLE001
            logger.warning("[JUDGE] API error (attempt %d): %s", attempt + 1, e)
            time.sleep(5 * (attempt + 1))
    return score, judgment_text
