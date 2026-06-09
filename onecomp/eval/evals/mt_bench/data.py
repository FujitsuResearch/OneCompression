"""JSONL loaders for the MT-Bench dataset.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)


def load_questions(question_file: str | Path) -> list[dict]:
    out: list[dict] = []
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_judge_prompts(prompt_file: str | Path) -> dict[str, dict]:
    prompts: dict[str, dict] = {}
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                prompts[obj["name"]] = obj
    return prompts


def load_reference_answers(ref_dir: str | Path, judge_model: str) -> dict[int, dict]:
    """Load reference answers for math/reasoning/coding categories."""
    refs: dict[int, dict] = {}
    for name in [judge_model, "gpt-4o", "gpt-4"]:
        ref_file = Path(ref_dir) / f"{name}.jsonl"
        if ref_file.exists():
            with open(ref_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        refs[obj["question_id"]] = obj
            logger.info("Loaded %d reference answers from %s", len(refs), ref_file.name)
            break
    return refs


def load_answers(answer_file: str | Path) -> dict[int, dict]:
    answers: dict[int, dict] = {}
    with open(answer_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                answers[obj["question_id"]] = obj
    return answers
