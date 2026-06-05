"""Aggregate per-category MT-Bench scores from the judgment JSONL."""

from __future__ import annotations

import json
from logging import getLogger
from pathlib import Path

from .data import load_questions
from .gen_answer import CATEGORIES

logger = getLogger(__name__)


def show_results(
    *,
    data_dir: Path,
    output_dir: Path,
    model_name: str,
    judge_model: str,
    bench_subdir: str = "mt_bench",
) -> dict:
    """Return {overall, categories, n_scores} and write summary JSON."""
    judgment_file = output_dir / bench_subdir / "model_judgment" / f"{judge_model}_single.jsonl"
    if not judgment_file.exists():
        logger.warning("Judgment file not found: %s", judgment_file)
        return {}

    q_cats = {q["question_id"]: q["category"] for q in load_questions(data_dir / "question.jsonl")}

    scores_by_cat: dict[str, list[float]] = {}
    with open(judgment_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("model") != model_name:
                continue
            score = obj.get("score", -1)
            if score < 0:
                continue
            cat = q_cats.get(obj["question_id"], "unknown")
            scores_by_cat.setdefault(cat, []).append(score)

    if not scores_by_cat:
        logger.warning("No scores found for %s", model_name)
        return {}

    cat_means: dict[str, float] = {}
    all_scores: list[float] = []
    for cat in CATEGORIES:
        if cat in scores_by_cat:
            mean = sum(scores_by_cat[cat]) / len(scores_by_cat[cat])
            cat_means[cat] = round(mean, 4)
            all_scores.extend(scores_by_cat[cat])

    overall = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

    logger.info("MT-Bench Results: %s (judge: %s)", model_name, judge_model)
    for cat in CATEGORIES:
        if cat in cat_means:
            logger.info("  %-15s %.2f  (n=%d)", cat, cat_means[cat], len(scores_by_cat[cat]))
        else:
            logger.info("  %-15s N/A", cat)
    logger.info("  Overall         %.2f  (n=%d)", overall, len(all_scores))

    summary = {
        "model": model_name,
        "judge": judge_model,
        "overall": overall,
        "categories": cat_means,
        "n_scores": len(all_scores),
    }

    summary_file = output_dir / bench_subdir / f"summary_{model_name}.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Summary saved to %s", summary_file)
    return summary
