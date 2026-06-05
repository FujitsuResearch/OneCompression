"""Child entrypoint for the MT-Bench evaluator.

Invoked by the orchestrator as:

    python -m onecomp.eval.evals.mt_bench.run \
        --config <task_config.yaml> \
        --output-dir <run_output_dir> \
        --model-name <model_name>

Pipeline: gen_answer (vLLM HTTP) -> judge (OpenAI) ->
show_result (per-category aggregation) -> radar_chart (optional).

The OpenAI judge phase is skipped (status="skipped") when no API key is
available; this lets --gen-only-style workflows (set
OPENAI_API_KEY="") keep functioning.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import sys
from logging import getLogger
from pathlib import Path

from omegaconf import DictConfig

from ...schema import TaskResult
from ...utils.resources import resolve_mt_bench_data_dir
from ..base import child_main

logger = getLogger(__name__)

EVAL_NAME = "mt_bench"


def run_mt_bench(
    cfg: DictConfig,
    output_dir: Path,
    model_name: str,
) -> TaskResult:
    """Run the full MT-Bench pipeline and return a TaskResult."""
    from .gen_answer import generate_answers
    from .judge import judge_answers
    from .radar_chart import generate_radar_chart
    from .show_result import show_results

    data_dir = resolve_mt_bench_data_dir(cfg.get("data_dir") or None)
    logger.info("[MT-Bench] data_dir=%s", data_dir)

    artifacts: dict[str, str] = {}
    metadata: dict[str, object] = {"data_dir": str(data_dir)}

    # 1) Generate answers ----------------------------------------------
    answer_file = generate_answers(
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
        max_new_tokens=int(cfg.get("max_new_tokens", 1024)),
        request_timeout_sec=int(cfg.get("request_timeout_sec", 600)),
    )
    artifacts["answers"] = str(answer_file)

    # 2) Judge ---------------------------------------------------------
    judge_model = str(cfg.judge_model)
    try:
        judgment_file = judge_answers(
            data_dir=data_dir,
            output_dir=output_dir,
            model_name=model_name,
            judge_model=judge_model,
            judge_api_base=str(cfg.get("judge_api_base") or ""),
        )
        artifacts["judgments"] = str(judgment_file)
    except RuntimeError as e:
        if "API key required" not in str(e):
            raise
        logger.warning("[MT-Bench] Judge skipped: %s", e)
        return TaskResult.create(
            eval_name=EVAL_NAME,
            model=model_name,
            status="skipped",
            error="judge skipped: no API key",
            artifacts=artifacts,
            metadata=metadata,
        )

    # 3) Aggregate per-category scores --------------------------------
    summary = show_results(
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
        judge_model=judge_model,
    )
    if not summary:
        return TaskResult.create(
            eval_name=EVAL_NAME,
            model=model_name,
            status="failed",
            error="no scores produced",
            artifacts=artifacts,
            metadata=metadata,
        )

    # 4) Radar chart ---------------------------------------------------
    if bool(cfg.get("plot", True)):
        chart_path_str = str(cfg.get("chart_path") or "")
        chart_path = (
            Path(chart_path_str)
            if chart_path_str
            else output_dir / "charts" / f"mt_bench_radar_{model_name}.png"
        )
        try:
            chart = generate_radar_chart(
                results_dir=output_dir,
                output_path=chart_path,
                models=[model_name],
                title=f"MT-Bench: {model_name}",
            )
            if chart is not None:
                artifacts["chart"] = str(chart)
        except Exception as e:  # noqa: BLE001
            logger.warning("[MT-Bench] Radar chart generation failed: %s", e)

    scores = {
        "overall": summary.get("overall", 0.0),
        "categories": summary.get("categories", {}),
    }
    metadata["judge_model"] = judge_model
    metadata["n_scores"] = summary.get("n_scores", 0)

    return TaskResult.create(
        eval_name=EVAL_NAME,
        model=model_name,
        status="success",
        scores=scores,
        artifacts=artifacts,
        metadata=metadata,
    )


def main(argv: list[str] | None = None) -> int:
    return child_main(EVAL_NAME, run_mt_bench, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
