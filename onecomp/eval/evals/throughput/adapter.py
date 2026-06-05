"""Parent-side adapter for the throughput evaluator."""

from __future__ import annotations

from omegaconf import DictConfig

from ..base import EvalAdapter, default_extract


def _extra_env(cfg: DictConfig) -> dict[str, str]:
    """Expose model path so the child can tokenize the synthetic prompt."""
    path = str(getattr(cfg.model, "path", "") or "").strip()
    if not path or path == "???":
        raise ValueError(
            "throughput eval requires model.path; set e.g. "
            "onecomp-eval model.path=/path/to/model"
        )
    return {"ONECOMP_MODEL_PATH": path}


ADAPTER = EvalAdapter(
    name="throughput",
    module="onecomp.eval.evals.throughput.run",
    extract_config=default_extract("throughput"),
    extra_env=_extra_env,
)
