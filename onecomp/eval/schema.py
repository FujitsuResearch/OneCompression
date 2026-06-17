"""Structured Hydra config and result schema for the evaluation harness.

The full configuration tree lives in onecomp/eval/conf/eval_config.yaml.
Every node below is a @dataclass so that OmegaConf.structured can
type-check the user's YAML overrides at load time.

Result schema (TaskResult) is the contract every evaluator subprocess
writes to <output_dir>/<eval_name>/result.json. The aggregator only
reads files matching this schema; nothing else couples the evaluators to
the parent process.

Copyright 2025-2026 Fujitsu Ltd.

"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, Optional

logger = getLogger(__name__)

JST = timezone(timedelta(hours=9))


# ---------------------------------------------------------------------------
# Inference (vLLM HTTP server)
# ---------------------------------------------------------------------------


@dataclass
class InferenceConfig:
    """vLLM OpenAI-compatible HTTP server launch parameters."""

    mode: str = "vllm_server"

    dtype: str = "auto"
    trust_remote_code: bool = True
    request_timeout_sec: int = 600

    host: str = "127.0.0.1"
    port: int = 0
    api_key: str = "EMPTY"

    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    quantization: Optional[str] = None
    enforce_eager: bool = False
    startup_timeout_sec: int = 600
    extra_args: list[str] = field(default_factory=list)


@dataclass
class VllmServerConfig(InferenceConfig):
    """Alias kept for the public API."""

    mode: str = "vllm_server"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Model under evaluation."""

    path: str = "???"
    name: Optional[str] = None


# ---------------------------------------------------------------------------
# Per-evaluator configs
# ---------------------------------------------------------------------------


@dataclass
class MtBenchConfig:
    """MT-Bench evaluator settings (kept simple; categories are dataset-driven)."""

    enabled: bool = True

    data_dir: str = ""
    judge_model: str = "gpt-4o-2024-08-06"
    judge_api_base: str = ""

    max_new_tokens: int = 1024
    request_timeout_sec: int = 600

    plot: bool = True
    chart_path: str = ""

    subprocess_timeout_sec: int = 7200


@dataclass
class ThroughputConfig:
    """vLLM serving throughput via Chat Completions (streaming).

    Uses a fixed-length synthetic user prompt and measures TTFT, ITL (TPOT),
    TPS per user, TPS decode, and aggregate TPS/RPS over the measured window.
    Independent of MT-Bench max_new_tokens.
    """

    enabled: bool = False

    prompt_tokens: int = 512
    max_tokens: int = 512
    num_warmup: int = 2
    num_trials: int = 5
    temperature: float = 0.0

    prompt_seed_text: str = (
        "This is a fixed prompt for throughput benchmarking. "
        "It compares decode performance of quantized models under the same conditions."
    )

    save_responses: bool = True
    save_warmup_responses: bool = False
    min_completion_tokens: int = 32

    request_timeout_sec: int = 600
    subprocess_timeout_sec: int = 1800


@dataclass
class EvalsConfig:
    """Container for all evaluator-specific configs.

    Add new evaluators by appending a field here and an entry under
    evals in conf/eval_config.yaml.
    """

    mt_bench: MtBenchConfig = field(default_factory=MtBenchConfig)
    throughput: ThroughputConfig = field(default_factory=ThroughputConfig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


@dataclass
class SummaryConfig:
    """Aggregator settings.

    include controls which evaluators are rolled into the final summary
    file; None means "include every evaluator that ran successfully".
    """

    include: Optional[list[str]] = None
    formats: list[str] = field(default_factory=lambda: ["json", "csv"])


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    """Root configuration consumed by run_evaluate.main."""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evals: EvalsConfig = field(default_factory=EvalsConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)

    output_dir: str = "./output/eval"
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Result schema (per-evaluator subprocess output)
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """Per-evaluator result written by each subprocess.

    Aggregator picks up files at <output_dir>/<eval_name>/result.json
    that conform to this schema and rolls them into a summary.
    """

    eval_name: str
    status: Literal["success", "failed", "skipped"] = "success"
    model: str = ""
    timestamp: str = ""

    scores: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @classmethod
    def create(cls, eval_name: str, model: str, **kwargs: Any) -> "TaskResult":
        return cls(
            eval_name=eval_name,
            model=model,
            timestamp=datetime.now(JST).isoformat(timespec="seconds"),
            **kwargs,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2))
        logger.info("Saved task result to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TaskResult":
        data = json.loads(Path(path).read_text())
        return cls(**data)
