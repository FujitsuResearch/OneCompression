"""Chat Completions streaming benchmark against a vLLM HTTP server.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any

logger = getLogger(__name__)


@dataclass
class TrialResult:
    """Metrics from one streamed Chat Completions request."""

    trial_index: int
    is_warmup: bool
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float | None
    itl_ms: float | None
    tps_per_user: float | None
    tps_decode: float | None
    decode_tok_s: float | None
    e2e_tok_s: float | None
    wall_time_sec: float
    request_start_mono: float = 0.0
    request_end_mono: float = 0.0
    error: str = ""
    response_text: str = ""
    finish_reason: str = ""
    response_sha256: str = ""
    response_issues: list[str] = field(default_factory=list)
    response_ok: bool = True


def run_throughput_benchmark(
    *,
    output_dir: Path,
    model_name: str,
    prompt_tokens: int,
    max_tokens: int,
    num_warmup: int,
    num_trials: int,
    temperature: float,
    prompt_seed_text: str,
    request_timeout_sec: int,
    save_responses: bool = True,
    save_warmup_responses: bool = False,
    min_completion_tokens: int = 32,
) -> tuple[list[TrialResult], Path, Path]:
    """Run warmup + measurement trials; return trials, JSONL path, prompt path."""
    mode = os.environ.get("ONECOMP_INFERENCE_MODE", "").strip()
    if mode != "vllm_server":
        raise RuntimeError(
            f"throughput eval requires ONECOMP_INFERENCE_MODE=vllm_server, got {mode!r}"
        )
    return _benchmark_vllm_http(
        output_dir=output_dir,
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        num_warmup=num_warmup,
        num_trials=num_trials,
        temperature=temperature,
        prompt_seed_text=prompt_seed_text,
        request_timeout_sec=request_timeout_sec,
        save_responses=save_responses,
        save_warmup_responses=save_warmup_responses,
        min_completion_tokens=min_completion_tokens,
    )


def _benchmark_vllm_http(
    *,
    output_dir: Path,
    model_name: str,
    prompt_tokens: int,
    max_tokens: int,
    num_warmup: int,
    num_trials: int,
    temperature: float,
    prompt_seed_text: str,
    request_timeout_sec: int,
    save_responses: bool,
    save_warmup_responses: bool,
    min_completion_tokens: int,
) -> tuple[list[TrialResult], Path, Path]:
    from openai import OpenAI

    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not base_url:
        raise RuntimeError(
            "OPENAI_BASE_URL is not set; the orchestrator should inject it "
            "before launching the throughput evaluator."
        )
    model_path = os.environ.get("ONECOMP_MODEL_PATH", "").strip()
    if not model_path:
        raise RuntimeError(
            "ONECOMP_MODEL_PATH is not set; required to build a fixed-length prompt."
        )

    prompt = _build_fixed_prompt(
        model_path,
        target_tokens=prompt_tokens,
        seed_text=prompt_seed_text,
    )
    logger.info(
        "[throughput] prompt_tokens(target)=%d max_tokens=%d warmup=%d trials=%d",
        prompt_tokens,
        max_tokens,
        num_warmup,
        num_trials,
    )

    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout_sec)
    served_model = _resolve_served_model(client, fallback=model_name)
    messages = [{"role": "user", "content": prompt}]

    bench_dir = output_dir / "throughput"
    bench_dir.mkdir(parents=True, exist_ok=True)
    trials_path = bench_dir / f"{model_name}_trials.jsonl"
    prompt_path = bench_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    results: list[TrialResult] = []
    total_runs = num_warmup + num_trials
    with trials_path.open("w", encoding="utf-8") as fout:
        for i in range(total_runs):
            is_warmup = i < num_warmup
            label = "warmup" if is_warmup else "trial"
            logger.info("[throughput] %s %d/%d", label, i + 1, total_runs)
            trial = _run_streamed_trial(
                client,
                served_model=served_model,
                messages=messages,
                trial_index=i,
                is_warmup=is_warmup,
                max_tokens=max_tokens,
                temperature=temperature,
                target_prompt_tokens=prompt_tokens,
                min_completion_tokens=min_completion_tokens,
            )
            results.append(trial)
            record = _trial_to_json_record(
                trial,
                save_responses=save_responses,
                save_warmup_responses=save_warmup_responses,
            )
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            if trial.error:
                logger.warning("[throughput] %s failed: %s", label, trial.error)
            elif not trial.response_ok:
                logger.warning(
                    "[throughput] %s response issues: %s",
                    label,
                    trial.response_issues,
                )

    return results, trials_path, prompt_path


def _run_streamed_trial(
    client: Any,
    *,
    served_model: str,
    messages: list[dict[str, str]],
    trial_index: int,
    is_warmup: bool,
    max_tokens: int,
    temperature: float,
    target_prompt_tokens: int,
    min_completion_tokens: int,
) -> TrialResult:
    t_start = time.perf_counter()
    t_first_token: float | None = None
    t_last_token: float | None = None
    completion_tokens = 0
    prompt_tokens = 0
    error = ""
    finish_reason = ""
    text_parts: list[str] = []

    try:
        stream = client.chat.completions.create(
            model=served_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if not chunk.choices:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    prompt_tokens = int(usage.prompt_tokens or prompt_tokens)
                    completion_tokens = int(usage.completion_tokens or completion_tokens)
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            content = getattr(delta, "content", None) or ""
            if content:
                text_parts.append(content)
                now = time.perf_counter()
                if t_first_token is None:
                    t_first_token = now
                t_last_token = now

            reason = getattr(choice, "finish_reason", None)
            if reason:
                finish_reason = str(reason)

            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = int(usage.prompt_tokens or prompt_tokens)
                completion_tokens = int(usage.completion_tokens or completion_tokens)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"

    t_end = time.perf_counter()
    wall = t_end - t_start

    ttft_ms: float | None = None
    if t_first_token is not None:
        ttft_ms = (t_first_token - t_start) * 1000.0

    decode_tok_s: float | None = None
    if t_first_token is not None and t_last_token is not None and completion_tokens > 1:
        decode_sec = t_last_token - t_first_token
        if decode_sec > 0:
            decode_tok_s = (completion_tokens - 1) / decode_sec

    itl_ms, tps_per_user, tps_decode = _compute_token_metrics(
        wall_time_sec=wall,
        ttft_ms=ttft_ms,
        completion_tokens=completion_tokens,
    )
    e2e_tok_s = tps_per_user

    if prompt_tokens <= 0:
        prompt_tokens = target_prompt_tokens

    response_text = "".join(text_parts)
    response_sha256 = _sha256_text(response_text)
    issues = _detect_response_issues(
        response_text,
        completion_tokens=completion_tokens,
        min_completion_tokens=min_completion_tokens,
        had_error=bool(error),
    )
    response_ok = not issues and not error

    return TrialResult(
        trial_index=trial_index,
        is_warmup=is_warmup,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_ms=ttft_ms,
        itl_ms=itl_ms,
        tps_per_user=tps_per_user,
        tps_decode=tps_decode,
        decode_tok_s=decode_tok_s,
        e2e_tok_s=e2e_tok_s,
        wall_time_sec=wall,
        request_start_mono=t_start,
        request_end_mono=t_end,
        error=error,
        response_text=response_text,
        finish_reason=finish_reason,
        response_sha256=response_sha256,
        response_issues=issues,
        response_ok=response_ok,
    )


def _compute_token_metrics(
    *,
    wall_time_sec: float,
    ttft_ms: float | None,
    completion_tokens: int,
) -> tuple[float | None, float | None, float | None]:
    """Return (ITL ms, TPS per user, TPS decode) from wall time and TTFT."""
    tps_per_user: float | None = None
    if completion_tokens > 0 and wall_time_sec > 0:
        tps_per_user = completion_tokens / wall_time_sec

    itl_ms: float | None = None
    tps_decode: float | None = None
    if completion_tokens > 1 and ttft_ms is not None and wall_time_sec > ttft_ms / 1000.0:
        generation_sec = wall_time_sec - (ttft_ms / 1000.0)
        if generation_sec > 0:
            itl_ms = (generation_sec / (completion_tokens - 1)) * 1000.0
            tps_decode = (completion_tokens - 1) / generation_sec

    return itl_ms, tps_per_user, tps_decode


def _aggregate_system_throughput(
    trials: list[TrialResult],
) -> dict[str, float | None]:
    """Aggregate TPS/RPS over the measured request window."""
    if not trials:
        return {"tps_system": None, "rps": None, "benchmark_window_sec": None}

    t_first = min(t.request_start_mono for t in trials)
    t_last = max(t.request_end_mono for t in trials)
    window = t_last - t_first
    if window <= 0:
        return {"tps_system": None, "rps": None, "benchmark_window_sec": None}

    total_tokens = sum(t.completion_tokens for t in trials)
    return {
        "tps_system": total_tokens / window,
        "rps": len(trials) / window,
        "benchmark_window_sec": window,
    }


def aggregate_trial_metrics(trials: list[TrialResult]) -> dict[str, Any]:
    """Summarize non-warmup trials (median speed + response health)."""
    all_measured = [t for t in trials if not t.is_warmup]
    measured = [t for t in all_measured if not t.error and t.completion_tokens > 0]
    if not measured:
        return {
            "n_trials": len(all_measured),
            "n_success": 0,
            "response_health_ok": 0.0,
        }

    def _median(values: list[float]) -> float:
        return float(statistics.median(values))

    def _median_or_none(values: list[float | None]) -> float | None:
        present = [v for v in values if v is not None]
        return _median(present) if present else None

    speed_scores: dict[str, Any] = {
        "n_trials": len(all_measured),
        "n_success": len(measured),
        "ttft_ms_median": _median_or_none([t.ttft_ms for t in measured]),
        "itl_ms_median": _median_or_none([t.itl_ms for t in measured]),
        "tps_per_user_median": _median_or_none([t.tps_per_user for t in measured]),
        "tps_decode_median": _median_or_none([t.tps_decode for t in measured]),
        # Legacy aliases kept for existing dashboards/scripts.
        "decode_tok_s_median": _median_or_none([t.decode_tok_s for t in measured]),
        "e2e_tok_s_median": _median_or_none([t.tps_per_user for t in measured]),
        "completion_tokens_median": _median([float(t.completion_tokens) for t in measured]),
        "prompt_tokens_median": _median([float(t.prompt_tokens) for t in measured]),
    }
    speed_scores.update(_aggregate_system_throughput(measured))
    speed_scores.update(aggregate_response_health(measured))
    return speed_scores


def aggregate_response_health(trials: list[TrialResult]) -> dict[str, Any]:
    """Aggregate checks that generations are non-empty and not obviously broken."""
    hashes = [t.response_sha256 for t in trials if t.response_sha256]
    distinct_hashes = sorted(set(hashes))
    unhealthy = [t for t in trials if not t.response_ok]
    all_ok = len(unhealthy) == 0

    issue_counts: dict[str, int] = {}
    for t in trials:
        for issue in t.response_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    first = trials[0]
    preview = first.response_text[:240].replace("\n", " ")

    return {
        "response_health_ok": 1.0 if all_ok else 0.0,
        "n_unhealthy_trials": len(unhealthy),
        "n_distinct_response_hashes": len(distinct_hashes),
        "responses_all_identical": len(distinct_hashes) <= 1,
        "response_sha256": distinct_hashes[0] if len(distinct_hashes) == 1 else "",
        "response_char_len_median": float(
            statistics.median([len(t.response_text) for t in trials])
        ),
        "response_preview": preview,
        "response_issue_counts": issue_counts,
    }


def _trial_to_json_record(
    trial: TrialResult,
    *,
    save_responses: bool,
    save_warmup_responses: bool,
) -> dict[str, Any]:
    record = asdict(trial)
    if trial.is_warmup and not save_warmup_responses:
        record["response_text"] = ""
    elif not save_responses:
        record["response_text"] = ""
    return record


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _detect_response_issues(
    text: str,
    *,
    completion_tokens: int,
    min_completion_tokens: int,
    had_error: bool,
) -> list[str]:
    """Heuristic sanity checks for corrupted or degenerate outputs."""
    if had_error:
        return ["request_error"]
    issues: list[str] = []
    stripped = text.strip()
    if not stripped:
        issues.append("empty")
    if completion_tokens < min_completion_tokens:
        issues.append("too_short")
    if "\ufffd" in text:
        issues.append("unicode_replacement")
    if _has_excessive_char_repetition(stripped):
        issues.append("char_repetition")
    if _has_excessive_word_repetition(stripped):
        issues.append("word_repetition")
    return issues


def _has_excessive_char_repetition(text: str, min_len: int = 80) -> bool:
    if len(text) < min_len:
        return False
    runs = re.findall(r"(.)\1{19,}", text)
    return bool(runs)


def _has_excessive_word_repetition(text: str, min_words: int = 40) -> bool:
    words = text.split()
    if len(words) < min_words:
        return False
    # same word >= 8 times in a row
    run = 1
    prev = words[0]
    for w in words[1:]:
        if w == prev:
            run += 1
            if run >= 8:
                return True
        else:
            run = 1
            prev = w
    return False


def _build_fixed_prompt(
    model_path: str,
    *,
    target_tokens: int,
    seed_text: str,
) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not ids:
        pad = tokenizer.pad_token_id or tokenizer.eos_token_id
        if pad is None:
            raise RuntimeError("Cannot build prompt: tokenizer produced empty ids")
        ids = [pad]

    repeat = (target_tokens // len(ids)) + 2
    truncated = (ids * repeat)[:target_tokens]
    return tokenizer.decode(truncated, skip_special_tokens=True)


def _resolve_served_model(client: Any, fallback: str) -> str:
    try:
        ids = [m.id for m in client.models.list().data]
        if ids:
            return ids[0]
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[throughput] /v1/models lookup failed (%s); falling back to %r",
            e,
            fallback,
        )
    return fallback
