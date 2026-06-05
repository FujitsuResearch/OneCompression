"""Redact and strip sensitive values from eval configs and Hydra artifacts.

Judge API keys must be supplied via environment variables or secrets files,
never via persisted YAML or CLI overrides that end up on disk.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import re
from logging import getLogger
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict

logger = getLogger(__name__)

REDACTED = "***REDACTED***"

# Leaf config keys that must never be written to disk or logged.
SENSITIVE_CONFIG_KEYS = frozenset({"openai_api_key"})

# Hydra CLI override prefixes (without leading +/~).
SENSITIVE_OVERRIDE_PREFIXES = ("evals.mt_bench.openai_api_key",)

_OVERRIDE_RE = re.compile(
    r"^(?P<prefix>[+~]?)" r"(?P<key>evals\.mt_bench\.openai_api_key)" r"=(?P<value>.*)$",
)


def is_sensitive_override(override: str) -> bool:
    """Return True when a Hydra override carries a sensitive value."""
    token = override.strip()
    if token.startswith("-"):
        token = token[1:].strip()
    return any(
        token.startswith(prefix)
        or token.startswith(f"+{prefix}")
        or token.startswith(f"~{prefix}")
        for prefix in SENSITIVE_OVERRIDE_PREFIXES
    )


def redact_override(override: str) -> str:
    """Return a redacted copy of a Hydra override token."""
    token = override.strip()
    dash = ""
    if token.startswith("-"):
        dash = "- "
        token = token[1:].strip()
    match = _OVERRIDE_RE.match(token)
    if match:
        return f"{dash}{match.group('prefix')}{match.group('key')}={REDACTED}"
    return override


def _redact_mapping(data: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in data.items():
        if key in SENSITIVE_CONFIG_KEYS and value:
            redacted[key] = REDACTED
        elif isinstance(value, dict):
            redacted[key] = _redact_mapping(value)
        else:
            redacted[key] = value
    return redacted


def _strip_mapping(data: dict[str, Any]) -> dict[str, Any]:
    stripped: dict[str, Any] = {}
    for key, value in data.items():
        if key in SENSITIVE_CONFIG_KEYS:
            continue
        if isinstance(value, dict):
            stripped[key] = _strip_mapping(value)
        else:
            stripped[key] = value
    return stripped


def redact_config_for_log(cfg: DictConfig | Any) -> DictConfig:
    """Return a copy of cfg safe to emit in logs."""
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        return OmegaConf.create({})
    return OmegaConf.create(_redact_mapping(container))


def strip_sensitive_fields(cfg: Any) -> Any:
    """Return a copy of cfg with sensitive keys removed (safe to persist)."""
    if isinstance(cfg, DictConfig):
        container = OmegaConf.to_container(cfg, resolve=True)
    elif hasattr(cfg, "__dataclass_fields__"):
        from dataclasses import asdict

        container = asdict(cfg)
    else:
        container = dict(cfg)

    if not isinstance(container, dict):
        return cfg

    stripped = _strip_mapping(container)
    return OmegaConf.create(stripped)


def drop_sensitive_fields(cfg: DictConfig) -> None:
    """Remove sensitive keys from cfg in place (if present)."""
    if not OmegaConf.select(cfg, "evals.mt_bench"):
        return
    mt_bench = cfg.evals.mt_bench
    with open_dict(mt_bench):
        for key in SENSITIVE_CONFIG_KEYS:
            if key in mt_bench:
                del mt_bench[key]


def warn_on_sensitive_overrides() -> None:
    """Warn when a sensitive value was passed via Hydra CLI overrides."""
    try:
        from hydra.core.hydra_config import HydraConfig
    except ImportError:
        return

    try:
        overrides = HydraConfig.get().overrides.task
    except Exception:
        return

    for override in overrides:
        if is_sensitive_override(override):
            logger.warning(
                "Ignoring sensitive Hydra override %s; set OPENAI_API_KEY instead.",
                redact_override(override),
            )


def sanitize_hydra_run_dir() -> None:
    """Redact sensitive values in Hydra's on-disk run artifacts."""
    try:
        from hydra.core.hydra_config import HydraConfig
    except ImportError:
        return

    try:
        run_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        return

    hydra_dir = run_dir / ".hydra"
    if not hydra_dir.is_dir():
        return

    config_path = hydra_dir / "config.yaml"
    if config_path.is_file():
        cfg = OmegaConf.load(config_path)
        drop_sensitive_fields(cfg)
        OmegaConf.save(redact_config_for_log(cfg), config_path)

    overrides_path = hydra_dir / "overrides.yaml"
    if overrides_path.is_file():
        lines = overrides_path.read_text(encoding="utf-8").splitlines()
        sanitized = [_sanitize_overrides_line(line) for line in lines]
        overrides_path.write_text("\n".join(sanitized) + "\n", encoding="utf-8")


def _sanitize_overrides_line(line: str) -> str:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return line
    if is_sensitive_override(stripped.lstrip("- ")):
        return redact_override(stripped)
    return line
