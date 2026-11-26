"""Agent context compaction — ARC (Addressable Recall for Compaction).

This package implements *context* compression for LLM agents,
complementing OneComp's weight quantization: when an agent's
conversation history exceeds the model's context window, older turns are
compacted into a summary.  With ARC, long observations are not dropped —
each is preserved verbatim in a lossless, ID-addressable store and
represented in context as a bounded-length *citation* that the LLM can
materialise on demand with a ``_recall §<obs_id>`` action.

The package is layered for reuse across agent frameworks (single-agent
loops, multi-agent systems, custom orchestrators):

======================  ================================================
Layer                   Contents
======================  ================================================
``store``               :class:`ObsStore` / :class:`ObsRecord` —
                        lossless observation store with optional WAL.
``protocol``            Recall grammar (``_recall`` / ``_recall_meta``),
                        citation/metadata formatters, prompt snippet.
``compactor``           Pure message-list operators:
                        :func:`compact_messages` (baseline),
                        :func:`compact_messages_max` (AutoComp),
                        :func:`compact_messages_arc` (citation-rich).
``engine``              :class:`ArcEngine` — framework-agnostic facade
                        binding the layers; three integration
                        touchpoints (prompt, action interception,
                        compaction).
``adapters``            Framework-specific integrations and shared
                        helpers (context-overflow detection, ...).
======================  ================================================

Nothing in this package imports torch, an LLM client, or an agent
framework; it is pure Python and safe to use in any environment.

Based on ARC (Addressable Recall for Compaction), ported from the
ARC_compaction_CMU_PJ reference implementation (Apache-2.0).

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

from .compactor import (
    DEFAULT_ACTION_REGEX,
    DEFAULT_OBSERVATION_REGEX,
    RESULT_MAX_CHARS,
    SUMMARY_HEADER,
    THOUGHT_MAX_CHARS,
    compact_messages,
    compact_messages_arc,
    compact_messages_max,
)
from .config import CompactionConfig, CompactionDurability, CompactionProfile
from .engine import (
    PRIVATE_MESSAGE_KEYS,
    ArcEngine,
    ArcEngineConfig,
    RecallError,
    RecallOutcome,
    sanitize_messages,
)
from .formal import (
    BudgetExceeded,
    FormalArcError,
    InvalidAction,
    RecallRequiresChunking,
)
from .protocol import (
    RECALL_META_RE,
    RECALL_RE,
    SYSTEM_PROMPT_RECALL_SNIPPET,
    format_citation,
    format_meta_response,
    format_recall_response,
    looks_like_recall_intent,
    normalise_cmd_signature,
    parse_any_recall,
    parse_recall,
    parse_recall_meta,
    suggest_closest_obs_id,
)
from .runner import CompactionRunner
from .store import ObsRecord, ObsStore

__all__ = [
    "ArcEngine",
    "ArcEngineConfig",
    "CompactionConfig",
    "CompactionDurability",
    "CompactionProfile",
    "CompactionRunner",
    "BudgetExceeded",
    "FormalArcError",
    "InvalidAction",
    "RecallRequiresChunking",
    "DEFAULT_ACTION_REGEX",
    "DEFAULT_OBSERVATION_REGEX",
    "ObsRecord",
    "ObsStore",
    "PRIVATE_MESSAGE_KEYS",
    "RECALL_META_RE",
    "RECALL_RE",
    "RESULT_MAX_CHARS",
    "RecallError",
    "RecallOutcome",
    "SUMMARY_HEADER",
    "SYSTEM_PROMPT_RECALL_SNIPPET",
    "THOUGHT_MAX_CHARS",
    "compact_messages",
    "compact_messages_arc",
    "compact_messages_max",
    "format_citation",
    "format_meta_response",
    "format_recall_response",
    "looks_like_recall_intent",
    "normalise_cmd_signature",
    "parse_any_recall",
    "parse_recall",
    "parse_recall_meta",
    "sanitize_messages",
    "suggest_closest_obs_id",
]
