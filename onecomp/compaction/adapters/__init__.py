"""Framework adapters for the ARC compaction engine.

Concrete adapters (mini-swe-agent, LangChain, AutoGen, OpenAI Agents,
custom MAS orchestrators, ...) live here, one module per framework, so
that framework dependencies never leak into the core layers.

An adapter wires exactly three touchpoints of
:class:`onecomp.compaction.engine.ArcEngine` into the host framework:

1.  Append ``engine.system_prompt_snippet()`` to the system prompt.
2.  In the agent step loop, call ``engine.begin_step()`` per LLM
    response and pass each emitted action through
    ``engine.handle_action(action, messages)`` before executing it.
    A :class:`~onecomp.compaction.engine.RecallOutcome` means the turn
    was served from the store (skip execution); a
    :class:`~onecomp.compaction.engine.RecallError` should be fed back
    to the LLM as this turn's observation.  After a real execution,
    register the result with ``engine.record_observation`` and append
    ``engine.make_observation_message(record, rendered=...)``.
3.  On a context-window overflow — detected with
    :func:`matches_context_overflow` or the framework's own typed
    exception — replace the buffer with ``engine.compact(messages)``
    and retry.  For proactive trimming, use
    ``engine.enforce_char_budget(messages, max_chars)`` before each
    request.

If the target LLM API rejects message fields other than
``role``/``content``, send ``sanitize_messages(messages)`` and keep the
tagged originals as the source of truth.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

# Public stateful adapters are provided through PEP 562 ``__getattr__`` below.
# pylint: disable=undefined-all-variable

from __future__ import annotations

import re

from .messages import (
    content_to_text,
    message_role,
    message_text,
    protected_prefix_count,
    recent_turn_start,
    total_text_chars,
)

#: Substrings/regexes seen in context-overflow errors across providers
#: (OpenAI, Anthropic, vLLM, TGI, litellm re-wraps).  Kept permissive on
#: purpose; framework adapters should prefer their host's typed
#: exception when one exists.
_CONTEXT_OVERFLOW_RE = re.compile(
    (
        r"context.length|context.window|input.tokens|prompt.too.long|"
        r"maximum.context|max_tokens.*exceed"
    ),
    re.IGNORECASE,
)


def matches_context_overflow(error_message: str) -> bool:
    """Best-effort, framework-free context-overflow detection.

    Providers do not agree on a typed error for "prompt too long"; this
    helper matches the common phrasings in the error text.  Use the host
    framework's typed exception when available and fall back to this.
    """
    if not error_message:
        return False
    return bool(_CONTEXT_OVERFLOW_RE.search(error_message))


__all__ = [
    "content_to_text",
    "GuardedProviderAdapter",
    "matches_context_overflow",
    "message_role",
    "message_text",
    "protected_prefix_count",
    "recent_turn_start",
    "total_text_chars",
]


def __getattr__(name: str):
    """Load stateful adapters lazily to keep message helpers cycle-free."""
    if name == "GuardedProviderAdapter":
        from .provider import GuardedProviderAdapter

        return GuardedProviderAdapter
    raise AttributeError(name)
