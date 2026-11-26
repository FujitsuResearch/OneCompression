"""Version-checked vLLM v1 adapter (optional dependency)."""

from .adapter import VLLMReasoningAdapter
from .probe_adapter import VLLMProbeReasoningAdapter

__all__ = ["VLLMProbeReasoningAdapter", "VLLMReasoningAdapter"]
