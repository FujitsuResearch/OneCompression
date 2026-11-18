"""MCP servers and autonomous compression tools with lazy public imports."""

# Public objects are resolved through PEP 562 ``__getattr__`` without loading
# model dependencies on the compaction-only path.
# pylint: disable=undefined-all-variable

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_ATTRS = {
    "MCPServer": (".protocol", "MCPServer"),
    "ToolDefinition": (".protocol", "ToolDefinition"),
    "ModernMCPServer": (".modern_protocol", "ModernMCPServer"),
    "CompatibleMCPServer": (".modern_protocol", "CompatibleMCPServer"),
    "AutoCompressConfig": (".agent", "AutoCompressConfig"),
    "AutoCompressionAgent": (".agent", "AutoCompressionAgent"),
    "AutoCompressResult": (".agent", "AutoCompressResult"),
    "TrialResult": (".agent", "TrialResult"),
    "build_tools": (".tools", "build_tools"),
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
