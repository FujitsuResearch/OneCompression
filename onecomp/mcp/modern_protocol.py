"""Stateless MCP 2026-07-28 server over the stdio transport.

The legacy initialized server remains in :mod:`onecomp.mcp.protocol`. This
module intentionally implements the modern protocol as a separate profile so
version semantics are never guessed from process-global session state.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import sys
from logging import getLogger
from typing import Any, Dict, List, Optional

from ..__version__ import __version__
from .protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    SERVER_NAME,
    ToolDefinition,
)

MODERN_PROTOCOL_VERSION = "2026-07-28"
MODERN_SUPPORTED_PROTOCOL_VERSIONS = (MODERN_PROTOCOL_VERSION,)
UNSUPPORTED_PROTOCOL_VERSION = -32022
_PROTOCOL_META = "io.modelcontextprotocol/protocolVersion"
_CLIENT_CAPABILITIES_META = "io.modelcontextprotocol/clientCapabilities"
_CLIENT_INFO_META = "io.modelcontextprotocol/clientInfo"

logger = getLogger(__name__)


class ModernMCPServer:
    """Stateless-first MCP server with discovery and per-request negotiation."""

    def __init__(
        self,
        reader=None,
        writer=None,
        tools: Optional[List[ToolDefinition]] = None,
        *,
        tool_page_size: int = 50,
        cache_ttl_ms: int = 300_000,
    ) -> None:
        if (
            not isinstance(tool_page_size, int)
            or isinstance(tool_page_size, bool)
            or tool_page_size <= 0
        ):
            raise ValueError("tool_page_size must be a positive integer")
        if not isinstance(cache_ttl_ms, int) or isinstance(cache_ttl_ms, bool) or cache_ttl_ms < 0:
            raise ValueError("cache_ttl_ms must be a non-negative integer")
        self._reader = sys.stdin if reader is None else reader
        self._writer = sys.stdout if writer is None else writer
        tool_list = list(tools or [])
        if len({tool.name for tool in tool_list}) != len(tool_list):
            raise ValueError("MCP tool names must be unique")
        self._tools = {tool.name: tool for tool in tool_list}
        self._tool_page_size = tool_page_size
        self._cache_ttl_ms = cache_ttl_ms
        digest_source = "\x1f".join(self._tools).encode("utf-8")
        self._tool_catalog_digest = hashlib.sha256(digest_source).hexdigest()[:16]

    def serve_forever(self) -> None:
        """Process newline-delimited JSON-RPC until the input closes."""
        for line in self._reader:
            if not line.strip():
                continue
            response = self.handle_line(line)
            if response is not None:
                self._send(response)

    def handle_line(self, line: str) -> Optional[Dict[str, Any]]:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return _error_response(None, PARSE_ERROR, "Parse error")
        if not isinstance(message, dict):
            return _error_response(None, INVALID_REQUEST, "Invalid request")
        if message.get("jsonrpc") != "2.0":
            return _error_response(
                None,
                INVALID_REQUEST,
                "Invalid request: jsonrpc must be '2.0'",
            )
        return self.handle_message(message)

    def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle one self-contained request without retaining client state."""
        has_id = "id" in message
        if has_id and not _valid_request_id(message["id"]):
            return _error_response(None, INVALID_REQUEST, "Invalid request id")
        method = message.get("method")
        if not isinstance(method, str):
            response = _error_response(
                message.get("id"),
                INVALID_REQUEST,
                "Invalid request: missing method",
            )
            return response if has_id else None
        if not has_id:
            return None

        request_id = message["id"]
        try:
            params = _request_params(message.get("params"))
            _validate_request_metadata(params)
        except _UnsupportedVersion as exc:
            return _error_response(
                request_id,
                UNSUPPORTED_PROTOCOL_VERSION,
                "Unsupported protocol version",
                data={
                    "supported": list(MODERN_SUPPORTED_PROTOCOL_VERSIONS),
                    "requested": exc.requested,
                },
            )
        except _InvalidParams as exc:
            return _error_response(request_id, INVALID_PARAMS, str(exc))

        try:
            if method == "server/discover":
                result = self._discover()
            elif method == "ping":
                result = _complete_result()
            elif method == "tools/list":
                result = self._list_tools(params)
            elif method == "tools/call":
                result = self._call_tool(params)
            else:
                return _error_response(
                    request_id,
                    METHOD_NOT_FOUND,
                    f"Method not found: {method}",
                )
        except _InvalidParams as exc:
            return _error_response(request_id, INVALID_PARAMS, str(exc))
        except Exception:  # pylint: disable=broad-except
            logger.exception("Internal error while handling modern MCP method %s", method)
            return _error_response(request_id, INTERNAL_ERROR, "Internal server error")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _discover(self) -> Dict[str, Any]:
        return {
            **_complete_result(),
            "supportedVersions": list(MODERN_SUPPORTED_PROTOCOL_VERSIONS),
            "capabilities": {"tools": {"listChanged": False}},
            "instructions": (
                "OneComp exposes compression utilities and bounded, exact "
                "addressable recall. List tools before calling them."
            ),
            "ttlMs": self._cache_ttl_ms,
            "cacheScope": "public",
        }

    def _list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        unknown = set(params) - {"_meta", "cursor"}
        if unknown:
            raise _InvalidParams(f"unknown tools/list params: {sorted(unknown)}")
        offset = self._decode_cursor(params.get("cursor"))
        tools = list(self._tools.values())
        page = tools[offset : offset + self._tool_page_size]
        next_offset = offset + len(page)
        result = {
            **_complete_result(),
            "tools": [tool.to_dict() for tool in page],
            "ttlMs": self._cache_ttl_ms,
            "cacheScope": "public",
        }
        if next_offset < len(tools):
            result["nextCursor"] = self._encode_cursor(next_offset)
        return result

    def _call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        unknown = set(params) - {
            "_meta",
            "name",
            "arguments",
            "inputResponses",
            "requestState",
        }
        if unknown:
            raise _InvalidParams(f"unknown tools/call params: {sorted(unknown)}")
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise _InvalidParams("tools/call name must be a non-empty string")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise _InvalidParams("tool arguments must be an object")
        tool = self._tools.get(name)
        if tool is None:
            raise _InvalidParams(f"unknown tool: {name}")
        try:
            inspect.signature(tool.handler).bind(**arguments)
        except TypeError as exc:
            raise _InvalidParams(f"invalid arguments for tool {name}: {exc}") from exc
        try:
            value = tool.handler(**arguments)
            serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Modern MCP tool %s failed", name)
            return {
                **_complete_result(),
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            }
        return {
            **_complete_result(),
            "content": [{"type": "text", "text": serialized}],
            "structuredContent": value,
            "isError": False,
        }

    def _encode_cursor(self, offset: int) -> str:
        payload = json.dumps([self._tool_catalog_digest, offset], separators=(",", ":")).encode(
            "utf-8"
        )
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

    def _decode_cursor(self, cursor: Any) -> int:
        if cursor is None:
            return 0
        if not isinstance(cursor, str) or not cursor or len(cursor) > 512:
            raise _InvalidParams("cursor must be an opaque non-empty string")
        try:
            padding = "=" * (-len(cursor) % 4)
            payload = base64.b64decode(
                cursor + padding,
                altchars=b"-_",
                validate=True,
            )
            digest, offset = json.loads(payload)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise _InvalidParams("invalid or expired tools/list cursor") from exc
        if (
            digest != self._tool_catalog_digest
            or not isinstance(offset, int)
            or isinstance(offset, bool)
            or not 0 <= offset <= len(self._tools)
        ):
            raise _InvalidParams("invalid or expired tools/list cursor")
        return offset

    def _send(self, response: Dict[str, Any]) -> None:
        self._writer.write(json.dumps(response, ensure_ascii=False) + "\n")
        self._writer.flush()


class CompatibleMCPServer:
    """Route each stdio request to modern or legacy protocol semantics.

    A modern request is identified by ``server/discover`` or the required
    protocol-version metadata. Requests without either marker retain initialized
    legacy behavior for existing clients.
    """

    def __init__(self, reader=None, writer=None, tools=None) -> None:
        from .protocol import MCPServer

        self._reader = sys.stdin if reader is None else reader
        self._writer = sys.stdout if writer is None else writer
        self._legacy = MCPServer(tools=tools)
        self._modern = ModernMCPServer(tools=tools)

    def serve_forever(self) -> None:
        for line in self._reader:
            if not line.strip():
                continue
            response = self.handle_line(line)
            if response is not None:
                self._writer.write(json.dumps(response, ensure_ascii=False) + "\n")
                self._writer.flush()

    def handle_line(self, line: str) -> Optional[Dict[str, Any]]:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return self._modern.handle_line(line)
        if not isinstance(message, dict):
            return self._modern.handle_line(line)
        params = message.get("params")
        meta = params.get("_meta") if isinstance(params, dict) else None
        if message.get("method") == "server/discover" or (
            isinstance(meta, dict) and _PROTOCOL_META in meta
        ):
            return self._modern.handle_line(line)
        return self._legacy.handle_line(line)


def _request_params(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise _InvalidParams("request params must be an object")
    return value


def _validate_request_metadata(params: Dict[str, Any]) -> None:
    meta = params.get("_meta")
    if not isinstance(meta, dict):
        raise _InvalidParams("request params._meta must be an object")
    requested = meta.get(_PROTOCOL_META)
    if not isinstance(requested, str) or not requested:
        raise _InvalidParams(f"_meta.{_PROTOCOL_META} must be a non-empty string")
    if requested not in MODERN_SUPPORTED_PROTOCOL_VERSIONS:
        raise _UnsupportedVersion(requested)
    capabilities = meta.get(_CLIENT_CAPABILITIES_META)
    if not isinstance(capabilities, dict):
        raise _InvalidParams(f"_meta.{_CLIENT_CAPABILITIES_META} must be an object")
    client_info = meta.get(_CLIENT_INFO_META)
    if client_info is not None:
        if not isinstance(client_info, dict):
            raise _InvalidParams(f"_meta.{_CLIENT_INFO_META} must be an object")
        if not isinstance(client_info.get("name"), str) or not isinstance(
            client_info.get("version"), str
        ):
            raise _InvalidParams("clientInfo name and version must be strings")


def _complete_result() -> Dict[str, Any]:
    return {
        "resultType": "complete",
        "_meta": {
            "io.modelcontextprotocol/serverInfo": {
                "name": SERVER_NAME,
                "version": __version__,
                "description": "LLM compression and addressable recall tools",
            }
        },
    }


def _valid_request_id(value: Any) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


class _InvalidParams(ValueError):
    pass


class _UnsupportedVersion(ValueError):
    def __init__(self, requested: str) -> None:
        self.requested = requested
        super().__init__(requested)


def _error_response(
    request_id: Any,
    code: int,
    message: str,
    *,
    data: Any = None,
) -> Dict[str, Any]:
    error: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


__all__ = [
    "CompatibleMCPServer",
    "MODERN_PROTOCOL_VERSION",
    "MODERN_SUPPORTED_PROTOCOL_VERSIONS",
    "ModernMCPServer",
    "UNSUPPORTED_PROTOCOL_VERSION",
]
