"""MCP (Model Context Protocol) server over the stdio transport.

Implements the newline-delimited JSON-RPC 2.0 framing used by the MCP
stdio transport (one message per line) with only the Python standard
library.  Supported methods: ``initialize``, ``notifications/initialized``,
``ping``, ``tools/list``, and ``tools/call``.

Classes:
    ToolDefinition: Tool metadata (name, description, JSON Schema) and handler.
    MCPServer: Newline-delimited JSON-RPC 2.0 server with injectable streams.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import inspect
import json
import sys
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional

from ..__version__ import __version__

# The tools subset implemented by this server is wire-compatible across these
# MCP revisions.  Keep the newest version first so an unknown/newer client
# gets the most recent version we can honestly advertise, while older clients
# can still negotiate the exact revision they requested.
SUPPORTED_PROTOCOL_VERSIONS = (
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)
PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]
SERVER_NAME = "onecomp"

# JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

logger = getLogger(__name__)


@dataclass
class ToolDefinition:
    """A single MCP tool: metadata plus the Python handler.

    Args:
        name: Tool name exposed via ``tools/list``.
        description: Human-readable description of the tool.
        input_schema: JSON Schema describing the tool arguments.
        handler: Callable invoked with the tool arguments as keyword
            arguments.  Must return a JSON-serialisable object.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Any] = field(repr=False, default=None)
    output_schema: Optional[Dict[str, Any]] = None
    title: Optional[str] = None
    annotations: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the ``tools/list`` entry for this tool."""
        value = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.output_schema is not None:
            value["outputSchema"] = self.output_schema
        if self.title is not None:
            value["title"] = self.title
        if self.annotations is not None:
            value["annotations"] = self.annotations
        return value


class MCPServer:
    """MCP server speaking newline-delimited JSON-RPC 2.0 over text streams.

    The reader and writer streams are injectable so the server can be
    driven from tests with :class:`io.StringIO` instead of stdio.

    Examples:
        >>> import io
        >>> reader = io.StringIO('{"jsonrpc": "2.0", "id": 1, "method": "ping"}\\n')
        >>> writer = io.StringIO()
        >>> MCPServer(reader, writer, tools=[]).serve_forever()
        >>> print(writer.getvalue())
    """

    def __init__(self, reader=None, writer=None, tools: Optional[List[ToolDefinition]] = None):
        """
        Args:
            reader: Text stream to read requests from (default: ``sys.stdin``).
            writer: Text stream to write responses to (default: ``sys.stdout``).
            tools: Tools to expose via ``tools/list`` / ``tools/call``.
        """
        self._reader = reader if reader is not None else sys.stdin
        self._writer = writer if writer is not None else sys.stdout
        self._tools: Dict[str, ToolDefinition] = {t.name: t for t in (tools or [])}

    def serve_forever(self) -> None:
        """Process messages line by line until the input stream is closed."""
        for line in self._reader:
            line = line.strip()
            if not line:
                continue
            response = self.handle_line(line)
            if response is not None:
                self._send(response)

    def handle_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse one JSON-RPC message line and return the response dict.

        Returns ``None`` for notifications (messages without an ``id``).
        """
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return _error_response(None, PARSE_ERROR, "Parse error")
        if not isinstance(message, dict):
            return _error_response(None, INVALID_REQUEST, "Invalid request")
        if message.get("jsonrpc") != "2.0":
            return _error_response(None, INVALID_REQUEST, "Invalid request: jsonrpc must be '2.0'")
        return self.handle_message(message)

    def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Dispatch a parsed JSON-RPC request or notification.

        Args:
            message: Parsed JSON-RPC 2.0 message.

        Returns:
            The response dict, or ``None`` for notifications.  A message
            without an ``id`` member is a JSON-RPC notification and never
            receives a response, not even on error.
        """
        response = self._dispatch(message)
        if "id" not in message:
            return None
        return response

    def _dispatch(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = message.get("method")
        msg_id = message.get("id")

        if not isinstance(method, str):
            return _error_response(msg_id, INVALID_REQUEST, "Invalid request: missing method")

        if method.startswith("notifications/"):
            if msg_id is None:
                return None
            # A "notifications/" method carrying an id is a request and
            # must be answered, or the client waits forever.
            return _error_response(msg_id, METHOD_NOT_FOUND, f"Method not found: {method}")

        try:
            if method == "initialize":
                result = self._handle_initialize(message.get("params"))
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": [t.to_dict() for t in self._tools.values()]}
            elif method == "tools/call":
                result = self._handle_tools_call(message.get("params"))
            else:
                return _error_response(msg_id, METHOD_NOT_FOUND, f"Method not found: {method}")
        except _InvalidParams as exc:
            return _error_response(msg_id, INVALID_PARAMS, str(exc))
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Internal error while handling %s", method)
            return _error_response(msg_id, INTERNAL_ERROR, f"Internal error: {exc}")

        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _handle_initialize(self, params: Any) -> Dict[str, Any]:
        if params is not None and not isinstance(params, dict):
            raise _InvalidParams("initialize params must be an object")
        requested = params.get("protocolVersion") if params else None
        if requested is not None and not isinstance(requested, str):
            raise _InvalidParams("initialize protocolVersion must be a string")
        protocol_version = (
            requested if requested in SUPPORTED_PROTOCOL_VERSIONS else PROTOCOL_VERSION
        )
        return {
            "protocolVersion": protocol_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": __version__},
        }

    def _handle_tools_call(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            raise _InvalidParams("tools/call params must be an object")
        name = params.get("name")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise _InvalidParams("Tool arguments must be an object")
        tool = self._tools.get(name)
        if tool is None:
            raise _InvalidParams(f"Unknown tool: {name}")
        try:
            # Validate the arguments against the handler signature up
            # front, so that a TypeError raised *inside* the handler is
            # reported as a tool error rather than as invalid params.
            inspect.signature(tool.handler).bind(**arguments)
        except TypeError as exc:
            raise _InvalidParams(f"Invalid arguments for tool {name}: {exc}") from exc
        try:
            result = tool.handler(**arguments)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Tool %s failed", name)
            return {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            }
        return {
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
            "isError": False,
        }

    def _send(self, response: Dict[str, Any]) -> None:
        self._writer.write(json.dumps(response, ensure_ascii=False) + "\n")
        self._writer.flush()


class _InvalidParams(ValueError):
    """Raised internally to signal a JSON-RPC invalid-params error."""


def _error_response(msg_id, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": code, "message": message},
    }
