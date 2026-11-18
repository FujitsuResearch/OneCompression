"""Unit tests for the MCP stdio protocol layer.

Drives :class:`onecomp.mcp.protocol.MCPServer` through in-memory
streams (``io.StringIO``) with newline-delimited JSON-RPC 2.0 messages
and stub tools, so no model, network, or GPU is required.

Copyright 2025-2026 Fujitsu Ltd.
"""

import io
import json

from onecomp.__version__ import __version__
from onecomp.mcp.protocol import PROTOCOL_VERSION, MCPServer, ToolDefinition


def _stub_tools():
    def _estimate_memory(model_id, wbits=None, context_len=4096, batch_size=1):
        return {
            "model_id": model_id,
            "wbits": 16 if wbits is None else wbits,
            "context_len": context_len,
            "batch_size": batch_size,
            "total_gb": 1.25,
        }

    def _boom(**_kwargs):
        raise RuntimeError("boom")

    def _type_error(**_kwargs):
        raise TypeError("type error raised inside the tool")

    return [
        ToolDefinition(
            name="estimate_memory",
            description="stub",
            input_schema={"type": "object", "properties": {"model_id": {"type": "string"}}},
            handler=_estimate_memory,
        ),
        ToolDefinition(
            name="failing_tool",
            description="stub that raises",
            input_schema={"type": "object", "properties": {}},
            handler=_boom,
        ),
        ToolDefinition(
            name="type_error_tool",
            description="stub that raises TypeError internally",
            input_schema={"type": "object", "properties": {}},
            handler=_type_error,
        ),
    ]


def _roundtrip(messages):
    """Feed JSON-RPC messages to the server and return parsed responses."""
    lines = "".join(json.dumps(m) + "\n" for m in messages)
    reader = io.StringIO(lines)
    writer = io.StringIO()
    MCPServer(reader, writer, tools=_stub_tools()).serve_forever()
    return [json.loads(line) for line in writer.getvalue().splitlines()]


def test_initialize_tools_list_tools_call_roundtrip():
    """initialize → tools/list → tools/call estimate_memory over StringIO."""
    responses = _roundtrip(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "estimate_memory",
                    "arguments": {"model_id": "some/model", "wbits": 4},
                },
            },
        ]
    )
    # The notification produces no response.
    assert len(responses) == 3

    init = responses[0]
    assert init["id"] == 1
    assert init["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert init["result"]["serverInfo"] == {"name": "onecomp", "version": __version__}
    assert "tools" in init["result"]["capabilities"]

    listing = responses[1]
    names = [t["name"] for t in listing["result"]["tools"]]
    assert "estimate_memory" in names
    for tool in listing["result"]["tools"]:
        assert tool["inputSchema"]["type"] == "object"

    call = responses[2]
    assert call["id"] == 3
    assert call["result"]["isError"] is False
    payload = json.loads(call["result"]["content"][0]["text"])
    assert payload["model_id"] == "some/model"
    assert payload["wbits"] == 4
    assert payload["total_gb"] == 1.25


def test_ping():
    (response,) = _roundtrip([{"jsonrpc": "2.0", "id": 7, "method": "ping"}])
    assert response == {"jsonrpc": "2.0", "id": 7, "result": {}}


def test_initialize_negotiates_supported_protocol_versions():
    responses = _roundtrip(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-03-26"},
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "initialize",
                "params": {"protocolVersion": "2099-01-01"},
            },
        ]
    )
    assert responses[0]["result"]["protocolVersion"] == "2025-03-26"
    assert responses[1]["result"]["protocolVersion"] == PROTOCOL_VERSION


def test_initialize_rejects_malformed_params():
    (response,) = _roundtrip([{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": []}])
    assert response["error"]["code"] == -32602


def test_zero_id_is_answered():
    """id=0 is a valid id and must not be confused with a notification."""
    (response,) = _roundtrip([{"jsonrpc": "2.0", "id": 0, "method": "ping"}])
    assert response == {"jsonrpc": "2.0", "id": 0, "result": {}}


def test_string_id_is_echoed_back():
    (response,) = _roundtrip([{"jsonrpc": "2.0", "id": "req-1", "method": "ping"}])
    assert response["id"] == "req-1"


def test_unknown_method_returns_method_not_found():
    (response,) = _roundtrip([{"jsonrpc": "2.0", "id": 5, "method": "no/such/method"}])
    assert response["error"]["code"] == -32601
    assert response["id"] == 5


def test_notifications_are_never_answered_even_on_error():
    """A message without an id is a notification: no response, ever."""
    responses = _roundtrip(
        [
            {"jsonrpc": "2.0", "method": "no/such/method"},
            {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "ghost"}},
            {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "failing_tool"}},
        ]
    )
    assert responses == []


def test_parse_error_on_invalid_json():
    reader = io.StringIO("this is not json\n")
    writer = io.StringIO()
    MCPServer(reader, writer, tools=[]).serve_forever()
    (response,) = [json.loads(line) for line in writer.getvalue().splitlines()]
    assert response["error"]["code"] == -32700
    assert response["id"] is None


def test_server_continues_after_parse_error():
    reader = io.StringIO('garbage\n{"jsonrpc": "2.0", "id": 1, "method": "ping"}\n')
    writer = io.StringIO()
    MCPServer(reader, writer, tools=[]).serve_forever()
    responses = [json.loads(line) for line in writer.getvalue().splitlines()]
    assert len(responses) == 2
    assert responses[0]["error"]["code"] == -32700
    assert responses[1] == {"jsonrpc": "2.0", "id": 1, "result": {}}


def test_missing_method_returns_invalid_request():
    (response,) = _roundtrip([{"jsonrpc": "2.0", "id": 9}])
    assert response["error"]["code"] == -32600


def test_wrong_or_missing_jsonrpc_version_is_invalid_request():
    responses = _roundtrip(
        [
            {"id": 1, "method": "ping"},
            {"jsonrpc": "1.0", "id": 2, "method": "ping"},
        ]
    )
    assert [r["error"]["code"] for r in responses] == [-32600, -32600]
    assert all(r["id"] is None for r in responses)


def test_non_object_params_return_invalid_params():
    """Array or scalar params must map to -32602, not an internal error."""
    responses = _roundtrip(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": [1, 2]},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": "x"},
        ]
    )
    for response in responses:
        assert response["error"]["code"] == -32602


def test_unknown_tool_returns_invalid_params():
    (response,) = _roundtrip(
        [
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "nonexistent_tool", "arguments": {}},
            }
        ]
    )
    assert response["error"]["code"] == -32602


def test_bad_arguments_return_invalid_params():
    (response,) = _roundtrip(
        [
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "estimate_memory",
                    "arguments": {"unexpected_argument": 1},
                },
            }
        ]
    )
    assert response["error"]["code"] == -32602


def test_missing_required_argument_returns_invalid_params():
    (response,) = _roundtrip(
        [
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": "estimate_memory", "arguments": {}},
            }
        ]
    )
    assert response["error"]["code"] == -32602


def test_non_object_arguments_return_invalid_params():
    (response,) = _roundtrip(
        [
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": "estimate_memory", "arguments": [1, 2]},
            }
        ]
    )
    assert response["error"]["code"] == -32602


def test_tool_exception_is_reported_as_tool_error():
    """A tool that raises maps to isError=True, not a protocol error."""
    (response,) = _roundtrip(
        [
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {"name": "failing_tool", "arguments": {}},
            }
        ]
    )
    assert "error" not in response
    assert response["result"]["isError"] is True
    assert "boom" in response["result"]["content"][0]["text"]


def test_type_error_inside_tool_is_tool_error_not_invalid_params():
    """A TypeError raised by the tool body is a tool failure, not -32602."""
    (response,) = _roundtrip(
        [
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {"name": "type_error_tool", "arguments": {}},
            }
        ]
    )
    assert "error" not in response
    assert response["result"]["isError"] is True
    assert "type error raised inside the tool" in response["result"]["content"][0]["text"]


def test_utf8_arguments_roundtrip():
    """Non-ASCII and embedded newlines survive the single-line framing."""
    payload = {"model_id": "モデル/日本語✨", "wbits": 4}  # jp-check: ignore
    (response,) = _roundtrip(
        [
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {"name": "estimate_memory", "arguments": payload},
            }
        ]
    )
    result = json.loads(response["result"]["content"][0]["text"])
    assert result["model_id"] == "モデル/日本語✨"  # jp-check: ignore


class TestNotificationWithId:
    """A ``notifications/`` method carrying an id must be answered."""

    def test_notification_without_id_is_ignored(self):
        responses = _roundtrip(
            [
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            ]
        )
        assert len(responses) == 1
        assert responses[0]["id"] == 1

    def test_notification_with_id_gets_method_not_found(self):
        responses = _roundtrip(
            [{"jsonrpc": "2.0", "id": 7, "method": "notifications/initialized"}]
        )
        assert len(responses) == 1
        assert responses[0]["id"] == 7
        assert responses[0]["error"]["code"] == -32601
