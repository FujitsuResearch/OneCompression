"""Conformance tests for the stateless MCP 2026-07-28 profile."""

import io
import json

from onecomp.__version__ import __version__
from onecomp.mcp.modern_protocol import (
    MODERN_PROTOCOL_VERSION,
    CompatibleMCPServer,
    ModernMCPServer,
)
from onecomp.mcp.protocol import ToolDefinition


def _meta(version=MODERN_PROTOCOL_VERSION):
    return {
        "io.modelcontextprotocol/protocolVersion": version,
        "io.modelcontextprotocol/clientInfo": {
            "name": "test-client",
            "version": "1.0",
        },
        "io.modelcontextprotocol/clientCapabilities": {},
    }


def _request(request_id, method, **params):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {"_meta": _meta(), **params},
    }


def _tools(count=3):
    return [
        ToolDefinition(
            name=f"tool_{index}",
            title=f"Tool {index}",
            description="test tool",
            input_schema={"type": "object", "properties": {}},
            output_schema={
                "type": "object",
                "properties": {"index": {"type": "integer"}},
                "required": ["index"],
            },
            annotations={
                "readOnlyHint": True,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            handler=lambda index=index: {"index": index},
        )
        for index in range(count)
    ]


def test_discovery_is_stateless_complete_and_self_describing():
    server = ModernMCPServer(tools=_tools())
    response = server.handle_message(_request("discover", "server/discover"))
    result = response["result"]
    assert result["resultType"] == "complete"
    assert result["supportedVersions"] == [MODERN_PROTOCOL_VERSION]
    assert result["capabilities"] == {"tools": {"listChanged": False}}
    assert result["cacheScope"] == "public"
    assert result["ttlMs"] >= 0
    assert result["_meta"]["io.modelcontextprotocol/serverInfo"] == {
        "name": "onecomp",
        "version": __version__,
        "description": "LLM compression and addressable recall tools",
    }


def test_every_request_requires_version_and_capabilities_metadata():
    server = ModernMCPServer(tools=[])
    valid = server.handle_message(_request(1, "ping"))
    missing = server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}})
    assert valid["result"]["resultType"] == "complete"
    assert missing["error"]["code"] == -32602

    unsupported = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "server/discover",
            "params": {"_meta": _meta("1900-01-01")},
        }
    )
    assert unsupported["error"] == {
        "code": -32022,
        "message": "Unsupported protocol version",
        "data": {
            "supported": [MODERN_PROTOCOL_VERSION],
            "requested": "1900-01-01",
        },
    }


def test_tool_catalog_uses_opaque_stable_pagination_and_output_schemas():
    server = ModernMCPServer(tools=_tools(), tool_page_size=2)
    first = server.handle_message(_request(1, "tools/list"))["result"]
    assert [tool["name"] for tool in first["tools"]] == ["tool_0", "tool_1"]
    assert first["tools"][0]["title"] == "Tool 0"
    assert first["tools"][0]["outputSchema"]["type"] == "object"
    assert first["tools"][0]["annotations"]["readOnlyHint"] is True
    assert isinstance(first["nextCursor"], str)

    second = server.handle_message(_request(2, "tools/list", cursor=first["nextCursor"]))["result"]
    assert [tool["name"] for tool in second["tools"]] == ["tool_2"]
    assert "nextCursor" not in second

    invalid = server.handle_message(_request(3, "tools/list", cursor="not-a-cursor"))
    assert invalid["error"]["code"] == -32602


def test_tool_call_returns_structured_and_legacy_text_content():
    server = ModernMCPServer(tools=_tools(1))
    response = server.handle_message(_request(1, "tools/call", name="tool_0"))
    result = response["result"]
    assert result["resultType"] == "complete"
    assert result["structuredContent"] == {"index": 0}
    assert json.loads(result["content"][0]["text"]) == {"index": 0}
    assert result["isError"] is False


def test_tool_failures_are_model_visible_not_protocol_internal_errors():
    tool = ToolDefinition(
        name="fail",
        description="fail",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: (_ for _ in ()).throw(ValueError("correct the input")),
    )
    result = ModernMCPServer(tools=[tool]).handle_message(_request(1, "tools/call", name="fail"))[
        "result"
    ]
    assert result["resultType"] == "complete"
    assert result["isError"] is True
    assert "correct the input" in result["content"][0]["text"]


def test_modern_profile_rejects_initialize_and_ignores_notifications():
    server = ModernMCPServer(tools=[])
    initialized = server.handle_message(_request(1, "initialize"))
    assert initialized["error"]["code"] == -32601
    assert (
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"_meta": _meta(), "requestId": 1},
            }
        )
        is None
    )


def test_auto_profile_routes_legacy_and_modern_requests_in_one_process():
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        _request(2, "server/discover"),
        {"jsonrpc": "2.0", "id": 3, "method": "ping"},
        _request(4, "ping"),
    ]
    reader = io.StringIO("".join(json.dumps(message) + "\n" for message in messages))
    writer = io.StringIO()
    CompatibleMCPServer(reader, writer, tools=[]).serve_forever()
    responses = [json.loads(line) for line in writer.getvalue().splitlines()]
    assert responses[0]["result"]["protocolVersion"] == "2025-11-25"
    assert responses[1]["result"]["supportedVersions"] == [MODERN_PROTOCOL_VERSION]
    assert responses[2]["result"] == {}
    assert responses[3]["result"]["resultType"] == "complete"
