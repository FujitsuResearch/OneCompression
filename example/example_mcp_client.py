"""

Example: Minimal MCP client for the OneComp MCP server

Starts ``onecomp-mcp`` (``python -m onecomp.mcp``) as a subprocess,
performs the MCP handshake over stdio (newline-delimited JSON-RPC 2.0),
and calls the ``estimate_memory`` and ``recommend_recipe`` tools.

Usage::

    python example/example_mcp_client.py [model_id_or_path]

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import json
import subprocess
import sys


class MCPClient:
    """Tiny stdio MCP client for demonstration purposes."""

    def __init__(self, command):
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self._next_id = 0

    def request(self, method, params=None):
        """Send one JSON-RPC request and wait for its response."""
        self._next_id += 1
        message = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            message["params"] = params
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())

    def notify(self, method):
        """Send one JSON-RPC notification (no response expected)."""
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.process.stdin.flush()

    def call_tool(self, name, arguments):
        """Call an MCP tool and decode its JSON text content."""
        response = self.request("tools/call", {"name": name, "arguments": arguments})
        return json.loads(response["result"]["content"][0]["text"])

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=30)


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"

    client = MCPClient([sys.executable, "-m", "onecomp.mcp"])
    try:
        init = client.request("initialize", {"protocolVersion": "2025-03-26"})
        print("server:", json.dumps(init["result"]["serverInfo"]))
        client.notify("notifications/initialized")

        listing = client.request("tools/list")
        print("tools:", [t["name"] for t in listing["result"]["tools"]])

        memory = client.call_tool(
            "estimate_memory", {"model_id": model_id, "wbits": 4, "context_len": 4096}
        )
        print("estimate_memory:", json.dumps(memory, indent=2))

        recipe = client.call_tool("recommend_recipe", {"model_id": model_id, "total_vram_gb": 8.0})
        print("recommend_recipe:", json.dumps(recipe, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
