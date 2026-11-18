# MCP Server

OneComp ships an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server so that
AI assistants such as Cursor and Claude Desktop can use OneComp directly: estimate how much GPU
memory a model needs, get a quantization recipe for a VRAM budget, run the quantization, and
evaluate the result — all through tool calls.

The protocol and formal-compaction server paths use the Python standard
library and speak newline-delimited JSON-RPC 2.0 over stdio. The full model
tool suite uses the corresponding OneComp model dependencies.

## Starting the server

After installing OneComp, either entry point works:

```bash
onecomp-mcp
# or
python -m onecomp.mcp
```

For a dependency-free context-compaction control plane, run
`onecomp-mcp --compaction-only --wal-root /private/onecomp-wal`. This mode does not import torch,
transformers, model code, or accelerator packages. `onecomp-mcp --help` is
also safe in that minimal environment.

The server reads requests from stdin and writes responses to stdout, one JSON message per line.
It is normally launched automatically by the MCP client (Cursor, Claude Desktop, etc.).

## Available tools

| Tool | Description | Runtime |
|------|-------------|---------|
| `estimate_memory` | Weight + KV-cache memory estimate (GB) from `config.json` only — no weights are loaded. Accepts a local path or Hugging Face model ID. | seconds |
| `recommend_recipe` | Recommends bit-width, group size, method, and KV-cache precision for a VRAM budget, with a rationale string. | seconds |
| `quantize` | Runs `Runner.auto_run` (AutoBit + GPTQ + QEP) and saves the model. | minutes–hours |
| `evaluate` | WikiText-2 perplexity of a saved quantized model. | minutes |
| `vllm_engine_args` | vLLM launch arguments for a saved model directory: plugin requirement (`mixed_gptq`/`dbf`), recommended `kv_cache_dtype`, `LLM(**kwargs)` kwargs, and a `vllm serve` command line. Reads only `config.json`. | seconds |

Example `tools/call` request and response for `estimate_memory`:

```json
{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "estimate_memory", "arguments": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "wbits": 4}}}
```

```json
{"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "{\"model_id\": \"Qwen/Qwen2.5-0.5B-Instruct\", \"num_params\": 494032768, \"wbits\": 4, \"weight_gb\": 0.29, \"kv_cache_gb\": 0.05, \"total_gb\": 0.34, ...}"}], "isError": false}}
```

## Context compaction (ARC) tools

For the short cross-platform setup and measured results, see the
[ARC Context Compaction guide](arc-compaction.md).

The server also exposes [`onecomp.compaction`](../algorithms/overview.md) — ARC
(Addressable Recall for Compaction) — as session-based tools, so MCP-capable agent
platforms (e.g. OpenHands) can manage an agent's context window without importing
OneComp. Each default server session is an authenticated `formal-v1`
`CompactionRunner`, so one server process can safely serve many tenant-bound agents.

| Tool | Description |
|------|-------------|
| `compaction_start_session` | Create a session; returns `session_id` and a system-prompt snippet that teaches the LLM the `_recall` protocol. |
| `compaction_close_session` | Close a session, flush its WAL, free its store. |
| `compaction_begin_step` | Optional step boundary (one LLM response); resets the per-step recall budget. |
| `compaction_record_observation` | Store an executed action's output verbatim; returns the stable `obs_id`. Also marks a step boundary. |
| `compaction_compact_messages` | Compact a message buffer: long observations tagged with `_obs_id` become bounded, recall-able citations. `max_chars > 0` shrinks until the buffer fits. |
| `compaction_handle_action` | Intercept a raw LLM action; serves `_recall §<id>` / `_recall_meta §<id>` from the store, passes anything else through. |
| `compaction_recall` | MCP-native recovery by `obs_id`: bounded `meta`, `full`, deterministic `chunk`, or exact character `range`. |
| `compaction_list_observations` | Bounded, paginated handle/metadata index with optional command/path search; does not return stored bodies. |
| `compaction_verify_value` | Anti-confabulation guard: check that a value the agent wants to submit appeared verbatim in some observation. |
| `compaction_session_stats` | Session telemetry: observations, compactions, recalls, evictions. |

The intended flow: the platform appends the prompt snippet to the system prompt,
calls `compaction_record_observation` after every execution, passes LLM actions
through `compaction_handle_action`, and calls `compaction_compact_messages`
when (or before) the history exceeds the model's context window.  All compaction
tools are pure Python — no GPU, network, or model loading. The CLI server
requires the random session token returned by `compaction_start_session`;
`--allow-unauthenticated-sessions` is an explicit compatibility escape hatch.
For persistent formal sessions, set `max_archive_bytes` on
`compaction_start_session` to an exact UTF-8 body quota. A value of `0` leaves
storage quota policy to the deployment. One persistent archive can belong to
only one live session in a server process. Close that session before resuming
the archive. Its tenant identity is stored as a SHA-256 binding, not plaintext;
a later session from another tenant is rejected even after process restart.

There are two integration levels. Any MCP host, including OpenHands, can discover
and call the session, record, recall, verification, and telemetry tools directly.
Automatic context replacement additionally requires a host adapter/condenser that
owns the conversation buffer and replaces it with the `messages` returned by
`compaction_compact_messages`. A stock MCP tool cannot silently rewrite its host's
private message history; without that adapter, ARC is available as an explicit
lossless store/recall service rather than an automatic OpenHands condenser.

Do not have the model send its full conversation to
`compaction_compact_messages`: that duplicates the very prompt cost ARC is meant to
remove. The host/condenser should call it before constructing the provider request.
In the compatibility profile, `evicted_obs_ids` returned by an MCP recall is an
instruction to the host to replace those earlier recalled bodies; server-side
bookkeeping cannot delete a message already owned by an external client.

### Integration support matrix

| Host | Automatic replacement | Lossless recall | Important limit |
|------|-----------------------|-----------------|-----------------|
| OpenHands SDK | `OpenHandsArcCondenser` emits native `Condensation` events | Paired `arc_recall` and `arc_list_observations` tools share the same store | Import the optional adapter before constructing the agent; use a SQLite archive for process-resume durability. |
| Claude Code | `PostToolUse.updatedToolOutput` via `onecomp-arc-hook` | `onecomp-arc-recall` command in each citation | Replacement must match each built-in tool's output schema; the supplied adapter supports text/content-block outputs and Bash's structured output. |
| Codex CLI/IDE | PostToolUse `replace` mode or `onecomp-arc run -- COMMAND` | `onecomp-arc recall` from each citation | Current PostToolUse block feedback replaces completed output; `context` remains archival-only. |
| OpenCode | Generated plugin mutates `output.output` in `tool.execute.after` | `onecomp-arc-recall` command in each citation | Generate the plugin with `onecomp-arc plugin`; validate it against the installed host version. |
| Custom provider host | `GuardedProviderAdapter` replaces the owned list and guards dispatch | Call `CompactionRunner.handle_action` | The host must supply its authoritative final-request counter, including tools and multimodal overhead. |
| MCP only | No | `compaction_recall` | MCP gives the model tools/context; it does not grant the server mutation access to the host's old transcript. |

If a session enables `wal_path`, the formal profile treats it as a SQLite
archive path. The path is restricted to
`ONECOMP_MCP_WAL_ROOT` (the server's working directory by default). Prefer the
equivalent `--wal-root` option when launching the server; relative WAL paths are
resolved beneath it. This prevents an
LLM-controlled tool argument from appending to unrelated server files.
Set `resume_wal=true` together with the same `wal_path` when starting a
replacement session to restore the persisted observations. Experimental JSONL
archives should be migrated with `onecomp-arc migrate`. Resume with the same
`tenant_id` that first bound the formal archive.

## Native OpenHands condenser

The optional adapter targets the current OpenHands SDK condenser API. It records
forgotten `ObservationEvent` bodies, returns a native `Condensation`, and registers a
recall tool backed by the same in-process engine:

```bash
pip install onecomp[openhands]
```

```python
from openhands.sdk import Agent, Tool
from onecomp.compaction.adapters.openhands import OpenHandsArcCondenser

condenser = OpenHandsArcCondenser(
    max_size=120,
    max_tokens=100_000,
    summary_max_chars=24_000,
    recall_max_chars=20_000,
    wal_path="/secure/onecomp-archive/openhands-agent.sqlite3",
)

agent = Agent(
    llm=llm,
    condenser=condenser,
    tools=[
        Tool(name="TerminalTool"),
        Tool(
            name="OpenHandsArcRecallTool",
            params={"runtime_id": condenser.runtime_id},
        ),
        Tool(
            name="OpenHandsArcListTool",
            params={"runtime_id": condenser.runtime_id},
        ),
    ],
)
```

`max_tokens` is evaluated with OpenHands' own model-aware token counter. When a
strict summary cap leaves a bare or omitted handle, `arc_list_observations` searches
the original command/path and supplies the id for `arc_recall`. The WAL is
required if the agent/condenser will be serialised and restored in another process;
without it, the lossless store is intentionally process-local.

## Registering the MCP tools with OpenHands

Current OpenHands releases use the FastMCP JSON configuration format. Add an
entry to `~/.openhands/mcp.json` (or pass the equivalent `mcp_config` to an
OpenHands SDK `Agent`):

```json
{
  "mcpServers": {
    "onecomp": {
      "command": "/path/to/onecomp/.venv/bin/python",
      "args": ["-m", "onecomp.mcp"],
      "env": {
        "ONECOMP_MCP_WAL_ROOT": "/path/to/onecomp-wal"
      }
    }
  }
}
```

OpenHands should then discover the OneComp tools during agent initialization.
Use one session per agent and close it at the end of the run; separate sessions
keep MAS agents' stores isolated even when they share one MCP server process.

MCP registration alone provides explicit record/recall tools. Use the native
condenser above when the requirement is automatic removal of forgotten events from
the OpenHands model view.

## Claude Code PostToolUse adapter

Claude Code can replace a completed tool result before the model sees it. Install
OneComp's entry points, put the environment's `bin` directory on `PATH`, and add a
project or user hook:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash|Read|Grep|Glob",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/onecomp/.venv/bin/onecomp-arc-hook --platform claude"
          }
        ]
      }
    ]
  }
}
```

Set `ONECOMP_ARC_HOOK_ROOT` to a private persistence directory before starting
Claude Code. Long results are stored there with directory/file modes 0700/0600 and
replaced by a citation. The citation's `onecomp-arc-recall` command prints the exact
stored JSON/text response when its middle is needed. Add additional matchers only
after validating that the replacement has the same output shape as that tool; unknown
structured shapes deliberately pass through unchanged.

## Codex replacement hook and command boundary

Current Codex replaces a completed tool result with PostToolUse block feedback.
OneComp's default `replace` mode uses this contract after the archive commit.
The standalone wrapper remains a version-independent alternative:

```bash
onecomp-arc --config .onecomp/compaction/config.json run -- pytest -q
```

A minimal hook entry is:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "^Bash$",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/onecomp/.venv/bin/onecomp-arc-hook --platform codex"
          }
        ]
      }
    ]
  }
}
```

The default `--codex-result-mode replace` returns `decision: block` with the
citation as `reason`; the host uses that reason as the model-visible tool
output. `context` deliberately keeps the original and adds archival context.
`block` is a compatibility alias for `replace`. The MCP server can be registered
for explicit store/recall tools, but MCP alone cannot rewrite prior private
transcript entries.

## Registering with Cursor

Add the server to `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "onecomp": {
      "command": "onecomp-mcp"
    }
  }
}
```

If OneComp is installed in a virtual environment, point at its interpreter instead:

```json
{
  "mcpServers": {
    "onecomp": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "onecomp.mcp"]
    }
  }
}
```

## Registering with Claude Desktop

Add the same entry to `claude_desktop_config.json`
(macOS: `~/Library/Application Support/Claude/`, Windows: `%APPDATA%\Claude\`):

```json
{
  "mcpServers": {
    "onecomp": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "onecomp.mcp"]
    }
  }
}
```

After restarting the client, ask the assistant things like *"How much VRAM does
Llama-2-7B need at 4-bit?"* or *"Recommend a quantization recipe for my 8 GB GPU"*
and it will call the OneComp tools.

## Minimal Python client

`example/example_mcp_client.py` shows the full stdio round trip
(initialize → tools/list → tools/call) in ~80 lines:

```bash
python example/example_mcp_client.py Qwen/Qwen2.5-0.5B-Instruct
```

## Serving a quantized model with vLLM

The read-only `vllm_engine_args` tool turns a saved model directory into
everything needed to serve it with vLLM:

```json
{"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "vllm_engine_args", "arguments": {"model_dir": "./Qwen2.5-0.5B-gptq-4bit", "kv_cache_precision": "fp8"}}}
```

The result reports:

- `quant_method` / `needs_onecomp_plugin` / `plugin` — whether the model uses
  an OneComp vLLM plugin (`mixed_gptq`, `dbf`). Plugin methods require
  `onecomp` to be installed in the vLLM environment so the entry points
  register the quantization config.
- `vllm_supported` — `false` for rotation-preprocessed models (vLLM kernels
  cannot apply the online Hadamard transform) and unknown quant methods.
- `kv_cache_dtype` — `"fp8"` (with `calculate_kv_scales`) when
  `kv_cache_precision="fp8"` is requested, else `"auto"`.
- `engine_kwargs` — pass directly to `vllm.LLM(**engine_kwargs)`.
- `serve_command` — an equivalent `vllm serve ...` command line.
- `quantization_bits_summary` — layer count, bit-widths, and group sizes from
  the per-module `quantization_bits` block, when present.

The tool never loads weights, so it responds in milliseconds even for large
models.

## Autonomous compression agent

`onecomp.mcp.agent` provides a goal-driven search on top of the tools: given a VRAM budget and a
maximum tolerated perplexity increase, it starts from the recommended recipe, quantizes,
evaluates, and hill-climbs over bit-widths until it finds the smallest-memory recipe that
satisfies both constraints.

```python
from onecomp.mcp import AutoCompressConfig, AutoCompressionAgent

config = AutoCompressConfig(
    target_vram_gb=8.0,          # the quantized model must fit here
    max_ppl_increase_pct=5.0,    # vs. the FP16 baseline
    max_trials=6,
)
agent = AutoCompressionAgent("meta-llama/Llama-2-7b-hf", config)

# Preview the search without quantizing anything:
plan = agent.dry_run()
print(plan.message)

# Full search (each trial quantizes and evaluates the model):
result = agent.run(save_dir="./llama2-compressed")
print(result.message)
print(result.best.wbits, result.best.save_dir)
```

The quantize/evaluate execution functions are injectable through the constructor, which allows
dry-run simulation and unit testing without touching a GPU.

## Implementation Notes

This section documents the internals for contributors: the protocol layer, the tool schema
conventions, the memory-estimation formulas, and the agent's search algorithm.

### Protocol layer

The server (`onecomp/mcp/protocol.py`) implements the MCP stdio transport directly on top of
the standard library (`json` + text streams) instead of depending on the official `mcp` SDK.
This keeps the runtime dependency set of OneComp unchanged, makes the server importable in any
environment where OneComp itself runs, and keeps the whole protocol layer small enough to audit
(~200 lines).

Framing follows the MCP stdio transport: one JSON-RPC 2.0 message per line, UTF-8, no embedded
newlines (newlines inside strings are JSON-escaped, so any argument content is safe).
Responses are flushed after every message.  Logging uses the `logging` module, which writes to
stderr — stdout carries protocol messages only.

The default `--protocol auto` server routes initialized legacy requests and
stateless modern requests independently. Modern requests use protocol version
`2026-07-28`, `server/discover`, per-request protocol metadata, structured tool
content, opaque pagination cursors, and explicit `resultType`. Supported legacy
methods are:

| Method | Behaviour |
|--------|-----------|
| `initialize` | Negotiates supported MCP versions through `2025-11-25`, then returns server info and a `tools` capability. |
| `notifications/*` | Accepted and ignored (notifications never get a response). |
| `ping` | Returns `{}`. |
| `tools/list` | Returns the `ToolDefinition` entries (name, description, `inputSchema`). |
| `tools/call` | Validates params and dispatches to the tool handler. |
| anything else | `-32601 Method not found`. |

Error handling distinguishes protocol errors from tool failures:

- Malformed JSON → `-32700 Parse error` with `id: null`.
- Non-object message or missing `method` → `-32600 Invalid request`.
- Unknown tool, non-object `params`/`arguments`, or arguments that do not match the handler
  signature (checked up front with `inspect.signature().bind`) → `-32602 Invalid params`.
- An exception raised *inside* a tool handler is **not** a protocol error: it is returned as a
  regular result with `isError: true` and the exception text as content, per the MCP tool
  convention.  The server never crashes on a tool failure and keeps serving.
- A message without an `id` member is a JSON-RPC notification and never receives a response,
  not even an error response.

`MCPServer` takes injectable reader/writer streams, so the whole protocol layer is unit-tested
with `io.StringIO` and no subprocess.

### Tool schema conventions

Each tool (`onecomp/mcp/tools.py`) is a plain Python function paired with a JSON Schema in a
`ToolDefinition`.  The conventions are:

- Handler keyword arguments mirror the schema properties one-to-one, so `tools/call` arguments
  can be splatted directly into the handler (`handler(**arguments)`).
- Schemas declare `required` for arguments without Python defaults and `default` for the rest,
  so an MCP client can render the parameter list without reading the code.
- Handlers return JSON-serialisable dicts; the protocol layer serialises them into the
  `content: [{"type": "text", ...}]` result shape.
- Heavy imports (`torch`, `transformers`, `onecomp.runner`) happen inside the handlers, keeping
  server startup instant and CUDA untouched until a tool actually needs it.

### Memory-estimation formulas

`estimate_memory` reads only `config.json`.  The model is instantiated on the `meta` device
(no weight data is allocated, nothing is downloaded beyond the config), and the parameter count
is taken from the instantiated architecture.  Because shared tensors appear once in
`model.parameters()`, models with tied embeddings (`tie_word_embeddings: true`, e.g.
Qwen2.5-0.5B, Qwen3-0.6B) are counted correctly — the LM head is not double-counted.

Weight memory for a bit-width \(w\) with group size \(g\) uses the effective bits per parameter
from `onecomp.utils.vram_estimator` (AutoGPTQ storage convention: FP16 scale plus a zero point
packed at \(w\) bits per group):

\[
\text{bits}_{\text{eff}} = w + \frac{16 + w}{g}
\]

Not all parameters are quantized (embeddings, norms, and the LM head stay in FP16), which the
estimate models with a quantizable ratio \(r = 0.95\):

\[
\text{weight\_bytes} = N \cdot r \cdot \frac{\text{bits}_{\text{eff}}}{8}
                     + N \cdot (1 - r) \cdot 2
\]

The KV cache is GQA-aware — it scales with `num_key_value_heads`, not `num_attention_heads`:

\[
\text{kv\_bytes} = 2 \cdot L \cdot H_{kv} \cdot d_{\text{head}} \cdot T \cdot B \cdot s
\]

where \(L\) is `num_hidden_layers`, \(H_{kv}\) is `num_key_value_heads`, \(d_{\text{head}}\) is
`head_dim` (falling back to `hidden_size / num_attention_heads` when the config does not set
it), \(T\) is the context length, \(B\) the batch size, \(s\) the bytes per element (2 for
FP16, 1 for FP8), and the leading 2 accounts for both K and V.

`recommend_recipe` budgets 80% of the requested VRAM for weights + KV cache (the rest is
reserved for activations and runtime overhead), evaluates the candidate bit-widths
(8, 6, 5, 4, 3, 2) with per-bit-width group sizes and KV precisions, and picks the largest
fitting bit-width for `target="quality"` or the smallest fitting one (not below 3 bits) for
`target="memory"`.  When nothing fits, it returns the smallest configuration with
`fits: false` and a rationale.

### Agent search algorithm

`AutoCompressionAgent.run` is a hill climb over the ordered candidate bit-widths:

```text
baseline ← config.baseline_ppl or measure FP16 perplexity
idx ← index of the recommend_recipe() bit-width in the candidate list
while trials < max_trials and idx in range:
    bits ← candidates[idx]
    if bits already tried: stop                  # revisit ⇒ converged
    estimate memory, quantize, evaluate          # a failure stops the
                                                 # search, keeping the
                                                 # completed trials
    record trial (fits_memory, meets_ppl)
    if feasible:            idx += 1             # try fewer bits
    elif not meets_ppl:     idx -= 1             # back off to more bits
    else:                   idx += 1             # too big ⇒ fewer bits
best ← feasible trial with the smallest estimated memory
```

Termination is guaranteed by three independent guards: the `max_trials` budget, the
already-tried check (revisiting a bit-width means the search oscillated, i.e. converged), and
the index leaving the candidate range.  If no trial satisfies both constraints, the result has
`success=False` with the full trial history and an explanatory message.  A trial that raises
(quantization or evaluation failure) stops the search but preserves the trials completed
before it.

`dry_run()` reuses exactly the same loop with the quantize/evaluate functions swapped for a
simulator (`simulate_ppl_increase`, a monotone bit-width → degradation model), so the printed
trial plan matches what a real run would attempt.

### Adding a new tool

1. Implement a plain function in `onecomp/mcp/tools.py` returning a JSON-serialisable dict.
   Keep heavy imports inside the function body.
2. Append a `ToolDefinition` (name, description, JSON Schema, handler) to `build_tools()`.
   Property names must match the function's keyword arguments; mark arguments without
   defaults as `required`.
3. Add unit tests in `tests/onecomp/mcp/test_tools.py` (mock long-running work) and, if the
   tool has protocol-visible behaviour, in `tests/onecomp/mcp/test_protocol.py` using
   `io.StringIO` streams.
4. Document the tool in the table at the top of this page.

No protocol-layer change is needed: `tools/list` and `tools/call` pick up the new definition
automatically.
