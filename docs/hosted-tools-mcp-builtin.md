---
title: Hosted tools via the MCP-builtin pattern
---

# Hosted tools via the MCP-builtin pattern

This page is the single source of truth for how SMG dispatches OpenAI
hosted-tool calls (`image_generation`, `web_search_preview`, `web_search`,
`code_interpreter`, `file_search`) through MCP servers, and what the
client-facing response items and streaming events look like for each tool.

For the orthogonal pattern — a user-defined MCP server invoked through a
`{"type": "mcp", ...}` tool entry — see
[MCP in Responses API](getting-started/mcp.md) and
[Model Context Protocol](concepts/extensibility/mcp.md). This page covers only
the **MCP-builtin** pattern, where a request that opts into a hosted tool
(`{"type": "image_generation"}`, `{"type": "web_search_preview"}`, etc.) is
served by an SMG-configured MCP server instead of being forwarded to the
upstream model.

---

## Overview

### What the pattern is

The MCP-builtin pattern is a registration knob on an MCP server entry in
`mcp.yaml` that binds a `BuiltinToolType` (`web_search_preview`,
`code_interpreter`, `file_search`, `image_generation`) to a specific tool on
that MCP server. Once a server is registered with `builtin_type` +
`builtin_tool_name`, every Responses API request that includes the matching
hosted-tool entry in its `tools` array is dispatched to that server, and the
MCP tool result is shaped into the OpenAI-spec response output item before it
reaches the client.

This lets a deployment substitute a real OpenAI-managed hosted tool with a
local or third-party backend without changing the client request shape.

### How a request flows

1. Client sends a Responses API request with one of the hosted-tool entries
   in `tools[]` — for example
   `{"type": "image_generation", "size": "1024x1024", ...}`.
2. The router classifies each tool. Hosted tools that match a configured
   `BuiltinToolType` are routed via the MCP orchestrator instead of being
   passed through to the upstream model. Routing is wired in
   `model_gateway/src/routers/common/mcp_utils.rs` through `extract_builtin_types`,
   `collect_builtin_routing`, and `ensure_request_mcp_client`.
3. The orchestrator looks up the registered server by `builtin_type`
   (`crates/mcp/src/core/orchestrator.rs::find_builtin_server`) and dispatches
   the tool call to that server's named tool (`builtin_tool_name`).
4. The MCP server returns a `CallToolResult`. The transformer in
   `crates/mcp/src/transform/transformer.rs::ResponseTransformer::transform`
   reshapes that result into the OpenAI-spec `ResponseOutputItem` matching the
   tool's `response_format` (`web_search_call`, `code_interpreter_call`,
   `file_search_call`, or `image_generation_call`).
5. Streaming envelopes (`response.output_item.added`,
   `response.<tool>.in_progress`, `response.<tool>.completed`,
   `response.output_item.done`) are emitted around the dispatch so wire
   parity matches OpenAI's hosted-tool surface.

### Supported tool / configuration matrix

The set of `BuiltinToolType` variants is the canonical list of hosted tools
that the gateway currently knows how to dispatch via MCP. From
`crates/mcp/src/core/config.rs::BuiltinToolType`:

| Variant (YAML)         | `response_format`         | Output item                  | Status                |
|------------------------|---------------------------|------------------------------|-----------------------|
| `web_search_preview`   | `web_search_call`         | `web_search_call`            | Supported             |
| `web_search`           | `web_search_call`         | `web_search_call`            | Supported (alias of `web_search_preview` in dispatch; same response format) |
| `code_interpreter`     | `code_interpreter_call`   | `code_interpreter_call`      | Supported             |
| `file_search`          | `file_search_call`        | `file_search_call`           | Partial (transformer present; vector-store integration is up to the MCP server backing it) |
| `image_generation`     | `image_generation_call`   | `image_generation_call`      | Supported             |

`web_search` and `web_search_preview` share dispatch routing today. Both map
to the `web_search_call` output item. Per
`crates/protocols/src/responses.rs::ResponseTool`, `web_search` is a distinct
spec variant that adds `filters.allowed_domains` and a typed
`search_context_size`, but on the MCP-builtin path the same handler serves
both — register one MCP server with `builtin_type: web_search_preview` and it
will pick up both request shapes.

---

## Per-tool sections

Each section below documents one supported hosted tool: server-config recipe,
the response-item shape the gateway emits, the streaming-event sequence, and
any tool-specific quirks.

The captured sequences and field samples come from real OpenAI Responses API
captures against `gpt-5-nano` (OpenAI Python SDK 2.8.1). They show the wire
shape the MCP-builtin path must reproduce so cloud-passthrough and local
dispatch behave identically.

### `image_generation`

**Status**: Supported.

**Server config recipe** (mirrors
`e2e_test/responses/conftest.py::_image_generation_mcp_config`, the canonical
fixture for this pattern):

```yaml title="mcp.yaml"
servers:
  - name: mock-image-gen
    protocol: streamable
    url: "http://127.0.0.1:8765/mcp"
    builtin_type: image_generation
    builtin_tool_name: image_generation
    tools:
      image_generation:
        response_format: image_generation_call
```

The MCP server's `image_generation` tool is expected to return either:

- A direct object containing `result` (or `image_base64` / `b64_json`) carrying
  the base64-encoded image, plus optional `revised_prompt`, `action`,
  `background`, `output_format`, `quality`, `size`.
- An MCP text-block array carrying a JSON payload with the same fields, either
  at the top level or nested under `openai_response`.

The transformer logic that extracts these fields lives in
`crates/mcp/src/transform/transformer.rs::ResponseTransformer::to_image_generation_call`.

**Expected response item shape** — verbatim from a real `gpt-5-nano` capture
(`/tmp/openai_ground_truth/results/image_generation.json`, base64 result
truncated):

```json
{
  "id": "ig_035e0a6781a0d8bb0169ebf9c973588190b4944335b9fd42d9",
  "type": "image_generation_call",
  "status": "completed",
  "result": "iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAIAAADwf7zUAABnL2NhQlgAAGcvanVtYgAAAB5qdW1k...(1388588 chars total)",
  "revised_prompt": "A 1x1 pixel image with a solid white color",
  "action": "generate",
  "background": "opaque",
  "output_format": "png",
  "quality": "medium",
  "size": "1024x1024"
}
```

The five metadata fields (`action`, `background`, `output_format`, `quality`,
`size`) are real on the wire from OpenAI even though earlier OpenAI Rust /
Python SDK versions omitted them. SMG carries them as `Option<String>` on
`ResponseOutputItem::ImageGenerationCall` (see
`crates/protocols/src/responses.rs`) so cloud passthrough preserves them and
locally-dispatched calls can surface them when the MCP server provides them.

**Streaming event sequence** (17 events total, from the same capture):

```
000 response.created
001 response.in_progress
002 response.output_item.added       (item.type=reasoning)
003 response.output_item.done        (item.type=reasoning)
004 response.output_item.added       (item.type=image_generation_call)
005 response.image_generation_call.in_progress       (output_index=1)
006 response.image_generation_call.generating        (output_index=1)
007 response.image_generation_call.partial_image     (output_index=1)
008 response.output_item.done        (item.type=image_generation_call)
009 response.output_item.added       (item.type=reasoning)
010 response.output_item.done        (item.type=reasoning)
011 response.output_item.added       (item.type=message)
012 response.content_part.added
013 response.output_text.done
014 response.content_part.done
015 response.output_item.done        (item.type=message)
016 response.completed
```

The `response.image_generation_call.partial_image` event (event 007) carries:

```json
{
  "type": "response.image_generation_call.partial_image",
  "item_id": "ig_00b3a7024140c0dc0169ebf9ea83848193bebae0c5d2ec849c",
  "output_index": 1,
  "sequence_number": 7,
  "partial_image_index": 0,
  "partial_image_b64": "iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAIAAADwf7zUAABcSmNhQlgAAFxK...(1388588 chars total)",
  "background": "opaque",
  "output_format": "png",
  "quality": "medium",
  "size": "1024x1024"
}
```

**Quirks**:

- **No `.completed` sub-event.** Real OpenAI does **not** emit
  `response.image_generation_call.completed`. Verified across three capture
  variants (default, full-config, partial_images=0) — none of them contain
  it. Completion is signaled solely by `response.output_item.done` for the
  `image_generation_call` item. The constant
  `ImageGenerationCallEvent::COMPLETED` exists in
  `crates/protocols/src/event_types.rs` for forward-compat, but production
  OpenAI does not currently emit it. MCP-builtin dispatch should match this
  behavior. Other hosted tools (`web_search_preview`, `web_search`,
  `code_interpreter`) **do** emit their respective `.completed` sub-event
  before `output_item.done`.
- **`partial_image` event present in all observed runs.** All three captured
  image-generation variants emitted exactly one `.partial_image` event before
  `output_item.done`. Tests should treat it as expected (not optional) for
  the `gpt-5-nano` reference but allow zero or more for forward-compat
  (the spec allows 0–3 partials when `partial_images` is configured on the
  tool input).
- **`output_item.done` carries the partial result.** In the captured stream,
  the `image_generation_call` item's `output_item.done` payload reports
  `status: "generating"` and a populated `result` field rather than
  `status: "completed"` — the final non-streaming response then carries
  `status: "completed"`. Treat `output_item.done` as the terminal streaming
  signal and trust the non-streaming response (or the final
  `response.completed` payload's `output[]`) for the canonical status.

### `web_search_preview`

**Status**: Supported.

**Server config recipe**:

```yaml title="mcp.yaml"
servers:
  - name: brave
    protocol: sse
    url: "https://mcp.brave.com/sse"
    builtin_type: web_search_preview
    builtin_tool_name: brave_web_search
    tools:
      brave_web_search:
        response_format: web_search_call
```

The MCP server's named tool returns either an object or an MCP text-block
array containing fields the transformer extracts: `queries` (list), `sources`
(list of `{type, url}` entries), and the original `query` (recovered from the
tool-call arguments). See
`crates/mcp/src/transform/transformer.rs::ResponseTransformer::to_web_search_call`.

**Expected response item shape** — from a `gpt-5-nano` capture
(`/tmp/openai_ground_truth/results/web_search_preview.json`):

```json
{
  "id": "ws_0a195f61fa34ede70169ebfa6ab43881969f0bc8f95e0b41e8",
  "type": "web_search_call",
  "status": "completed",
  "action": {
    "type": "search",
    "query": "weather: Paris, France",
    "queries": ["weather: Paris, France"],
    "sources": null
  }
}
```

The `action` discriminator can be `search` (shape above), `open_page`
(`{type, url}`), or `find` (`{type, url, pattern}`); see
`crates/protocols/src/responses.rs::WebSearchAction`.
`web_search_call.results` (an array of `{url, title?, snippet?, score?}` hits)
is populated only when the caller requests it via the top-level `include[]`
array — by default the transformer leaves it absent so the wire shape stays
`{id, action, status, type}`.

**Streaming event sequence** (20 events total):

```
000 response.created
001 response.in_progress
002 response.output_item.added       (item.type=reasoning)
003 response.output_item.done        (item.type=reasoning)
004 response.output_item.added       (item.type=web_search_call)
005 response.web_search_call.in_progress    (output_index=1)
006 response.web_search_call.searching      (output_index=1)
007 response.web_search_call.completed      (output_index=1)
008 response.output_item.done        (item.type=web_search_call)
009 response.output_item.added       (item.type=reasoning)
010 response.output_item.done        (item.type=reasoning)
011 response.output_item.added       (item.type=message)
012 response.content_part.added
013-015 response.output_text.delta * 3
016 response.output_text.done
017 response.content_part.done
018 response.output_item.done        (item.type=message)
019 response.completed
```

**Quirks**:

- The `web_search_call.completed` sub-event **is** emitted (event 007), unlike
  `image_generation`. It always precedes `response.output_item.done` for the
  same item.
- `output_item.added` for the `web_search_call` item carries `action: null`
  and `status: "in_progress"`. The terminal `output_item.done` carries the
  fully-populated `action` and `status: "completed"`.

### `web_search`

**Status**: Supported (dispatched through the same MCP-builtin path as
`web_search_preview`; the transformer emits a `web_search_call` output item
for both).

**Server config recipe**: identical to `web_search_preview` above; register
one MCP server with `builtin_type: web_search_preview` (the canonical key)
and both `{"type": "web_search_preview"}` and `{"type": "web_search"}` /
`{"type": "web_search_2025_08_26"}` request entries will be dispatched
through it. There is no separate `BuiltinToolType::WebSearch` — see
`crates/mcp/src/core/config.rs::BuiltinToolType`.

**Expected response item shape** — from
`/tmp/openai_ground_truth/results/web_search.json` (same `gpt-5-nano` model,
issued with `tools: [{"type": "web_search"}]`):

```json
{
  "id": "ws_09ae4f6fd622d5000169ebfa7f850081939a07b39931c211ce",
  "type": "web_search_call",
  "status": "completed",
  "action": {
    "type": "search",
    "query": "weather: Paris, France",
    "queries": ["weather: Paris, France"],
    "sources": null
  }
}
```

The output item type on the wire is still `web_search_call`. The
`web_search` request-side variant is purely an input-tool distinction
(filters, search-context size); the response item is unified.

**Streaming event sequence**: identical shape to `web_search_preview` —
`response.web_search_call.in_progress` → `.searching` → `.completed`,
bracketed by `response.output_item.added` / `response.output_item.done` for
the `web_search_call` item.

**Quirks**:

- Per
  `crates/protocols/src/responses.rs::ResponseTool::WebSearch`, the
  request-side type alias `web_search_2025_08_26` deserializes into the same
  `WebSearch` variant. All three accept the same MCP-builtin dispatch.
- The non-preview `web_search` tool input adds `filters.allowed_domains` and
  `search_context_size`; if your MCP server backend honors those, surface
  them in the tool-call arguments and propagate to the upstream search
  service. They do not appear on the response item itself.

### `code_interpreter`

**Status**: Supported.

**Server config recipe**:

```yaml title="mcp.yaml"
servers:
  - name: ci
    protocol: streamable
    url: "http://localhost:9100/mcp"
    builtin_type: code_interpreter
    builtin_tool_name: execute
    tools:
      execute:
        response_format: code_interpreter_call
```

The MCP server's named tool is expected to return an object containing
`container_id`, `code`, and optional `outputs[]`. The transformer extracts
those fields verbatim — see
`crates/mcp/src/transform/transformer.rs::ResponseTransformer::to_code_interpreter_call`.

**Expected response item shape** — from
`/tmp/openai_ground_truth/results/code_interpreter.json`:

```json
{
  "id": "ci_043c40d9e51b9b760169ebfa94512c8194befafe008b6e86e3",
  "type": "code_interpreter_call",
  "status": "completed",
  "container_id": "cntr_69ebfa8ec31081909a5ae4f3d082688e0cbe6a5a82398d87",
  "code": "sum(range(1,11))",
  "outputs": null
}
```

`outputs` is `null` (or omitted) when the tool produces no captured stdout /
artifacts.

**Streaming event sequence** (37 events total — the bracketed
`code_interpreter_call` portion is shown; reasoning / message envelopes match
the cross-tool patterns below):

```
004 response.output_item.added                   (item.type=code_interpreter_call,
                                                   item.code="", status="in_progress")
005 response.code_interpreter_call.in_progress  (output_index=1)
006 response.code_interpreter_call_code.delta * N (incremental code chunks)
... (events 006-012 are 7 deltas in this capture)
013 response.code_interpreter_call_code.done    (full code emitted)
014 response.code_interpreter_call.interpreting (output_index=1)
015 response.code_interpreter_call.completed    (output_index=1)
016 response.output_item.done                   (item.type=code_interpreter_call,
                                                   status="completed")
```

A representative `_code.delta` event from the same capture:

```json
{
  "type": "response.code_interpreter_call_code.delta",
  "item_id": "ci_0177a3e3afd00ad50169ebfa9e39508197b8307e2f33af2a26",
  "output_index": 1,
  "sequence_number": 6,
  "delta": "sum",
  "obfuscation": "XPzusfd2FM5P9"
}
```

The terminal `_code.done` event:

```json
{
  "type": "response.code_interpreter_call_code.done",
  "item_id": "ci_0177a3e3afd00ad50169ebfa9e39508197b8307e2f33af2a26",
  "output_index": 1,
  "sequence_number": 13,
  "code": "sum(range(1,11))"
}
```

**Quirks**:

- `code_interpreter` emits four kinds of sub-events between
  `output_item.added` and `output_item.done`:
  `.in_progress`, `_code.delta`, `_code.done`, `.interpreting`, and
  `.completed`. All are present in real captures.
- `_code.delta` events carry an `obfuscation` field — an opaque per-chunk
  token. Pass it through if you observe it; do not rely on it.
- The captured `output_item.added` for the `code_interpreter_call` item
  carries `code: ""` and `status: "in_progress"`. Final `output_item.done`
  carries the full `code` and `status: "completed"`.

### `file_search`

**Status**: Partial. The transformer is implemented
(`crates/mcp/src/transform/transformer.rs::ResponseTransformer::to_file_search_call`)
and the `BuiltinToolType::FileSearch` variant is wired through orchestration,
but no ground-truth capture against real OpenAI is available in this
repository. **Schema only — no captured event sequence.**

**Server config recipe**:

```yaml title="mcp.yaml"
servers:
  - name: vector-search
    protocol: streamable
    url: "http://localhost:9200/mcp"
    builtin_type: file_search
    builtin_tool_name: search
    tools:
      search:
        response_format: file_search_call
```

The MCP server's named tool is expected to return an object containing
`queries` (list) or a single `query` string, plus an optional `results[]`
array shaped like `FileSearchResult`. See
`crates/protocols/src/responses.rs::FileSearchResult`.

**Expected response item shape** (from
`crates/protocols/src/responses.rs::ResponseOutputItem::FileSearchCall`):

```json
{
  "id": "fs_<call-id>",
  "type": "file_search_call",
  "status": "completed",
  "queries": ["…"],
  "results": [
    {"…file-search-result fields…"}
  ]
}
```

`results` is omitted when the MCP server returns no hits.

**Streaming event sequence**: no ground-truth capture available. Per
`crates/protocols/src/event_types.rs::FileSearchCallEvent` the event names
follow the same convention as `web_search_call`:
`response.file_search_call.in_progress`,
`response.file_search_call.searching`,
`response.file_search_call.completed`. The full envelope (reasoning →
`output_item.added` → sub-events → `output_item.done` → message → completed)
is expected to mirror `web_search_preview`.

**Quirks**: vector-store integration is the responsibility of the MCP server
backing this builtin type. The gateway transformer does not itself open a
vector index.

---

## Tools that do NOT use the MCP-builtin pattern

The remaining variants of `crates/protocols/src/responses.rs::ResponseTool` are
declared in the protocol surface but are not dispatched through MCP-builtin
servers, either because they are user-defined or because they target a
non-Responses-API surface. Listed here so contributors do not try to register
an MCP server for them.

| `ResponseTool` variant     | Why not MCP-builtin                                                                                  |
|----------------------------|------------------------------------------------------------------------------------------------------|
| `function`                 | User-defined function tool. The model emits `function_call`; the client owns execution.             |
| `mcp`                      | The user's own MCP server, declared inline by `server_label` / `server_url`. Not a hosted-tool surface. |
| `custom`                   | User-defined custom tool with optional grammar-constrained input. Client-executed.                  |
| `namespace`                | Grouping construct over `Function` and `Custom` tools. Not directly callable.                       |
| `computer`                 | Declared in protocol; no MCP-builtin transformer or `BuiltinToolType` variant today.                |
| `computer_use_preview`     | Same — declared, no MCP-builtin surface yet.                                                         |
| `shell`                    | Codex-specific containerized shell tool; not on the OpenAI Responses API hosted-tool surface.       |
| `local_shell`              | Codex-specific host-execute shell; client-executed.                                                  |
| `apply_patch`              | Codex-specific file-edit tool; client-executed.                                                      |
| `tool_search` (planned)    | Not yet a `ResponseTool` variant. Tracked in PR #1383.                                              |

To add a hosted tool to the MCP-builtin surface, extend
`BuiltinToolType` and add a transformer arm — see
[Adding a new hosted tool](#adding-a-new-hosted-tool) below.

---

## Cross-tool patterns

These behaviors hold for every hosted tool dispatched through the MCP-builtin
path. Tests and downstream consumers should rely on them rather than on
tool-specific assumptions.

- **Reasoning items appear unconditionally.** Every Responses API response —
  including `plain_text`, `function_call`, `image_generation`,
  `web_search_*`, `code_interpreter` — emits one or more `reasoning` items
  in the output array, both before and around the tool call. The first
  output item is **never** guaranteed to be the tool call; expect a
  `reasoning` item in front. Real captures (
  `/tmp/openai_ground_truth/results/`) confirm this for all five sampled
  scenarios.
- **Output items are bracketed.** Every output item in the streaming wire
  format is opened by `response.output_item.added` and closed by
  `response.output_item.done`. The closing event is the **terminal** signal
  for that item — its absence means the item did not finalize.
- **Hosted-tool sub-events sit between the bracket.** All
  `response.<tool_kind>.<state>` events for a given output item appear
  strictly between that item's `output_item.added` and `output_item.done`.
- **Universal envelope.** Every response (tool-bearing or not) starts with
  `response.created` then `response.in_progress`, and ends with
  `response.completed`. The terminal `response.completed` payload carries
  the canonical final `output[]` array with all items in their fully
  resolved form.
- **`output_item.done` carries the canonical streaming snapshot.** When the
  streaming and non-streaming snapshots disagree (as for `image_generation`
  where `output_item.done` reports `status: "generating"` mid-stream), trust
  the non-streaming response or the final `response.completed` payload's
  `output[]` for the resolved status.
- **Sub-event presence varies by tool.** `web_search_preview`, `web_search`,
  and `code_interpreter` emit a `<tool>.completed` sub-event before
  `output_item.done`. `image_generation` does **not** — its terminal signal
  is `output_item.done` alone. See per-tool quirks above.

---

## `user` forwarding

The Responses API request body carries an optional `user` field
(`crates/protocols/src/responses.rs::ResponsesRequest::user`) — the same
client-supplied identifier defined for OpenAI's Chat Completions API. For
hosted tools dispatched via the MCP-builtin path, the gateway is being
updated to forward the request's `user` value into the MCP tool-call
arguments so the upstream MCP server can attribute usage and apply
per-user rate limits. Tracking PR: see PR #&lt;TBD&gt; (in-flight). This
page should be updated to describe the final propagation surface once
that PR merges.

---

## Adding a new hosted tool

Checklist for landing a new MCP-builtin hosted tool. All edits stay inside
the gateway and protocol crates; no client changes are required.

1. **Protocol surface** — add a new variant to
   `crates/protocols/src/responses.rs::ResponseTool` (e.g.
   `#[serde(rename = "<new_tool>")] NewTool(NewToolDef)`). Add the matching
   `ResponseOutputItem::NewToolCall` variant and any `*Status` /
   `*Action` enums needed for the response shape. Add an `ItemType` and
   `<NewTool>CallEvent` constants in
   `crates/protocols/src/event_types.rs`.
2. **MCP config** — extend
   `crates/mcp/src/core/config.rs::BuiltinToolType` with a new `NewTool`
   variant. Add the matching `Display` arm and `response_format()` mapping
   to the new `ResponseFormatConfig::NewToolCall`. Update the validation
   tests for the duplicate-builtin-type guard.
3. **Transformer** — add a new arm in
   `crates/mcp/src/transform/transformer.rs::ResponseTransformer::transform`
   and a corresponding `to_new_tool_call` helper that pulls fields from the
   MCP `CallToolResult` (`result`, embedded text-block JSON, etc.) and
   constructs the `ResponseOutputItem::NewToolCall`.
4. **Router execution sites** — update the four router-side dispatch
   helpers in `model_gateway/src/routers/common/mcp_utils.rs`:
    - `extract_builtin_types` — recognize the new `ResponseTool::NewTool`
      and map it to `BuiltinToolType::NewTool`.
    - `collect_builtin_routing` — same mapping in the routing-info
      collector.
    - `ensure_request_mcp_client` and `ensure_mcp_servers` will pick up
      the new variant via the helpers above; verify the new builtin type
      flows through.
   The OpenAI hosted-tool stream synthesis in
   `model_gateway/src/routers/openai/mcp/tool_handler.rs` and
   `model_gateway/src/routers/openai/mcp/tool_loop.rs` may also need a new
   `output_item.added` / `output_item.done` recognition arm if the new tool
   has an item type discriminator that the loop's gate logic does not yet
   know about.
5. **End-to-end test** — add a test under `e2e_test/responses/` modeled on
   `test_image_generation.py`. Register the new tool on the in-process mock
   in `e2e_test/infra/mock_mcp_server.py` (a `@FastMCP.tool`-decorated
   function returning a dict in the expected shape), and write the MCP
   config fixture in `e2e_test/responses/conftest.py` mirroring
   `_image_generation_mcp_config`.

The image-generation lane (R6.5 in `e2e_test/responses/test_image_generation.py`)
is the canonical worked example — it covers the OpenAI cloud router, the
gRPC-harmony router (gpt-oss via SGLang), and the gRPC-regular router (Llama
via vLLM) for the same MCP-builtin dispatch. Use it as the template when
adding a new hosted tool.

---

## Reference

- Configuration enum: `crates/mcp/src/core/config.rs::BuiltinToolType`
- Transformer: `crates/mcp/src/transform/transformer.rs::ResponseTransformer`
- Routing helpers: `model_gateway/src/routers/common/mcp_utils.rs`
- OpenAI passthrough handler: `model_gateway/src/routers/openai/mcp/tool_handler.rs`
- Tool union: `crates/protocols/src/responses.rs::ResponseTool`
- Response items: `crates/protocols/src/responses.rs::ResponseOutputItem`
- Event constants: `crates/protocols/src/event_types.rs`
- Canonical fixture: `e2e_test/responses/conftest.py::_image_generation_mcp_config`
- Canonical e2e: `e2e_test/responses/test_image_generation.py`
- Mock MCP server: `e2e_test/infra/mock_mcp_server.py`
- MCP general usage: [MCP in Responses API](getting-started/mcp.md),
  [Model Context Protocol concepts](concepts/extensibility/mcp.md)
