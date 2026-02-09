---
title: MCP in Responses API
---

# MCP in Responses API

MCP tool execution in SMG is currently implemented through the **Responses API** (`/v1/responses`), not through Chat Completions tool loops.

<div class="prerequisites" markdown>

#### Before you begin

- Completed the [Getting Started](index.md) guide
- Running SMG with a model that supports tool calling

</div>

---

## 1. Create `mcp.yaml`

```yaml title="mcp.yaml"
servers:
  - name: brave
    protocol: sse
    url: "https://mcp.brave.com/sse"
    token: "${BRAVE_API_KEY}"
    required: true

    tools:
      brave_web_search:
        alias: web_search
        response_format: web_search_call

# Optional: built-in tool routing
# builtin_type: web_search_preview
# builtin_tool_name: brave_web_search

# Optional: global proxy for MCP traffic
# proxy:
#   http: "http://proxy.internal:8080"
#   https: "http://proxy.internal:8080"
#   no_proxy: "localhost,127.0.0.1"

# Optional: approval policy
# policy:
#   default: allow
#   servers:
#     brave:
#       trust_level: trusted
```

---

## 2. Start SMG with MCP config

```bash
smg \
  --worker-urls grpc://localhost:50051 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --mcp-config-path /path/to/mcp.yaml
```

`--mcp-config-path` loads MCP servers, inventory settings, proxy policy, and approval policy.

---

## 3. Call `/v1/responses` with MCP tools

### Static MCP server by label

```bash
curl http://localhost:30000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "input": "Find Rust 2024 edition highlights",
    "tools": [
      {
        "type": "mcp",
        "server_label": "brave"
      }
    ]
  }'
```

### Dynamic MCP server by URL

```bash
curl http://localhost:30000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "input": "Search for SMG docs",
    "tools": [
      {
        "type": "mcp",
        "server_url": "https://mcp.example.com/sse",
        "authorization": "Bearer token-value",
        "headers": {
          "X-Tenant-ID": "tenant-a"
        },
        "server_label": "tenant-search"
      }
    ]
  }'
```

### Built-in tool routed to MCP

```bash
curl http://localhost:30000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "input": "Search latest release notes",
    "tools": [
      {
        "type": "web_search_preview"
      }
    ]
  }'
```

When `builtin_type` is configured in `mcp.yaml`, SMG routes the built-in tool to MCP.

---

## Approval Behavior

MCP approval mode is API-dependent in the orchestrator:

- Responses API supports interactive MCP approval handling
- Other API paths use policy-only approval mode

If a tool requires approval and the flow does not support interactive approval handling, execution is denied or deferred based on policy.

---

## Verify

```bash
curl http://localhost:30000/health
curl http://localhost:29000/metrics | grep smg_mcp
```

---

## Next Steps

- [MCP Concepts](../concepts/extensibility/mcp.md) — full architecture and lifecycle
- [Responses API Reference](../reference/api/responses.md)
- [Configuration Reference](../reference/configuration.md#mcp-configuration)
