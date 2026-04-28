# Internal MCP Servers

SMG supports marking an MCP server as internal by setting `internal: true` in
the MCP config.

Internal servers are still available to the gateway runtime, but they can be
treated differently from normal client-visible MCP servers by higher layers.

Example:

```yaml
servers:
  - name: internal-memory
    protocol: streamable
    url: http://127.0.0.1:28080/mcp
    internal: true
```

In the current implementation, `internal: true` applies only to self-provided
MCP servers declared under `servers:`. The model may still see and call these
tools during gateway-managed tool loops, but OpenAI Responses client-facing
output hides internal non-builtin tool details before returning data to the
client. That includes final non-streaming responses, final streaming
`response.completed` events, live streaming tool-call events, live
`mcp_list_tools` events, and response envelope `tools` / `tool_choice` fields.

This flag does not apply to builtin-routed MCP results such as
`web_search_call`, `code_interpreter_call`, or `file_search_call`.

This flag is generic. It does not imply any vendor-specific behavior and does
not change transport setup or tool execution on its own.
