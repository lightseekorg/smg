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
MCP servers declared under `servers:`. It affects final assembled,
non-streaming MCP responses by allowing higher layers to strip internal server
tool lists and tool-call trace items before the response is returned to the
client.

This flag does not currently hide streaming output, and it does not apply to
builtin-routed MCP results such as `web_search_call`, `code_interpreter_call`,
or `file_search_call`.

This flag is generic. It does not imply any vendor-specific behavior and does
not change transport setup or tool execution on its own.
