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

This flag is generic. It does not imply any vendor-specific behavior and does
not change transport setup or tool execution on its own.
