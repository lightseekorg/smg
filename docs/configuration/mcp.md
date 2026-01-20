---
title: MCP Configuration
---

# MCP Configuration

Configure Model Context Protocol (MCP) servers to extend SMG with external tools, prompts, and resources.

---

## Overview

### What is MCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open standard for connecting AI systems with external tools and data sources. It provides a unified interface for:

- **Tools**: Executable functions that models can invoke (e.g., web search, code execution)
- **Prompts**: Reusable prompt templates
- **Resources**: External data sources (files, databases, APIs)

### How SMG Integrates with MCP

SMG acts as an MCP client that connects to MCP servers on behalf of the model. When configured:

1. **Startup**: SMG connects to configured MCP servers and discovers available tools
2. **Request**: Tools are exposed to the model during inference
3. **Execution**: When the model calls a tool, SMG routes the request to the appropriate MCP server
4. **Response**: Tool results are returned to the model for continued generation

This architecture allows models to access external capabilities without direct network access, while SMG handles connection management, pooling, and authentication.

---

## Configuration File

### CLI Option

Specify the MCP configuration file path:

| Option | `--mcp-config-path` |
|--------|---------------------|
| Type | File path |
| Default | None (MCP disabled) |

```bash
smg --worker-urls http://localhost:8000 --mcp-config-path /etc/smg/mcp.yaml
```

### Configuration File Format

The configuration file uses YAML format with the following structure:

```yaml
# Static MCP servers (connected at startup)
servers:
  - name: "server-name"
    protocol: sse | stdio | streamable
    # Transport-specific options...
    required: false  # Optional: fail startup if connection fails
    proxy: ...       # Optional: per-server proxy override

# Global proxy configuration (default for all servers)
proxy:
  http: "http://proxy:8080"
  https: "http://proxy:8080"
  no_proxy: "localhost,127.0.0.1"

# Connection pool settings
pool:
  max_connections: 100
  idle_timeout: 300

# Tool inventory refresh settings
inventory:
  enable_refresh: true
  tool_ttl: 300
  refresh_interval: 60
  refresh_on_error: true

# Pre-warm connections at startup
warmup:
  - url: "http://localhost:3000/sse"
    label: "local-dev"
```

---

## Transport Types

MCP supports multiple transport protocols for connecting to servers. The transport is specified via the `protocol` field.

### stdio Transport

Command-based MCP servers that communicate via standard input/output. Best for local tools and sandboxed environments.

```yaml
servers:
  - name: "filesystem"
    protocol: stdio
    command: "npx"
    args:
      - "-y"
      - "@modelcontextprotocol/server-filesystem"
      - "/workspace"
    envs:
      NODE_ENV: "production"
```

| Field | Type | Description |
|-------|------|-------------|
| `command` | String | Executable to run |
| `args` | List | Command-line arguments |
| `envs` | Map | Environment variables |

**Use Cases**:

- Local file system access
- Sandboxed code execution
- CLI tool wrappers

### sse Transport

Server-Sent Events transport for HTTP-based MCP servers. Supports real-time streaming and works well with remote servers.

```yaml
servers:
  - name: "web-tools"
    protocol: sse
    url: "https://mcp.example.com/sse"
    token: "your-api-token"  # Optional: Bearer token
```

| Field | Type | Description |
|-------|------|-------------|
| `url` | String | SSE endpoint URL (must end with `/sse`) |
| `token` | String | Optional Bearer token for authentication |

!!! tip "Environment Variable Expansion"
    Tokens support environment variable expansion using the `${VAR_NAME}` syntax. For example: `token: "${MCP_API_TOKEN}"`. This allows you to keep sensitive credentials out of configuration files.

**Use Cases**:

- Remote hosted MCP services
- Real-time tool responses
- Multi-tenant MCP platforms

### streamable Transport

HTTP-based transport with bidirectional streaming support. The newest MCP transport offering improved performance.

```yaml
servers:
  - name: "compute-tools"
    protocol: streamable
    url: "https://mcp.example.com"
    token: "your-api-token"  # Optional: Bearer token
```

| Field | Type | Description |
|-------|------|-------------|
| `url` | String | HTTP endpoint URL |
| `token` | String | Optional Bearer token for authentication |

**Use Cases**:

- High-performance remote tools
- Long-running operations
- Bidirectional streaming

!!! note "Proxy Support"
    The `sse` transport fully supports proxy configuration. The `streamable` transport does not currently support proxies; a warning will be logged if proxy is configured.

---

## Server Configuration Options

### Required Server

Mark a server as required for startup:

```yaml
servers:
  - name: "critical-tools"
    protocol: sse
    url: "https://mcp.example.com/sse"
    required: true  # Router fails to start if this server is unreachable
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `required` | Boolean | `false` | Fail startup if server cannot be reached |

### Per-Server Proxy Override

Override the global proxy for a specific server:

```yaml
servers:
  - name: "internal-tools"
    protocol: sse
    url: "http://internal.example.com/sse"
    proxy:
      http: "http://internal-proxy:8080"
```

Force direct connection (bypass global proxy):

```yaml
servers:
  - name: "local-tools"
    protocol: sse
    url: "http://localhost:3000/sse"
    proxy: null  # Explicitly disable proxy
```

---

## Connection Pool

The connection pool manages connections to dynamic MCP servers (servers specified per-request rather than in configuration).

```yaml
pool:
  max_connections: 100
  idle_timeout: 300
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_connections` | Integer | `100` | Maximum cached connections (LRU eviction) |
| `idle_timeout` | Integer | `300` | Idle timeout in seconds before closing connection |

!!! note "Dynamic Client Limit"
    The MCP manager internally enforces a hard limit of 200 dynamic clients (`MAX_DYNAMIC_CLIENTS`) regardless of `max_connections`. This prevents unbounded resource usage when many unique server URLs are requested.

### How Connection Pooling Works

1. **Static servers** (from config) are connected at startup and never evicted
2. **Dynamic servers** (from request `server_url`) use the connection pool
3. Pool uses LRU (Least Recently Used) eviction when capacity is reached
4. Evicted connections have their tools removed from the inventory

**Performance Characteristics**:

- Cache hit: <1ms (returns existing connection)
- Cache miss: 70-650ms (new connection establishment)

---

## Tool Inventory

The tool inventory caches tool definitions from all connected MCP servers.

```yaml
inventory:
  enable_refresh: true
  tool_ttl: 300
  refresh_interval: 60
  refresh_on_error: true
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_refresh` | Boolean | `true` | Enable automatic tool inventory refresh |
| `tool_ttl` | Integer | `300` | Tool cache TTL in seconds |
| `refresh_interval` | Integer | `60` | Background refresh interval in seconds |
| `refresh_on_error` | Boolean | `true` | Refresh inventory when tool not found |

### Inventory Behavior

- **Startup**: Tools are discovered from all configured servers
- **Background refresh**: Static servers refresh every `refresh_interval` seconds
- **On-demand refresh**: Dynamic servers refresh when accessed via `get_or_create_client`
- **Error recovery**: When `refresh_on_error` is enabled, missing tool triggers refresh

---

## Proxy Configuration

Configure HTTP/HTTPS proxy for MCP server connections. This affects only MCP traffic, not LLM API traffic.

### Global Proxy

Applied to all servers by default:

```yaml
proxy:
  http: "http://proxy.internal:8080"
  https: "https://proxy.internal:8080"
  no_proxy: "localhost,127.0.0.1,*.internal,10.*"
  username: "proxy-user"      # Optional: proxy authentication
  password: "proxy-password"  # Optional: proxy authentication
```

| Field | Type | Description |
|-------|------|-------------|
| `http` | String | HTTP proxy URL |
| `https` | String | HTTPS proxy URL |
| `no_proxy` | String | Comma-separated hosts to exclude from proxying |
| `username` | String | Optional proxy username for basic auth |
| `password` | String | Optional proxy password for basic auth |

### No-Proxy Patterns

The `no_proxy` field supports:

- Exact hostnames: `localhost`, `internal.example.com`
- IP addresses: `127.0.0.1`, `192.168.1.100`
- Wildcard patterns: `*.internal`, `10.*`

### Environment Variables

Proxy can also be configured via environment variables (used when no config proxy is set):

| Variable | Priority | Description |
|----------|----------|-------------|
| `MCP_HTTP_PROXY` | Highest | MCP-specific HTTP proxy |
| `HTTP_PROXY` | Lower | Standard HTTP proxy |
| `MCP_HTTPS_PROXY` | Highest | MCP-specific HTTPS proxy |
| `HTTPS_PROXY` | Lower | Standard HTTPS proxy |
| `MCP_NO_PROXY` | Highest | MCP-specific exclusions |
| `NO_PROXY` | Lower | Standard exclusions |

### Proxy Priority Resolution

For each server connection:

1. Server-specific `proxy` config (if set)
2. Global `proxy` config (if set)
3. Environment variables (fallback)
4. Direct connection (no proxy)

---

## Warmup Configuration

Pre-warm connections at startup for faster first requests:

```yaml
warmup:
  - url: "http://localhost:3000/sse"
    label: "local-dev"
    token: "optional-token"
```

| Field | Type | Description |
|-------|------|-------------|
| `url` | String | Server URL to pre-connect |
| `label` | String | Human-readable server name |
| `token` | String | Optional authentication token |

---

## Complete Configuration Examples

### Minimal Configuration

Single MCP server with defaults:

```yaml
servers:
  - name: "tools"
    protocol: sse
    url: "http://localhost:3000/sse"
```

### Local Development

Multiple local servers for development:

```yaml
servers:
  # File system access
  - name: "filesystem"
    protocol: stdio
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]

  # Local web tools
  - name: "web"
    protocol: sse
    url: "http://localhost:3001/sse"

pool:
  max_connections: 50
  idle_timeout: 600

inventory:
  refresh_interval: 30  # Faster refresh for development
```

### Production Deployment

Enterprise configuration with proxy and authentication:

```yaml
# Global proxy for corporate network
proxy:
  http: "http://proxy.corp.example.com:8080"
  https: "http://proxy.corp.example.com:8080"
  no_proxy: "localhost,127.0.0.1,*.internal.example.com"

# Connection pool tuned for high throughput
pool:
  max_connections: 200
  idle_timeout: 300

# Conservative inventory refresh
inventory:
  enable_refresh: true
  tool_ttl: 600        # 10 minute TTL
  refresh_interval: 120  # 2 minute refresh
  refresh_on_error: true

servers:
  # Critical internal tools - required for operation
  - name: "internal-tools"
    protocol: sse
    url: "https://mcp.internal.example.com/sse"
    token: "${MCP_INTERNAL_TOKEN}"
    required: true
    proxy: null  # Direct connection to internal server

  # External vendor tools
  - name: "vendor-tools"
    protocol: streamable
    url: "https://api.vendor.com/mcp"
    token: "${MCP_VENDOR_TOKEN}"
    required: false
    # Uses global proxy

  # Sandboxed code execution
  - name: "code-sandbox"
    protocol: stdio
    command: "/usr/local/bin/mcp-sandbox"
    args: ["--timeout", "30"]
    envs:
      SANDBOX_MODE: "strict"
    required: true

# Pre-warm critical connections
warmup:
  - url: "https://mcp.internal.example.com/sse"
    label: "internal-tools"
    token: "${MCP_INTERNAL_TOKEN}"
```

### Multi-Tenant Configuration

Different tools per environment:

```yaml
servers:
  # Shared tools for all tenants
  - name: "shared-tools"
    protocol: sse
    url: "https://mcp.shared.example.com/sse"
    token: "${MCP_SHARED_TOKEN}"
    required: true

  # Premium tenant tools
  - name: "premium-tools"
    protocol: sse
    url: "https://mcp.premium.example.com/sse"
    token: "${MCP_PREMIUM_TOKEN}"
    required: false

pool:
  max_connections: 100  # Per-tenant connections managed by pool

inventory:
  tool_ttl: 300
  refresh_on_error: true  # Handle tenant-specific tool availability
```

---

## Troubleshooting

### Connection Failures

**Symptom**: "Failed to connect to static server" at startup

**Possible Causes**:

1. Server URL is incorrect or unreachable
2. Proxy configuration blocking connection
3. Authentication token invalid or expired

**Resolution**:

```bash
# Test server connectivity
curl -v https://mcp.example.com/sse

# Check proxy settings
curl -v --proxy http://proxy:8080 https://mcp.example.com/sse
```

### Tool Not Found

**Symptom**: "Tool not found" error during inference

**Possible Causes**:

1. Tool inventory not refreshed
2. Server connection was evicted from pool
3. Tool was removed from MCP server

**Resolution**:

Enable `refresh_on_error` in inventory config, or check server connectivity.

### Proxy Issues

**Symptom**: Connection timeouts or SSL errors

**Resolution**:

- Verify proxy URL format (`http://host:port`)
- Check `no_proxy` patterns for internal servers
- Ensure proxy supports HTTPS CONNECT for TLS connections
