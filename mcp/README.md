# smg-mcp

Model Context Protocol (MCP) client implementation with approval system for SMG.

## Overview

This crate provides:

1. **MCP Client Management** - Connect to MCP servers (stdio, SSE, HTTP)
2. **Tool Inventory** - Cache and query tools with collision handling
3. **Approval System** - Dual-mode approval for tool execution

## Architecture

```
McpManager
├── Static Clients (always on)
├── Connection Pool (LRU, dynamic)
└── Tool Inventory (collision-aware)
         │
         ▼
Approval System
    Hints → Policy → Approval → Audit
```

## Modules

### Core MCP

- `McpManager` - Orchestrates MCP server connections
- `ToolInventory` - Caches tools with qualified naming (`server:tool`)
- `QualifiedToolName` - Prevents collisions when multiple servers have same tool name

### Approval System

#### ToolAnnotations (`annotations.rs`)

Hints about tool behavior, extracted from MCP server annotations:

```rust
ToolAnnotations {
    read_only: bool,      // No side effects
    destructive: bool,    // May cause irreversible changes
    idempotent: bool,     // Safe to retry
    open_world: bool,     // Accesses external systems
}
```

#### TenantContext (`tenant.rs`)

Per-tenant isolation:

```rust
TenantContext {
    tenant_id: TenantId,
    session_id: SessionId,
    rate_limits: RateLimits,
}
```

#### PolicyEngine (`policy.rs`)

Rule-based engine for automatic approval decisions:

```rust
PolicyEngine {
    default_policy: PolicyDecision,
    server_policies: Map<String, ServerPolicy>,
    tool_policies: Map<QualifiedToolName, ToolPolicy>,
    rules: Vec<PolicyRule>,
}
```

**Trust levels** for servers:
- `Trusted` - Allow everything
- `Standard` - Respect hints, apply rules
- `Untrusted` - Deny destructive operations
- `Sandboxed` - Read-only only

#### ApprovalManager (`approval.rs`)

Dual-mode coordinator for tool execution approval.

**Mode 1: Policy-Only** (default)
```
Request → PolicyEngine.evaluate() → Allow/Deny
```

**Mode 2: Interactive**
```
Request → ApprovalManager.request_interactive()
       → Returns McpApprovalRequest to client
       → User approves/denies
       → ApprovalManager.resolve()
```

#### AuditLog (`audit.rs`)

Decision logging for compliance and debugging:

```rust
AuditEntry {
    id, timestamp, tenant_id, request_id,
    server_key, tool_name, result, source,
}
```

## Usage

### Basic MCP Client

```rust
use smg_mcp::{McpManager, McpConfig};

let config = McpConfig::from_yaml_file("mcp.yaml")?;
let manager = McpManager::new(config).await?;

let tools = manager.list_tools();
let result = manager.call_tool("read_file", json!({"path": "/tmp/test"})).await?;
```

### With Approval System

```rust
use smg_mcp::{
    ApprovalManager, PolicyEngine, TenantContext,
    ApprovalMode, PolicyDecision, TrustLevel, ServerPolicy,
};

let policy_engine = PolicyEngine::new(audit_log.clone())
    .with_default_policy(PolicyDecision::Allow)
    .with_server_policy("trusted-server", ServerPolicy {
        default: PolicyDecision::Allow,
        trust_level: TrustLevel::Trusted,
    });

let approval_manager = ApprovalManager::new(
    Arc::new(policy_engine),
    audit_log,
);

let tenant_ctx = TenantContext::new("customer-123");

let outcome = approval_manager.handle_approval(
    ApprovalMode::PolicyOnly,
    "req-1", "server", "elicit-1",
    "delete_user", &hints, "Delete user?",
    &tenant_ctx,
).await?;
```

## File Structure

```
mcp/src/
├── lib.rs              # Public exports
├── error.rs            # Error types
├── annotations.rs      # ToolAnnotations (shared)
├── tenant.rs           # TenantContext (shared)
│
├── core/               # MCP client infrastructure
│   ├── mod.rs
│   ├── manager.rs      # McpManager
│   ├── config.rs       # Configuration
│   └── pool.rs         # LRU connection pool
│
├── inventory/          # Tool storage and indexing
│   ├── mod.rs
│   ├── index.rs        # ToolInventory
│   ├── types.rs        # QualifiedToolName, ToolEntry, etc.
│   └── args.rs         # ToolArgs
│
└── approval/           # Approval system
    ├── mod.rs
    ├── manager.rs      # ApprovalManager
    ├── policy.rs       # PolicyEngine
    └── audit.rs        # AuditLog
```

## Design Decisions

### Why Qualified Tool Names?

Multiple MCP servers can expose tools with the same name. Without qualification, the second registration overwrites the first. With `QualifiedToolName`, both are stored: `server-a:run_query`, `server-b:run_query`.

### Why Dual-Mode Approval?

- **Interactive mode**: For APIs that support user confirmation
- **Policy-only mode**: For batch processing or when you want fast auto-decisions

### Why Trust Levels?

Not all MCP servers are equal:
- `Trusted`: Your own internal servers
- `Standard`: Third-party but vetted servers
- `Untrusted`: User-provided servers
- `Sandboxed`: Completely isolated
