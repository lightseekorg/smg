//! Model Context Protocol (MCP) client implementation.
//!
//! ## Modules
//!
//! - [`core`]: MCP client infrastructure (manager, config, connections)
//! - [`inventory`]: Tool storage and indexing
//! - [`approval`]: Approval system for tool execution
//!
//! ## Shared Types
//!
//! - [`ToolAnnotations`]: Tool behavior hints (read_only, destructive, etc.)
//! - [`TenantContext`]: Per-tenant isolation and configuration

// Shared types (used across modules)
pub mod annotations;
pub mod error;
pub mod tenant;
pub mod transform;

// Subsystems
pub mod approval;
pub mod core;
pub mod inventory;

// Backward-compatible re-exports (old module paths)
// These allow `mcp::config::*` to continue working
pub use core::{config, pool as connection_pool};
// Re-export from core
pub use core::{
    ArgMappingConfig, HandlerRequestContext, LatencySnapshot, McpConfig, McpMetrics,
    McpOrchestrator, McpRequestContext, McpServerConfig, McpTransport, MetricsSnapshot,
    PolicyConfig, PolicyDecisionConfig, PoolKey, RefreshRequest, ResponseFormatConfig,
    ServerPolicyConfig, SmgClientHandler, Tool, ToolCallResult, ToolConfig, ToolExecutionInput,
    ToolExecutionOutput, TrustLevelConfig,
};

// Re-export shared types
pub use annotations::{AnnotationType, ToolAnnotations};
// Re-export from approval
pub use approval::{
    ApprovalDecision, ApprovalKey, ApprovalManager, ApprovalMode, ApprovalOutcome, ApprovalParams,
    AuditEntry, AuditLog, DecisionResult, DecisionSource, McpApprovalRequest, McpApprovalResponse,
    PolicyDecision, PolicyEngine, PolicyRule, RuleCondition, RulePattern, ServerPolicy, TrustLevel,
};
pub use error::{ApprovalError, McpError, McpResult};
// Re-export from inventory
pub use inventory::{
    AliasTarget, ArgMapping, QualifiedToolName, ToolCategory, ToolEntry, ToolInventory,
};
pub use tenant::{SessionId, TenantContext, TenantId};
// Re-export from transform
pub use transform::{ResponseFormat, ResponseTransformer};
