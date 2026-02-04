//! Core MCP client infrastructure.

pub mod config;
pub mod handler;
pub mod metrics;
pub mod oauth;
pub mod orchestrator;
pub mod pool;
pub mod proxy;

pub use config::{
    ArgMappingConfig, BuiltinToolType, ConfigValidationError, McpConfig, McpServerConfig,
    McpTransport, PolicyConfig, PolicyDecisionConfig, ResponseFormatConfig, ServerPolicyConfig,
    Tool, ToolConfig, TrustLevelConfig,
};
pub use handler::{HandlerRequestContext, RefreshRequest, SmgClientHandler};
pub use metrics::{LatencySnapshot, McpMetrics, MetricsSnapshot};
pub use orchestrator::{
    ApprovalResponseInput, McpOrchestrator, McpRequestContext, PendingApprovalOutput,
    ToolCallContext, ToolCallResult, ToolExecutionInput, ToolExecutionOutcome, ToolExecutionOutput,
};
pub use pool::{McpConnectionPool, PoolKey};
