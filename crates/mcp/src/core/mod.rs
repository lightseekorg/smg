//! Core MCP client infrastructure.

/// Sentinel value used as the server key when the actual server cannot be determined
/// (e.g. tool not found, ambiguous collision).
pub const UNKNOWN_SERVER_KEY: &str = "unknown";

pub mod config;
pub mod handler;
pub mod metrics;
pub mod oauth;
pub mod orchestrator;
pub mod pool;
pub mod proxy;
pub mod reconnect;
pub mod session;

pub use config::{
    ArgMappingConfig, BuiltinToolType, ConfigValidationError, McpConfig, McpServerConfig,
    McpTransport, PolicyConfig, PolicyDecisionConfig, ResponseFormatConfig, ServerPolicyConfig,
    Tool, ToolConfig, TrustLevelConfig,
};
pub use handler::{HandlerRequestContext, RefreshRequest, SmgClientHandler};
pub use metrics::{LatencySnapshot, McpMetrics, MetricsSnapshot};
pub use orchestrator::{
    McpOrchestrator, McpRequestContext, PendingToolExecution, ToolCallResult, ToolExecutionInput,
    ToolExecutionOutput, ToolExecutionResult,
};
pub use pool::{McpConnectionPool, PoolKey};
pub use reconnect::ReconnectionManager;
pub use session::{
    inject_user_into_hosted_tool_args, resolve_response_format, McpServerBinding, McpToolSession,
    DEFAULT_SERVER_LABEL,
};
