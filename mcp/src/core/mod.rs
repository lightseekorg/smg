//! Core MCP client infrastructure.

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
    McpOrchestrator, McpRequestContext, ToolCallResult, ToolExecutionInput, ToolExecutionOutput,
};
pub use pool::{McpConnectionPool, PoolKey};
pub use reconnect::ReconnectionManager;
pub use session::McpToolSession;
