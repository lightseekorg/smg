//! Core MCP client infrastructure.

pub mod config;
pub mod handler;
pub mod manager;
pub mod metrics;
pub mod oauth;
pub mod orchestrator;
pub mod pool;
pub mod proxy;

pub use config::{
    ArgMappingConfig, McpConfig, McpServerConfig, McpTransport, PolicyConfig, PolicyDecisionConfig,
    ResponseFormatConfig, ServerPolicyConfig, Tool, ToolConfig, TrustLevelConfig,
};
pub use handler::{HandlerRequestContext, RefreshRequest, SmgClientHandler};
pub use manager::{McpManager, RequestMcpContext};
pub use metrics::{LatencySnapshot, McpMetrics, MetricsSnapshot};
pub use orchestrator::{McpOrchestrator, McpRequestContext, ToolCallResult};
pub use pool::McpConnectionPool;
