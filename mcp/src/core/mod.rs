//! Core MCP client infrastructure.

pub mod config;
pub mod handler;
pub mod manager;
pub mod metrics;
pub mod oauth;
pub mod pool;
pub mod proxy;

pub use config::{McpConfig, McpServerConfig, McpTransport, Tool};
pub use handler::{HandlerRequestContext, RefreshRequest, SmgClientHandler};
pub use manager::{McpManager, RequestMcpContext};
pub use metrics::{LatencySnapshot, McpMetrics, MetricsSnapshot};
pub use pool::McpConnectionPool;
