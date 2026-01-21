//! Core MCP client infrastructure.

pub mod config;
pub mod manager;
pub mod oauth;
pub mod pool;
pub mod proxy;

pub use config::{McpConfig, McpServerConfig, McpTransport, Tool};
pub use manager::{McpManager, RequestMcpContext};
pub use pool::McpConnectionPool;
