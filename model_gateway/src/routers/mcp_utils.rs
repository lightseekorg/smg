//! Shared MCP utilities for routers.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use axum::http::HeaderMap;
use tracing::warn;

use super::header_utils::should_forward_request_header;
use crate::{
    mcp::{McpOrchestrator, McpServerConfig, McpTransport},
    protocols::responses::{ResponseTool, ResponseToolType},
};

/// Default maximum tool loop iterations (safety limit).
pub const DEFAULT_MAX_ITERATIONS: usize = 10;

/// Configuration for MCP tool calling loops.
#[derive(Debug, Clone)]
pub struct McpLoopConfig {
    /// Maximum iterations (default: DEFAULT_MAX_ITERATIONS).
    pub max_iterations: usize,
    /// Server keys for filtering MCP tools.
    pub server_keys: Vec<String>,
}

impl Default for McpLoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
            server_keys: Vec::new(),
        }
    }
}

/// Extract MCP server label from request tools, falling back to default.
pub fn extract_server_label(tools: Option<&[ResponseTool]>, default_label: &str) -> String {
    tools
        .and_then(|tools| {
            tools.iter().find_map(|tool| {
                if matches!(tool.r#type, ResponseToolType::Mcp) {
                    tool.server_label.clone()
                } else {
                    None
                }
            })
        })
        .unwrap_or_else(|| default_label.to_string())
}

/// Ensure MCP clients are connected for request-level MCP tools.
///
/// Extracts server configurations from request tools and establishes connections.
/// Forwards filtered HTTP request headers (auth, tracing, correlation IDs) to MCP servers.
///
/// Returns `Some((orchestrator, server_keys))` if connections were established,
/// `None` if no MCP tools with server_url were found.
pub async fn ensure_request_mcp_client(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    tools: &[ResponseTool],
    request_headers: Option<&HeaderMap>,
) -> Option<(Arc<McpOrchestrator>, Vec<String>)> {
    let mut server_keys = HashSet::new();
    let mut has_mcp_tools = false;
    let forwarded_headers = extract_forwardable_headers(request_headers);

    for tool in tools {
        let Some(server_url) = tool
            .server_url
            .as_ref()
            .filter(|_| matches!(tool.r#type, ResponseToolType::Mcp))
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        else {
            continue;
        };

        has_mcp_tools = true;

        if !(server_url.starts_with("http://") || server_url.starts_with("https://")) {
            warn!(
                "Ignoring MCP server_url with unsupported scheme: {}",
                server_url
            );
            continue;
        }

        let name = tool
            .server_label
            .clone()
            .unwrap_or_else(|| "request-mcp".to_string());
        let token = tool.authorization.clone();
        let headers = forwarded_headers.clone();
        let server_url = server_url.to_string();

        let transport = if server_url.contains("/sse") {
            McpTransport::Sse {
                url: server_url,
                token,
                headers,
            }
        } else {
            McpTransport::Streamable {
                url: server_url,
                token,
                headers,
            }
        };

        let server_config = McpServerConfig {
            name,
            transport,
            proxy: None,
            required: false,
            tools: None,
        };

        let server_key = McpOrchestrator::server_key(&server_config);

        match mcp_orchestrator.connect_dynamic_server(server_config).await {
            Ok(_) => {
                server_keys.insert(server_key);
            }
            Err(err) => {
                warn!("Failed to connect MCP server {}: {}", server_key, err);
            }
        }
    }

    if has_mcp_tools && !server_keys.is_empty() {
        Some((mcp_orchestrator.clone(), server_keys.into_iter().collect()))
    } else {
        None
    }
}

/// Extract headers that should be forwarded to MCP servers.
fn extract_forwardable_headers(request_headers: Option<&HeaderMap>) -> HashMap<String, String> {
    let Some(headers) = request_headers else {
        return HashMap::new();
    };

    headers
        .iter()
        .filter_map(|(name, value)| {
            let name_str = name.as_str();
            if should_forward_request_header(name_str) {
                value
                    .to_str()
                    .ok()
                    .map(|v| (name_str.to_string(), v.to_string()))
            } else {
                None
            }
        })
        .collect()
}
