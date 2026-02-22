//! Shared MCP utilities for routers.

use std::{collections::HashMap, sync::Arc};

use openai_protocol::responses::{ResponseTool, ResponseToolType};
use smg_mcp::{
    BuiltinToolType, McpOrchestrator, McpServerBinding, McpServerConfig, McpToolSession,
    McpTransport, ResponseFormat,
};
use tracing::{debug, warn};

/// Default maximum tool loop iterations (safety limit).
pub const DEFAULT_MAX_ITERATIONS: usize = 10;

/// Protocol-agnostic MCP server descriptor for connection setup.
///
/// Contains only the fields needed by [`connect_mcp_servers`]. Each router
/// converts its protocol-specific type into this struct.
pub struct McpServerInput {
    pub label: String,
    pub url: Option<String>,
    pub authorization: Option<String>,
    pub headers: HashMap<String, String>,
    /// Optional per-server tool allowlist.
    pub allowed_tools: Option<Vec<String>>,
}

/// Connect to MCP servers described by protocol-agnostic inputs.
///
/// For each input:
/// - If `url` is present, connects a dynamic MCP server (SSE or Streamable).
/// - If `url` is absent, registers the label as a static server reference.
///
/// Returns a list of [`McpServerBinding`]s for successfully connected servers.
pub async fn connect_mcp_servers(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    inputs: &[McpServerInput],
) -> Vec<McpServerBinding> {
    let mut mcp_servers: Vec<McpServerBinding> = Vec::new();

    for input in inputs {
        // Case A: Dynamic Server (Has URL)
        if let Some(server_url) = input
            .url
            .as_ref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            if !(server_url.starts_with("http://") || server_url.starts_with("https://")) {
                warn!(
                    "Ignoring MCP server_url with unsupported scheme: {}",
                    server_url
                );
                continue;
            }

            let token = input.authorization.clone();
            let headers = input.headers.clone();
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
                name: input.label.clone(),
                transport,
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
            };

            let server_key = McpOrchestrator::server_key(&server_config);

            match mcp_orchestrator.connect_dynamic_server(server_config).await {
                Ok(_) => {
                    if !mcp_servers.iter().any(|b| b.server_key == server_key) {
                        mcp_servers.push(McpServerBinding {
                            label: input.label.clone(),
                            server_key,
                            allowed_tools: input.allowed_tools.clone(),
                        });
                    }
                }
                Err(err) => {
                    warn!("Failed to connect MCP server {}: {}", server_key, err);
                }
            }
        }
        // Case B: Static Server (No URL)
        else if !mcp_servers.iter().any(|b| b.server_key == input.label) {
            mcp_servers.push(McpServerBinding {
                label: input.label.clone(),
                server_key: input.label.clone(),
                allowed_tools: input.allowed_tools.clone(),
            });
        }
    }

    mcp_servers
}

/// Routing information for a built-in tool type.
///
/// When a built-in tool type (web_search_preview, code_interpreter, file_search)
/// is configured to route to an MCP server, this struct holds the routing details.
#[derive(Debug, Clone)]
pub struct BuiltinToolRouting {
    /// The built-in tool type being routed.
    pub builtin_type: BuiltinToolType,
    /// The MCP server name to route to.
    pub server_name: String,
    /// The MCP tool name to call on the server.
    pub tool_name: String,
    /// The response format for transforming the output.
    pub response_format: ResponseFormat,
}

/// Collect routing information for built-in tools in a request.
///
/// Scans request tools for built-in types (web_search_preview, code_interpreter, file_search)
/// and looks up configured MCP servers to handle them.
///
/// # Arguments
/// * `mcp_orchestrator` - The MCP orchestrator with server configuration
/// * `tools` - Request tools to scan for built-in types
///
/// # Returns
/// Vector of routing information for built-in tools that have configured MCP servers.
/// Empty if no built-in tools are found or none have MCP server configurations.
pub fn collect_builtin_routing(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    tools: Option<&[ResponseTool]>,
) -> Vec<BuiltinToolRouting> {
    let Some(tools) = tools else {
        return Vec::new();
    };

    let mut routing = Vec::new();

    for tool in tools {
        let builtin_type = match tool.r#type {
            ResponseToolType::WebSearchPreview => BuiltinToolType::WebSearchPreview,
            ResponseToolType::CodeInterpreter => BuiltinToolType::CodeInterpreter,
            // FileSearch is not in ResponseToolType yet, but we handle it if added
            _ => continue,
        };

        if let Some((server_name, tool_name, response_format)) =
            mcp_orchestrator.find_builtin_server(builtin_type)
        {
            debug!(
                builtin_type = ?builtin_type,
                server = %server_name,
                tool = %tool_name,
                "Found MCP server for built-in tool type"
            );

            routing.push(BuiltinToolRouting {
                builtin_type,
                server_name,
                tool_name,
                response_format,
            });
        } else {
            warn!(
                builtin_type = %builtin_type,
                "Request includes built-in tool but no MCP server is configured for it"
            );
        }
    }

    routing
}

/// Extract builtin tool types from an OpenAI `ResponseTool` array.
///
/// Used by routers to determine which built-in tool types are present in
/// a request, for passing to [`ensure_mcp_servers`].
pub fn extract_builtin_types(tools: &[ResponseTool]) -> Vec<BuiltinToolType> {
    tools
        .iter()
        .filter_map(|t| match t.r#type {
            ResponseToolType::WebSearchPreview => Some(BuiltinToolType::WebSearchPreview),
            ResponseToolType::CodeInterpreter => Some(BuiltinToolType::CodeInterpreter),
            _ => None,
        })
        .collect()
}

/// Unified MCP server connection logic shared by all routers.
///
/// Connects dynamic/static MCP servers described by `inputs`, then adds
/// any static builtin servers for the given `builtin_types`.
///
/// Returns `Some(servers)` if at least one server is available, `None` otherwise.
pub async fn ensure_mcp_servers(
    orchestrator: &Arc<McpOrchestrator>,
    inputs: &[McpServerInput],
    builtin_types: &[BuiltinToolType],
    request_id: &str,
) -> Option<McpToolSession> {
    let mut mcp_servers = connect_mcp_servers(orchestrator, inputs).await;

    // Add builtin tool routing servers
    for &builtin_type in builtin_types {
        if let Some((server_name, tool_name, _)) = orchestrator.find_builtin_server(builtin_type) {
            debug!(
                builtin_type = ?builtin_type,
                server = %server_name,
                tool = %tool_name,
                "Adding static server for built-in tool routing"
            );
            if !mcp_servers.iter().any(|b| b.server_key == server_name) {
                mcp_servers.push(McpServerBinding {
                    label: server_name.clone(),
                    server_key: server_name,
                    allowed_tools: None,
                });
            }
        } else {
            warn!(
                builtin_type = %builtin_type,
                "Request includes built-in tool but no MCP server is configured for it"
            );
        }
    }

    if mcp_servers.is_empty() {
        None
    } else {
        Some(McpToolSession::new(
            Arc::clone(orchestrator),
            mcp_servers,
            request_id,
        ))
    }
}

/// Convenience wrapper for OpenAI Responses API routers.
///
/// Extracts MCP server inputs and builtin types from `ResponseTool` array,
/// then delegates to [`ensure_mcp_servers`].
pub async fn ensure_request_mcp_client(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    tools: &[ResponseTool],
    request_id: &str,
) -> Option<McpToolSession> {
    let inputs: Vec<McpServerInput> = tools
        .iter()
        .filter(|t| matches!(t.r#type, ResponseToolType::Mcp))
        .map(|tool| McpServerInput {
            label: tool
                .server_label
                .clone()
                .unwrap_or_else(|| "mcp".to_string()),
            url: tool.server_url.clone(),
            authorization: tool.authorization.clone(),
            headers: tool.headers.clone().unwrap_or_default(),
            allowed_tools: tool.allowed_tools.clone(),
        })
        .collect();

    let builtin_types = extract_builtin_types(tools);

    ensure_mcp_servers(mcp_orchestrator, &inputs, &builtin_types, request_id).await
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use openai_protocol::responses::ResponseTool;
    use smg_mcp::{McpConfig, ResponseFormatConfig, ToolConfig};

    use super::*;

    /// Create a test orchestrator with a built-in server configuration
    async fn create_test_orchestrator_with_builtin() -> Arc<McpOrchestrator> {
        let mut tools_config = HashMap::new();
        tools_config.insert(
            "web_search".to_string(),
            ToolConfig {
                response_format: ResponseFormatConfig::WebSearchCall,
                ..Default::default()
            },
        );

        let config = McpConfig {
            servers: vec![McpServerConfig {
                name: "search-server".to_string(),
                transport: McpTransport::Streamable {
                    url: "http://localhost:9999/mcp".to_string(),
                    token: None,
                    headers: HashMap::new(),
                },
                proxy: None,
                required: false,
                tools: Some(tools_config),
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("web_search".to_string()),
            }],
            pool: Default::default(),
            proxy: None,
            warmup: Vec::new(),
            inventory: Default::default(),
            policy: Default::default(),
        };

        // Note: This will fail to connect but still create the orchestrator with config
        Arc::new(McpOrchestrator::new(config).await.unwrap())
    }

    /// Create a test orchestrator without built-in server configuration
    async fn create_test_orchestrator_no_builtin() -> Arc<McpOrchestrator> {
        let config = McpConfig {
            servers: vec![],
            pool: Default::default(),
            proxy: None,
            warmup: Vec::new(),
            inventory: Default::default(),
            policy: Default::default(),
        };

        Arc::new(McpOrchestrator::new(config).await.unwrap())
    }

    #[tokio::test]
    async fn test_collect_builtin_routing_with_configured_server() {
        let orchestrator = create_test_orchestrator_with_builtin().await;

        let tools = vec![ResponseTool {
            r#type: ResponseToolType::WebSearchPreview,
            ..Default::default()
        }];

        let routing = collect_builtin_routing(&orchestrator, Some(&tools));

        assert_eq!(routing.len(), 1);
        assert_eq!(routing[0].builtin_type, BuiltinToolType::WebSearchPreview);
        assert_eq!(routing[0].server_name, "search-server");
        assert_eq!(routing[0].tool_name, "web_search");
        assert_eq!(routing[0].response_format, ResponseFormat::WebSearchCall);
    }

    #[tokio::test]
    async fn test_collect_builtin_routing_no_configured_server() {
        let orchestrator = create_test_orchestrator_no_builtin().await;

        let tools = vec![ResponseTool {
            r#type: ResponseToolType::WebSearchPreview,
            ..Default::default()
        }];

        let routing = collect_builtin_routing(&orchestrator, Some(&tools));

        // No routing because no server configured for this built-in type
        assert!(routing.is_empty());
    }

    #[tokio::test]
    async fn test_collect_builtin_routing_ignores_mcp_tools() {
        let orchestrator = create_test_orchestrator_with_builtin().await;

        let tools = vec![ResponseTool {
            r#type: ResponseToolType::Mcp,
            server_url: Some("http://example.com/mcp".to_string()),
            ..Default::default()
        }];

        let routing = collect_builtin_routing(&orchestrator, Some(&tools));

        // MCP tools are not built-in types, should be empty
        assert!(routing.is_empty());
    }

    #[tokio::test]
    async fn test_collect_builtin_routing_ignores_function_tools() {
        let orchestrator = create_test_orchestrator_with_builtin().await;

        let tools = vec![ResponseTool {
            r#type: ResponseToolType::Function,
            ..Default::default()
        }];

        let routing = collect_builtin_routing(&orchestrator, Some(&tools));

        // Function tools are not built-in types, should be empty
        assert!(routing.is_empty());
    }

    #[tokio::test]
    async fn test_collect_builtin_routing_none_tools() {
        let orchestrator = create_test_orchestrator_no_builtin().await;

        let routing = collect_builtin_routing(&orchestrator, None);

        assert!(routing.is_empty());
    }

    #[tokio::test]
    async fn test_collect_builtin_routing_multiple_builtin_tools() {
        // Create orchestrator with both web search and code interpreter
        let mut web_search_tools = HashMap::new();
        web_search_tools.insert(
            "web_search".to_string(),
            ToolConfig {
                response_format: ResponseFormatConfig::WebSearchCall,
                ..Default::default()
            },
        );

        let mut code_interp_tools = HashMap::new();
        code_interp_tools.insert(
            "run_code".to_string(),
            ToolConfig {
                response_format: ResponseFormatConfig::CodeInterpreterCall,
                ..Default::default()
            },
        );

        let config = McpConfig {
            servers: vec![
                McpServerConfig {
                    name: "search-server".to_string(),
                    transport: McpTransport::Streamable {
                        url: "http://localhost:9999/search".to_string(),
                        token: None,
                        headers: HashMap::new(),
                    },
                    proxy: None,
                    required: false,
                    tools: Some(web_search_tools),
                    builtin_type: Some(BuiltinToolType::WebSearchPreview),
                    builtin_tool_name: Some("web_search".to_string()),
                },
                McpServerConfig {
                    name: "code-server".to_string(),
                    transport: McpTransport::Streamable {
                        url: "http://localhost:9998/code".to_string(),
                        token: None,
                        headers: HashMap::new(),
                    },
                    proxy: None,
                    required: false,
                    tools: Some(code_interp_tools),
                    builtin_type: Some(BuiltinToolType::CodeInterpreter),
                    builtin_tool_name: Some("run_code".to_string()),
                },
            ],
            pool: Default::default(),
            proxy: None,
            warmup: Vec::new(),
            inventory: Default::default(),
            policy: Default::default(),
        };

        let orchestrator = Arc::new(McpOrchestrator::new(config).await.unwrap());

        let tools = vec![
            ResponseTool {
                r#type: ResponseToolType::WebSearchPreview,
                ..Default::default()
            },
            ResponseTool {
                r#type: ResponseToolType::CodeInterpreter,
                ..Default::default()
            },
        ];

        let routing = collect_builtin_routing(&orchestrator, Some(&tools));

        assert_eq!(routing.len(), 2);

        // Find web search routing
        let web_routing = routing
            .iter()
            .find(|r| r.builtin_type == BuiltinToolType::WebSearchPreview)
            .expect("Should have web search routing");
        assert_eq!(web_routing.server_name, "search-server");
        assert_eq!(web_routing.tool_name, "web_search");
        assert_eq!(web_routing.response_format, ResponseFormat::WebSearchCall);

        // Find code interpreter routing
        let code_routing = routing
            .iter()
            .find(|r| r.builtin_type == BuiltinToolType::CodeInterpreter)
            .expect("Should have code interpreter routing");
        assert_eq!(code_routing.server_name, "code-server");
        assert_eq!(code_routing.tool_name, "run_code");
        assert_eq!(
            code_routing.response_format,
            ResponseFormat::CodeInterpreterCall
        );
    }

    // =========================================================================
    // ensure_request_mcp_client tests
    // =========================================================================

    #[tokio::test]
    async fn test_ensure_request_mcp_client_with_builtin_routing() {
        // Create orchestrator with a built-in server configured
        let orchestrator = create_test_orchestrator_with_builtin().await;

        // Request has web_search_preview tool (no server_url, not MCP type)
        let tools = vec![ResponseTool {
            r#type: ResponseToolType::WebSearchPreview,
            ..Default::default()
        }];

        let result = ensure_request_mcp_client(&orchestrator, &tools, "test_req").await;

        // Should return Some because built-in routing is configured
        assert!(result.is_some());

        let session = result.unwrap();
        let all_servers = session.all_mcp_servers();
        assert_eq!(all_servers.len(), 1);

        // The server key should be the static server name
        assert_eq!(all_servers[0].label, "search-server");
        assert_eq!(all_servers[0].server_key, "search-server");
    }

    #[tokio::test]
    async fn test_ensure_request_mcp_client_no_builtin_routing() {
        // Create orchestrator WITHOUT built-in server configured
        let orchestrator = create_test_orchestrator_no_builtin().await;

        // Request has web_search_preview tool
        let tools = vec![ResponseTool {
            r#type: ResponseToolType::WebSearchPreview,
            ..Default::default()
        }];

        let result = ensure_request_mcp_client(&orchestrator, &tools, "test_req").await;

        // Should return None because no MCP or built-in routing is available
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_ensure_request_mcp_client_function_tools_only() {
        let orchestrator = create_test_orchestrator_with_builtin().await;

        // Request has only function tools (no MCP, no built-in)
        let tools = vec![ResponseTool {
            r#type: ResponseToolType::Function,
            ..Default::default()
        }];

        let result = ensure_request_mcp_client(&orchestrator, &tools, "test_req").await;

        // Should return None - function tools don't need MCP processing
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_ensure_request_mcp_client_mixed_tools() {
        // Create orchestrator with built-in server
        let orchestrator = create_test_orchestrator_with_builtin().await;

        // Request has mixed tools: function + web_search_preview
        let tools = vec![
            ResponseTool {
                r#type: ResponseToolType::Function,
                ..Default::default()
            },
            ResponseTool {
                r#type: ResponseToolType::WebSearchPreview,
                ..Default::default()
            },
        ];

        let result = ensure_request_mcp_client(&orchestrator, &tools, "test_req").await;

        // Should return Some because web_search_preview has built-in routing
        assert!(result.is_some());

        let session = result.unwrap();
        let all_servers = session.all_mcp_servers();
        assert_eq!(all_servers.len(), 1);
        assert_eq!(all_servers[0].label, "search-server");
    }
}
