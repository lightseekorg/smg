//! Shared MCP utilities for routers.

use std::sync::Arc;

use tracing::{debug, warn};

use crate::{
    mcp::{BuiltinToolType, McpOrchestrator, McpServerConfig, McpTransport, ResponseFormat},
    protocols::responses::{ResponseTool, ResponseToolType},
};

/// Default maximum tool loop iterations (safety limit).
pub const DEFAULT_MAX_ITERATIONS: usize = 10;

/// Configuration for MCP tool calling loops.
#[derive(Debug, Clone)]
pub struct McpLoopConfig {
    /// Maximum iterations (default: DEFAULT_MAX_ITERATIONS).
    pub max_iterations: usize,
    /// MCP servers for this request (label, server_key).
    pub mcp_servers: Vec<(String, String)>,
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

impl Default for McpLoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
            mcp_servers: Vec::new(),
        }
    }
}

/// Resolve the MCP server label for a tool name.
///
/// Uses orchestrator inventory to find the tool's server key, then maps it to the
/// request's MCP server label. Falls back to the first MCP server label (or "mcp").
pub fn resolve_tool_server_label(
    orchestrator: Option<&McpOrchestrator>,
    tool_name: &str,
    mcp_servers: &[(String, String)],
    server_keys: &[String],
) -> String {
    let fallback_label = mcp_servers
        .first()
        .map(|(label, _)| label.as_str())
        .unwrap_or("mcp");
    let Some(orchestrator) = orchestrator else {
        return fallback_label.to_string();
    };
    let Some(entry) = orchestrator.find_tool_by_name(tool_name, server_keys) else {
        return fallback_label.to_string();
    };
    let server_key = entry.qualified_name.server_key();
    mcp_servers
        .iter()
        .find(|(_, key)| key == server_key)
        .map(|(label, _)| label.clone())
        .unwrap_or_else(|| fallback_label.to_string())
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

/// Ensure MCP clients are connected for request-level MCP tools and built-in tool routing.
///
/// This function handles three cases:
/// 1. **Dynamic MCP tools**: Tools with `type: mcp` and `server_url` in the request.
///    These require connecting to the MCP server dynamically.
/// 2. **Static MCP tools**: Tools with `type: mcp` and `server_label` (but no URL).
///    These resolve to pre-configured static servers by name.
/// 3. **Built-in tool routing**: Tools like `web_search_preview` that have a static
///    MCP server configured via `builtin_type`. These use pre-connected static servers.
///
/// Headers for MCP servers come from the tool payload (`tool.headers`), not HTTP request headers.
///
/// Returns `Some((orchestrator, mcp_servers))` if MCP tools or built-in routing is available,
/// `None` otherwise.
pub async fn ensure_request_mcp_client(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    tools: &[ResponseTool],
) -> Option<(Arc<McpOrchestrator>, Vec<(String, String)>)> {
    let mut mcp_servers = Vec::new();

    // 1. Process explicit MCP tools (dynamic via `server_url`, or static via `server_label`)
    for tool in tools {
        if !matches!(tool.r#type, ResponseToolType::Mcp) {
            continue;
        }

        let label = tool
            .server_label
            .clone()
            .unwrap_or_else(|| "mcp".to_string());

        // Case A: Dynamic Server (Has `server_url`)
        if let Some(server_url) = tool
            .server_url
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

            let token = tool.authorization.clone();
            // Use headers from tool payload instead of HTTP request headers
            let headers = tool.headers.clone().unwrap_or_default();
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
                name: label.clone(),
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
                    if !mcp_servers.iter().any(|(_, key)| key == &server_key) {
                        mcp_servers.push((label.clone(), server_key));
                    }
                }
                Err(err) => {
                    warn!("Failed to connect MCP server {}: {}", server_key, err);
                }
            }
        }
        // Case B: Static Server (No `server_url`, but has `server_label`)
        else if let Some(label) = &tool.server_label {
            if !mcp_servers.iter().any(|(_, key)| key == label) {
                mcp_servers.push((label.clone(), label.clone()));
            }
        }
    }

    // 2. Process built-in tool routing (static servers configured with builtin_type)
    for routing in collect_builtin_routing(mcp_orchestrator, Some(tools)) {
        debug!(
            builtin_type = %routing.builtin_type,
            server = %routing.server_name,
            tool = %routing.tool_name,
            "Adding static server for built-in tool routing"
        );

        let server_name = routing.server_name;
        if !mcp_servers.iter().any(|(_, key)| key == &server_name) {
            mcp_servers.push((server_name.clone(), server_name));
        }
    }

    if mcp_servers.is_empty() {
        None
    } else {
        Some((mcp_orchestrator.clone(), mcp_servers))
    }
}

#[cfg(test)]
mod tests {
    use std::{borrow::Cow, collections::HashMap, sync::Arc};

    use serde_json::{Map, Value};

    use super::*;
    use crate::{
        mcp::{McpConfig, ResponseFormatConfig, Tool, ToolConfig, ToolEntry},
        protocols::responses::ResponseTool,
    };

    fn create_test_tool(name: &str) -> Tool {
        let schema_obj = serde_json::json!({
            "type": "object",
            "properties": {}
        });
        let schema_map = if let Value::Object(m) = schema_obj {
            m
        } else {
            Map::new()
        };

        Tool {
            name: Cow::Owned(name.to_string()),
            title: None,
            description: Some(Cow::Owned(format!("Test tool: {}", name))),
            input_schema: Arc::new(schema_map),
            output_schema: None,
            annotations: None,
            icons: None,
        }
    }

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

    #[test]
    fn test_resolve_tool_server_label_no_orchestrator_uses_first() {
        let mcp_servers = vec![
            ("brave".to_string(), "http://localhost:8001/sse".to_string()),
            (
                "deepwiki".to_string(),
                "https://mcp.deepwiki.com/mcp".to_string(),
            ),
        ];
        let server_keys: Vec<String> = mcp_servers.iter().map(|(_, key)| key.clone()).collect();

        let label = resolve_tool_server_label(None, "search", &mcp_servers, &server_keys);
        assert_eq!(label, "brave");
    }

    #[test]
    fn test_resolve_tool_server_label_empty_servers_falls_back_mcp() {
        let mcp_servers: Vec<(String, String)> = Vec::new();
        let server_keys: Vec<String> = Vec::new();

        let label = resolve_tool_server_label(None, "search", &mcp_servers, &server_keys);
        assert_eq!(label, "mcp");
    }

    #[tokio::test]
    async fn test_resolve_tool_server_label_missing_tool_fallbacks() {
        let orchestrator = create_test_orchestrator_no_builtin().await;
        let mcp_servers = vec![("brave".to_string(), "http://localhost:8001/sse".to_string())];
        let server_keys: Vec<String> = mcp_servers.iter().map(|(_, key)| key.clone()).collect();

        let label = resolve_tool_server_label(
            Some(orchestrator.as_ref()),
            "missing_tool",
            &mcp_servers,
            &server_keys,
        );
        assert_eq!(label, "brave");
    }

    #[tokio::test]
    async fn test_resolve_tool_server_label_happy_path() {
        let orchestrator = create_test_orchestrator_no_builtin().await;
        let server_key = "http://localhost:8001/sse".to_string();

        let tool = create_test_tool("search");
        let entry = ToolEntry::from_server_tool(&server_key, tool);
        orchestrator.tool_inventory().insert_entry(entry);

        let mcp_servers = vec![("brave".to_string(), server_key)];
        let server_keys: Vec<String> = mcp_servers.iter().map(|(_, key)| key.clone()).collect();
        let label = resolve_tool_server_label(
            Some(orchestrator.as_ref()),
            "search",
            &mcp_servers,
            &server_keys,
        );
        assert_eq!(label, "brave");
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

        let result = ensure_request_mcp_client(&orchestrator, &tools).await;

        // Should return Some because built-in routing is configured
        assert!(result.is_some());

        let (_, mcp_servers) = result.unwrap();
        assert_eq!(mcp_servers.len(), 1);

        // The server key should be the static server name
        let (label, key) = &mcp_servers[0];
        assert_eq!(label, "search-server");
        assert_eq!(key, "search-server");
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

        let result = ensure_request_mcp_client(&orchestrator, &tools).await;

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

        let result = ensure_request_mcp_client(&orchestrator, &tools).await;

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

        let result = ensure_request_mcp_client(&orchestrator, &tools).await;

        // Should return Some because web_search_preview has built-in routing
        assert!(result.is_some());

        let (_, mcp_servers) = result.unwrap();
        assert_eq!(mcp_servers.len(), 1);
        assert_eq!(mcp_servers[0].0, "search-server");
    }
}
