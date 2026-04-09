//! Shared MCP utilities for routers.

use std::{collections::HashMap, sync::Arc};

use axum::http::HeaderMap;
use openai_protocol::responses::ResponseTool;
use smg_mcp::{
    BuiltinToolType, McpOrchestrator, McpServerBinding, McpServerConfig, McpTransport,
    ResponseFormat,
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
    /// Built-in routing metadata to preserve response format for dynamic connections.
    pub builtin_type: Option<BuiltinToolType>,
    pub builtin_tool_name: Option<String>,
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
                builtin_type: input.builtin_type.clone(),
                builtin_tool_name: input.builtin_tool_name.clone(),
                internal: false,
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
/// When a built-in tool type (web_search_preview, code_interpreter, image_generation, file_search)
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
/// Scans request tools for built-in types (web_search_preview, code_interpreter, image_generation, file_search)
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
        let builtin_type = match tool {
            ResponseTool::WebSearchPreview(_) => BuiltinToolType::WebSearchPreview,
            ResponseTool::CodeInterpreter(_) => BuiltinToolType::CodeInterpreter,
            ResponseTool::ImageGeneration(_) => BuiltinToolType::ImageGeneration,
            _ => continue,
        };

        if let Some((server_name, tool_name, response_format, _)) =
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
        .filter_map(|t| match t {
            ResponseTool::WebSearchPreview(_) => Some(BuiltinToolType::WebSearchPreview),
            ResponseTool::CodeInterpreter(_) => Some(BuiltinToolType::CodeInterpreter),
            ResponseTool::ImageGeneration(_) => Some(BuiltinToolType::ImageGeneration),
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
) -> Option<Vec<McpServerBinding>> {
    let mut mcp_servers = connect_mcp_servers(orchestrator, inputs).await;

    // Add builtin tool routing servers
    for &builtin_type in builtin_types {
        if let Some((server_name, tool_name, _, _)) =
            orchestrator.find_builtin_server(builtin_type)
        {
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
        Some(mcp_servers)
    }
}

/// Convenience wrapper for OpenAI Responses API routers.
///
/// Extracts MCP server inputs and builtin types from `ResponseTool` array,
/// then delegates to [`ensure_mcp_servers`].
pub async fn ensure_request_mcp_client(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    tools: &[ResponseTool],
    request_headers: Option<&HeaderMap>,
) -> Option<Vec<McpServerBinding>> {
    let inputs: Vec<McpServerInput> = tools
        .iter()
        .filter_map(|tool| match tool {
            ResponseTool::Mcp(mcp) => Some(McpServerInput {
                label: mcp.server_label.clone(),
                url: mcp.server_url.clone(),
                authorization: mcp.authorization.clone(),
                headers: mcp.headers.clone().unwrap_or_default(),
                builtin_type: None,
                builtin_tool_name: None,
                allowed_tools: mcp.allowed_tools.clone(),
            }),
            _ => None,
        })
        .collect();

    let builtin_types = extract_builtin_types(tools);

    let mut inputs = inputs;
    // Built-in tools (e.g. image_generation/web_search_preview) are also routed
    // through MCP when configured in mcp.local.yaml, so we must add dynamic
    // inputs here to carry forwarded transport headers from the request.
    if !builtin_types.is_empty() {
        for builtin_type in &builtin_types {
            let Some((server_name, _, _, server_cfg)) =
                mcp_orchestrator.find_builtin_server(*builtin_type)
            else {
                continue;
            };

            if let Some((url, token, headers)) =
                resolve_forwarded_headers(&server_cfg, request_headers)
            {
                inputs.push(McpServerInput {
                    label: server_name,
                    url: Some(url),
                    authorization: token,
                    headers,
                    builtin_type: server_cfg.builtin_type.clone(),
                    builtin_tool_name: server_cfg.builtin_tool_name.clone(),
                    allowed_tools: None,
                });
            }
        }
    }

    ensure_mcp_servers(mcp_orchestrator, &inputs, &builtin_types).await
}

/// Resolve forwarded headers from a static MCP server config.
///
/// In `mcp.local.yaml` transport.headers, an empty value means:
/// "forward the same header from the incoming request".
///
/// Returns dynamic connection settings only when at least one forwarded-header
/// reference is resolved from request headers, and the transport is HTTP-based.
///
/// Behavior note:
/// - Empty configured header value (`""`) is treated as a per-request placeholder.
/// - Non-empty configured header value is copied as a static header value.
/// - Dynamic routing is enabled only if at least one placeholder header is
///   successfully resolved from `request_headers`; otherwise callers should
///   fall back to static server routing.
fn resolve_forwarded_headers(
    server_cfg: &McpServerConfig,
    request_headers: Option<&HeaderMap>,
) -> Option<(String, Option<String>, HashMap<String, String>)> {
    // Header forwarding only applies to HTTP transports (SSE/Streamable).
    // Stdio has no HTTP header channel, so it cannot participate here.
    let (url, token, configured_headers) = match &server_cfg.transport {
        McpTransport::Sse {
            url,
            token,
            headers,
        }
        | McpTransport::Streamable {
            url,
            token,
            headers,
        } => (url.clone(), token.clone(), headers),
        McpTransport::Stdio { .. } => return None,
    };

    let mut merged_headers = HashMap::new();
    let mut has_resolved_forward_ref = false;

    for (name, value) in configured_headers {
        if value.is_empty() {
            if let Some(header_value) = request_headers
                .and_then(|h| h.get(name.as_str()))
                .and_then(|v| v.to_str().ok())
            {
                merged_headers.insert(name.clone(), header_value.to_string());
                has_resolved_forward_ref = true;
            }
        } else {
            merged_headers.insert(name.clone(), value.clone());
        }
    }

    has_resolved_forward_ref.then_some((url, token, merged_headers))
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use axum::http::{HeaderMap, HeaderValue};
    use openai_protocol::{
        common::Function,
        responses::{
            CodeInterpreterTool, FunctionTool, ImageGenerationTool, McpTool, ResponseTool,
            WebSearchPreviewTool,
        },
    };
    use serde_json::json;
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
                internal: false,
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

        let tools = vec![ResponseTool::WebSearchPreview(
            WebSearchPreviewTool::default(),
        )];

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

        let tools = vec![ResponseTool::WebSearchPreview(
            WebSearchPreviewTool::default(),
        )];

        let routing = collect_builtin_routing(&orchestrator, Some(&tools));

        // No routing because no server configured for this built-in type
        assert!(routing.is_empty());
    }

    #[tokio::test]
    async fn test_collect_builtin_routing_ignores_mcp_tools() {
        let orchestrator = create_test_orchestrator_with_builtin().await;

        let tools = vec![ResponseTool::Mcp(McpTool {
            server_url: Some("http://example.com/mcp".to_string()),
            authorization: None,
            headers: None,
            server_label: "mcp".to_string(),
            server_description: None,
            require_approval: None,
            allowed_tools: None,
        })];

        let routing = collect_builtin_routing(&orchestrator, Some(&tools));

        // MCP tools are not built-in types, should be empty
        assert!(routing.is_empty());
    }

    #[tokio::test]
    async fn test_collect_builtin_routing_ignores_function_tools() {
        let orchestrator = create_test_orchestrator_with_builtin().await;

        let tools = vec![ResponseTool::Function(FunctionTool {
            function: Function {
                name: "dummy".to_string(),
                description: None,
                parameters: json!({}),
                strict: None,
            },
        })];

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
                    internal: false,
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
                    internal: false,
                },
                McpServerConfig {
                    name: "image-server".to_string(),
                    transport: McpTransport::Streamable {
                        url: "http://localhost:9997/image".to_string(),
                        token: None,
                        headers: HashMap::new(),
                    },
                    proxy: None,
                    required: false,
                    tools: Some(HashMap::from([(
                        "generate_image".to_string(),
                        ToolConfig {
                            response_format: ResponseFormatConfig::ImageGenerationCall,
                            ..Default::default()
                        },
                    )])),
                    builtin_type: Some(BuiltinToolType::ImageGeneration),
                    builtin_tool_name: Some("generate_image".to_string()),
                    internal: false,
                },
                McpServerConfig {
                    name: "image-server".to_string(),
                    transport: McpTransport::Streamable {
                        url: "http://localhost:9997/image".to_string(),
                        token: None,
                        headers: HashMap::new(),
                    },
                    proxy: None,
                    required: false,
                    tools: Some(HashMap::from([(
                        "generate_image".to_string(),
                        ToolConfig {
                            response_format: ResponseFormatConfig::ImageGenerationCall,
                            ..Default::default()
                        },
                    )])),
                    builtin_type: Some(BuiltinToolType::ImageGeneration),
                    builtin_tool_name: Some("generate_image".to_string()),
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
            ResponseTool::WebSearchPreview(WebSearchPreviewTool::default()),
            ResponseTool::CodeInterpreter(CodeInterpreterTool::default()),
            ResponseTool::ImageGeneration(ImageGenerationTool::default()),
        ];

        let routing = collect_builtin_routing(&orchestrator, Some(&tools));

        assert_eq!(routing.len(), 3);

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

        // Find image generation routing
        let image_routing = routing
            .iter()
            .find(|r| r.builtin_type == BuiltinToolType::ImageGeneration)
            .expect("Should have image generation routing");
        assert_eq!(image_routing.server_name, "image-server");
        assert_eq!(image_routing.tool_name, "generate_image");
        assert_eq!(
            image_routing.response_format,
            ResponseFormat::ImageGenerationCall
        );
    }

    #[test]
    fn test_extract_builtin_types_includes_image_generation() {
        let tools = vec![
            ResponseTool::WebSearchPreview(WebSearchPreviewTool::default()),
            ResponseTool::CodeInterpreter(CodeInterpreterTool::default()),
            ResponseTool::ImageGeneration(ImageGenerationTool::default()),
        ];

        let types = extract_builtin_types(&tools);
        assert!(types.contains(&BuiltinToolType::WebSearchPreview));
        assert!(types.contains(&BuiltinToolType::CodeInterpreter));
        assert!(types.contains(&BuiltinToolType::ImageGeneration));
    }

    // =========================================================================
    // ensure_request_mcp_client tests
    // =========================================================================

    #[tokio::test]
    async fn test_ensure_request_mcp_client_with_builtin_routing() {
        // Create orchestrator with a built-in server configured
        let orchestrator = create_test_orchestrator_with_builtin().await;

        // Request has web_search_preview tool (no server_url, not MCP type)
        let tools = vec![ResponseTool::WebSearchPreview(
            WebSearchPreviewTool::default(),
        )];

        let result = ensure_request_mcp_client(&orchestrator, &tools, None).await;

        // Should return Some because built-in routing is configured
        assert!(result.is_some());

        let mcp_servers = result.unwrap();
        assert_eq!(mcp_servers.len(), 1);

        // The server key should be the static server name
        assert_eq!(mcp_servers[0].label, "search-server");
        assert_eq!(mcp_servers[0].server_key, "search-server");
    }

    #[tokio::test]
    async fn test_ensure_request_mcp_client_no_builtin_routing() {
        // Create orchestrator WITHOUT built-in server configured
        let orchestrator = create_test_orchestrator_no_builtin().await;

        // Request has web_search_preview tool
        let tools = vec![ResponseTool::WebSearchPreview(
            WebSearchPreviewTool::default(),
        )];

        let result = ensure_request_mcp_client(&orchestrator, &tools, None).await;

        // Should return None because no MCP or built-in routing is available
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_ensure_request_mcp_client_function_tools_only() {
        let orchestrator = create_test_orchestrator_with_builtin().await;

        // Request has only function tools (no MCP, no built-in)
        let tools = vec![ResponseTool::Function(FunctionTool {
            function: Function {
                name: "dummy".to_string(),
                description: None,
                parameters: json!({}),
                strict: None,
            },
        })];

        let result = ensure_request_mcp_client(&orchestrator, &tools, None).await;

        // Should return None - function tools don't need MCP processing
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_ensure_request_mcp_client_mixed_tools() {
        // Create orchestrator with built-in server
        let orchestrator = create_test_orchestrator_with_builtin().await;

        // Request has mixed tools: function + web_search_preview
        let tools = vec![
            ResponseTool::Function(FunctionTool {
                function: Function {
                    name: "dummy".to_string(),
                    description: None,
                    parameters: json!({}),
                    strict: None,
                },
            }),
            ResponseTool::WebSearchPreview(WebSearchPreviewTool::default()),
        ];

        let result = ensure_request_mcp_client(&orchestrator, &tools, None).await;

        // Should return Some because web_search_preview has built-in routing
        assert!(result.is_some());

        let mcp_servers = result.unwrap();
        assert_eq!(mcp_servers.len(), 1);
        assert_eq!(mcp_servers[0].label, "search-server");
    }

    #[test]
    fn test_resolve_forwarded_headers_returns_none_when_no_forward_header_value_present() {
        let server_cfg = McpServerConfig {
            name: "image-server".to_string(),
            transport: McpTransport::Streamable {
                url: "http://localhost:9999/mcp".to_string(),
                token: None,
                headers: HashMap::from([
                    ("opc-compartment-id".to_string(), "".to_string()),
                    ("x-static".to_string(), "fixed".to_string()),
                ]),
            },
            proxy: None,
            required: false,
            tools: None,
            builtin_type: Some(BuiltinToolType::ImageGeneration),
            builtin_tool_name: Some("generate_image".to_string()),
        };

        let headers = HeaderMap::new();
        let resolved = resolve_forwarded_headers(&server_cfg, Some(&headers));
        assert!(resolved.is_none());
    }

    #[test]
    fn test_resolve_forwarded_headers_returns_some_when_forward_header_value_present() {
        let server_cfg = McpServerConfig {
            name: "image-server".to_string(),
            transport: McpTransport::Streamable {
                url: "http://localhost:9999/mcp".to_string(),
                token: None,
                headers: HashMap::from([
                    ("opc-compartment-id".to_string(), "".to_string()),
                    ("x-static".to_string(), "fixed".to_string()),
                ]),
            },
            proxy: None,
            required: false,
            tools: None,
            builtin_type: Some(BuiltinToolType::ImageGeneration),
            builtin_tool_name: Some("generate_image".to_string()),
        };

        let mut headers = HeaderMap::new();
        headers.insert(
            "opc-compartment-id",
            HeaderValue::from_str("ocid1.compartment.oc1..example").unwrap(),
        );

        let resolved = resolve_forwarded_headers(&server_cfg, Some(&headers));
        assert!(resolved.is_some());
        let (_, _, merged_headers) = resolved.unwrap();
        assert_eq!(
            merged_headers.get("opc-compartment-id").map(String::as_str),
            Some("ocid1.compartment.oc1..example")
        );
        assert_eq!(
            merged_headers.get("x-static").map(String::as_str),
            Some("fixed")
        );
    }
}
