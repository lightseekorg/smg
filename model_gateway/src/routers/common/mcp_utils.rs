//! Shared MCP utilities for routers.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use openai_protocol::responses::{McpAllowedTools, ResponseTool, ResponsesRequest};
use smg_mcp::{
    BuiltinToolType, McpOrchestrator, McpServerBinding, McpServerConfig, McpTransport,
    ResponseFormat,
};
use tracing::{debug, warn};

/// Default maximum tool loop iterations (safety limit).
pub const DEFAULT_MAX_ITERATIONS: usize = 10;

/// Project the T11 `McpAllowedTools` union into the flat name list consumed by
/// the router-side `McpServerInput` and `McpServerBinding` allowlist paths.
///
/// Mapping (documented on the calling sites):
///   * `None` → `None` (no constraint; all tools exposed)
///   * `Some(List(names))` → `Some(names)`
///   * `Some(Filter { tool_names: Some(v), read_only: None })` → `Some(v)`
///   * `Some(Filter { read_only: Some(_), .. })` → `Some(vec![])` —
///     fail-closed whenever the caller specified a `read_only` restriction,
///     regardless of `tool_names`. smg has no `readOnlyHint`-based filter
///     implementation yet, so honoring only the `tool_names` half would
///     silently broaden exposure past caller intent (e.g. `{read_only: true,
///     tool_names: ["mutating_tool"]}` must NOT expose `mutating_tool`).
///     Narrow to nothing so the downstream retain path exposes none.
///   * `Some(Filter { tool_names: None, read_only: None })` → `None`
pub(crate) fn project_allowed_tools(value: Option<&McpAllowedTools>) -> Option<Vec<String>> {
    value.and_then(|at| match at {
        McpAllowedTools::List(names) => Some(names.clone()),
        McpAllowedTools::Filter(filter) => match (&filter.tool_names, &filter.read_only) {
            // Any `read_only` restriction fails closed: readOnlyHint-based
            // filtering is unimplemented, so we cannot safely project a subset.
            (_, Some(_)) => Some(Vec::new()),
            (Some(names), None) => Some(names.clone()),
            (None, None) => None,
        },
    })
}

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
        let builtin_type = match tool {
            ResponseTool::WebSearchPreview(_) => BuiltinToolType::WebSearchPreview,
            ResponseTool::CodeInterpreter(_) => BuiltinToolType::CodeInterpreter,
            ResponseTool::ImageGeneration(_) => BuiltinToolType::ImageGeneration,
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
        .filter_map(|t| match t {
            ResponseTool::WebSearchPreview(_) => Some(BuiltinToolType::WebSearchPreview),
            ResponseTool::CodeInterpreter(_) => Some(BuiltinToolType::CodeInterpreter),
            ResponseTool::ImageGeneration(_) => Some(BuiltinToolType::ImageGeneration),
            _ => None,
        })
        .collect()
}

/// Collect user-declared function tool names from a Responses request.
pub(crate) fn collect_user_function_names(request: &ResponsesRequest) -> HashSet<String> {
    request
        .tools
        .as_deref()
        .unwrap_or_default()
        .iter()
        .filter_map(|tool| match tool {
            ResponseTool::Function(function_tool) => Some(function_tool.function.name.clone()),
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
        Some(mcp_servers)
    }
}

/// Convenience wrapper for request-level MCP resolution.
///
/// Explicit MCP tools are validated one server at a time so that any failure
/// aborts the request. Built-in tool routing still uses the existing
/// best-effort lookup.
///
/// Returns:
/// - `Ok(Some(...))` when MCP interception should run with the returned bindings.
/// - `Ok(None)` when no MCP interception is required and the request can continue upstream.
/// - `Err(message)` when an explicit MCP server could not be connected.
pub async fn ensure_request_mcp_client(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    tools: &[ResponseTool],
) -> Result<Option<Vec<McpServerBinding>>, String> {
    let mut mcp_servers: Vec<McpServerBinding> = Vec::new();

    for tool in tools {
        let ResponseTool::Mcp(mcp) = tool else {
            continue;
        };

        let input = McpServerInput {
            label: mcp.server_label.clone(),
            url: mcp.server_url.clone(),
            authorization: mcp.authorization.clone(),
            headers: mcp.headers.clone().unwrap_or_default(),
            // T11: project the `allowed_tools` union (List | Filter) into
            // the flat name list `McpServerInput` still expects. See
            // [`project_allowed_tools`] for the mapping table and the
            // rationale for the `read_only`-only fail-closed case.
            allowed_tools: project_allowed_tools(mcp.allowed_tools.as_ref()),
        };

        let connected = connect_mcp_servers(mcp_orchestrator, std::slice::from_ref(&input)).await;
        if connected.is_empty() {
            return Err(format!(
                "Error retrieving tool list from MCP server: '{}'. Check server_url and authorization.",
                mcp.server_label
            ));
        }

        for binding in connected {
            if !mcp_servers
                .iter()
                .any(|existing| existing.server_key == binding.server_key)
            {
                mcp_servers.push(binding);
            }
        }
    }

    let builtin_types = extract_builtin_types(tools);
    if let Some(builtin_servers) = ensure_mcp_servers(mcp_orchestrator, &[], &builtin_types).await
    {
        for binding in builtin_servers {
            if !mcp_servers
                .iter()
                .any(|existing| existing.server_key == binding.server_key)
            {
                mcp_servers.push(binding);
            }
        }
    }

    if mcp_servers.is_empty() {
        Ok(None)
    } else {
        Ok(Some(mcp_servers))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

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
            connector_id: None,
            defer_loading: None,
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

    #[tokio::test]
    async fn test_collect_builtin_routing_image_generation() {
        // Image generation is wired through the same hosted-tool MCP plumbing
        // as web_search / code_interpreter / file_search. This test proves
        // BuiltinToolType::ImageGeneration → ResponseFormat::ImageGenerationCall
        // so per-router wiring can rely on the infrastructure.
        let mut image_gen_tools = HashMap::new();
        image_gen_tools.insert(
            "generate_image".to_string(),
            ToolConfig {
                response_format: ResponseFormatConfig::ImageGenerationCall,
                ..Default::default()
            },
        );

        let config = McpConfig {
            servers: vec![McpServerConfig {
                name: "image-server".to_string(),
                transport: McpTransport::Streamable {
                    url: "http://localhost:9997/image".to_string(),
                    token: None,
                    headers: HashMap::new(),
                },
                proxy: None,
                required: false,
                tools: Some(image_gen_tools),
                builtin_type: Some(BuiltinToolType::ImageGeneration),
                builtin_tool_name: Some("generate_image".to_string()),
                internal: false,
            }],
            pool: Default::default(),
            proxy: None,
            warmup: Vec::new(),
            inventory: Default::default(),
            policy: Default::default(),
        };

        let orchestrator = Arc::new(McpOrchestrator::new(config).await.unwrap());

        let tools = vec![ResponseTool::ImageGeneration(ImageGenerationTool::default())];

        let routing = collect_builtin_routing(&orchestrator, Some(&tools));

        assert_eq!(routing.len(), 1);
        assert_eq!(routing[0].builtin_type, BuiltinToolType::ImageGeneration);
        assert_eq!(routing[0].server_name, "image-server");
        assert_eq!(routing[0].tool_name, "generate_image");
        assert_eq!(
            routing[0].response_format,
            ResponseFormat::ImageGenerationCall
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
        let tools = vec![ResponseTool::WebSearchPreview(
            WebSearchPreviewTool::default(),
        )];

        let result = ensure_request_mcp_client(&orchestrator, &tools).await;

        // Should return Some because built-in routing is configured
        assert!(result.is_ok());

        let mcp_servers = result.unwrap().unwrap();
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

        let result = ensure_request_mcp_client(&orchestrator, &tools).await;

        // Should return None because no MCP or built-in routing is available
        assert!(matches!(result, Ok(None)));
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

        let result = ensure_request_mcp_client(&orchestrator, &tools).await;

        // Should return None - function tools don't need MCP processing
        assert!(matches!(result, Ok(None)));
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

        let result = ensure_request_mcp_client(&orchestrator, &tools).await;

        // Should return Some because web_search_preview has built-in routing
        assert!(result.is_ok());

        let mcp_servers = result.unwrap().unwrap();
        assert_eq!(mcp_servers.len(), 1);
        assert_eq!(mcp_servers[0].label, "search-server");
    }

    // ---- T11 projection ---------------------------------------------------

    use openai_protocol::responses::McpToolFilter;

    #[test]
    fn test_project_allowed_tools_none() {
        assert_eq!(project_allowed_tools(None), None);
    }

    #[test]
    fn test_project_allowed_tools_list_variant() {
        let value = McpAllowedTools::List(vec!["a".to_string(), "b".to_string()]);
        assert_eq!(
            project_allowed_tools(Some(&value)),
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn test_project_allowed_tools_filter_with_tool_names() {
        let value = McpAllowedTools::Filter(McpToolFilter {
            read_only: None,
            tool_names: Some(vec!["x".to_string()]),
        });
        assert_eq!(
            project_allowed_tools(Some(&value)),
            Some(vec!["x".to_string()])
        );
    }

    /// `Filter { read_only: Some(true) }` with no `tool_names` must project
    /// to `Some(vec![])` (fail-closed) so downstream does not expose the full
    /// tool surface when the caller explicitly asked to restrict. See
    /// [`project_allowed_tools`] rationale.
    #[test]
    fn test_project_allowed_tools_filter_read_only_only_is_fail_closed() {
        let value = McpAllowedTools::Filter(McpToolFilter {
            read_only: Some(true),
            tool_names: None,
        });
        assert_eq!(project_allowed_tools(Some(&value)), Some(Vec::new()));
    }

    /// `Filter { read_only: None, tool_names: None }` — both sub-fields
    /// absent — is indistinguishable from an absent allowlist and projects
    /// to `None` so the downstream retain path treats it as unconstrained.
    #[test]
    fn test_project_allowed_tools_filter_empty_is_unconstrained() {
        let value = McpAllowedTools::Filter(McpToolFilter::default());
        assert_eq!(project_allowed_tools(Some(&value)), None);
    }

    /// `Filter { read_only: Some(_), tool_names: Some(_) }` — both set — must
    /// still fail-closed. Any `read_only` restriction disables the normal
    /// name-list projection because `readOnlyHint`-based filtering is
    /// unimplemented; honoring only the `tool_names` half would broaden
    /// exposure past caller intent (e.g. `tool_names: ["mutating_tool"]` with
    /// `read_only: true` must not expose `mutating_tool`).
    #[test]
    fn test_project_allowed_tools_filter_read_only_plus_names_is_fail_closed() {
        let value = McpAllowedTools::Filter(McpToolFilter {
            read_only: Some(true),
            tool_names: Some(vec!["mutating_tool".to_string()]),
        });
        assert_eq!(project_allowed_tools(Some(&value)), Some(Vec::new()));
    }

    #[tokio::test]
    async fn test_ensure_request_mcp_client_fails_when_any_explicit_mcp_fails() {
        let orchestrator = create_test_orchestrator_no_builtin().await;

        let tools = vec![
            ResponseTool::Mcp(McpTool {
                server_url: None,
                authorization: None,
                headers: None,
                server_label: "static-ok".to_string(),
                server_description: None,
                require_approval: None,
                allowed_tools: None,
            }),
            ResponseTool::Mcp(McpTool {
                server_url: Some("http://127.0.0.1:1/mcp".to_string()),
                authorization: None,
                headers: None,
                server_label: "broken".to_string(),
                server_description: None,
                require_approval: None,
                allowed_tools: None,
            }),
        ];

        let result = ensure_request_mcp_client(&orchestrator, &tools).await;

        assert!(matches!(
            result,
            Err(ref message)
                if message
                    == "Error retrieving tool list from MCP server: 'broken'. Check server_url and authorization."
        ));
    }
}
