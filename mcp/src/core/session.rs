//! MCP Tool Session — bundles all MCP execution state for a single request.
//!
//! Instead of threading `orchestrator`, `request_ctx`, `mcp_servers`,
//! and `mcp_tools` through every function, callers create one `McpToolSession` and
//! pass `&session` everywhere. When an MCP parameter changes (e.g. `mcp_servers`
//! representation), only this struct and its constructor need updating — not every
//! router function signature.

use std::collections::{HashMap, HashSet};

use openai_protocol::responses::ResponseTool;

use super::orchestrator::{
    McpOrchestrator, McpRequestContext, ToolExecutionInput, ToolExecutionOutput,
};
use crate::{
    approval::ApprovalMode,
    inventory::{QualifiedToolName, ToolEntry},
    responses_bridge::{
        build_chat_function_tools_with_names, build_function_tools_json_with_names,
        build_mcp_list_tools_item, build_mcp_list_tools_json, build_response_tools_with_names,
    },
    tenant::TenantContext,
    transform::ResponseFormat,
};

/// Named pair of `(label, server_key)` for a connected MCP server.
///
/// Replaces the opaque `(String, String)` tuple that was threaded through
/// ~20 call sites, improving readability and preventing field-swap bugs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServerBinding {
    /// User-facing label (e.g. the `server_label` from the request).
    pub label: String,
    /// Internal key used to look up the server in the orchestrator.
    pub server_key: String,
    /// Optional per-server tool allowlist.
    ///
    /// When `Some`, only the listed tool names are exposed for this server.
    /// When `None`, all tools from the server are exposed.
    pub allowed_tools: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
struct ExposedToolBinding {
    server_key: String,
    server_label: String,
    resolved_tool_name: String,
    response_format: ResponseFormat,
}

/// Bundles all MCP execution state for a single request.
///
/// Created once per request, then passed by reference to every function
/// that needs MCP infrastructure. This eliminates repeated parameter
/// threading of `orchestrator`, `request_ctx`, `mcp_servers`,
/// and `mcp_tools`.
pub struct McpToolSession<'a> {
    orchestrator: &'a McpOrchestrator,
    request_ctx: McpRequestContext<'a>,
    /// All MCP servers in this session (including builtin).
    all_mcp_servers: Vec<McpServerBinding>,
    /// Non-builtin MCP servers only — used for `mcp_list_tools` output.
    mcp_servers: Vec<McpServerBinding>,
    mcp_tools: Vec<ToolEntry>,
    exposed_name_map: HashMap<String, ExposedToolBinding>,
    exposed_name_by_qualified: HashMap<QualifiedToolName, String>,
}

impl<'a> McpToolSession<'a> {
    /// Create a new session by performing the setup every path currently repeats:
    /// 1. Create request context with default tenant and policy-only approval
    /// 2. List tools for the selected servers
    /// 3. Apply per-server allowed_tools filtering from bindings
    pub fn new(
        orchestrator: &'a McpOrchestrator,
        mcp_servers: Vec<McpServerBinding>,
        request_id: impl Into<String>,
    ) -> Self {
        let request_ctx = orchestrator.create_request_context(
            request_id,
            TenantContext::default(),
            ApprovalMode::PolicyOnly,
        );
        let server_keys: Vec<String> = mcp_servers.iter().map(|b| b.server_key.clone()).collect();
        let mut mcp_tools = orchestrator.list_tools_for_servers(&server_keys);

        // Build per-server allowlists from bindings that specify allowed_tools.
        let allowed_tools_by_server_key: HashMap<&str, HashSet<&str>> = mcp_servers
            .iter()
            .filter_map(|b| {
                b.allowed_tools.as_ref().map(|tools| {
                    let set: HashSet<&str> = tools
                        .iter()
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .collect();
                    (b.server_key.as_str(), set)
                })
            })
            .collect();

        if !allowed_tools_by_server_key.is_empty() {
            mcp_tools.retain(
                |entry| match allowed_tools_by_server_key.get(entry.server_key()) {
                    None => true,
                    Some(allowed) => allowed.contains(entry.tool_name()),
                },
            );
        }
        let (exposed_name_map, exposed_name_by_qualified) =
            Self::build_exposed_function_tools(&mcp_tools, &mcp_servers);

        // Filter out servers configured with builtin_type from the visible list.
        let builtin_names = orchestrator.builtin_server_names();
        let visible_mcp_servers: Vec<McpServerBinding> = mcp_servers
            .iter()
            .filter(|b| !builtin_names.contains(&b.server_key))
            .cloned()
            .collect();

        Self {
            orchestrator,
            request_ctx,
            all_mcp_servers: mcp_servers,
            mcp_servers: visible_mcp_servers,
            mcp_tools,
            exposed_name_map,
            exposed_name_by_qualified,
        }
    }

    // --- Accessors ---

    pub fn orchestrator(&self) -> &McpOrchestrator {
        self.orchestrator
    }

    pub fn request_ctx(&self) -> &McpRequestContext<'a> {
        &self.request_ctx
    }

    /// Returns only non-builtin MCP servers
    pub fn mcp_servers(&self) -> &[McpServerBinding] {
        &self.mcp_servers
    }

    /// Returns all MCP servers including builtin ones.
    pub fn all_mcp_servers(&self) -> &[McpServerBinding] {
        &self.all_mcp_servers
    }

    pub fn mcp_tools(&self) -> &[ToolEntry] {
        &self.mcp_tools
    }

    /// Returns true if the name is exposed to the model for this session.
    pub fn has_exposed_tool(&self, tool_name: &str) -> bool {
        self.exposed_name_map.contains_key(tool_name)
    }

    /// Returns the session's qualified-name -> exposed-name mapping.
    ///
    /// Router adapters should use this with response bridge builders.
    pub fn exposed_name_by_qualified(&self) -> &HashMap<QualifiedToolName, String> {
        &self.exposed_name_by_qualified
    }

    // --- Delegation methods ---

    /// Execute multiple tools using this session's exposed-name mapping.
    pub async fn execute_tools(&self, inputs: Vec<ToolExecutionInput>) -> Vec<ToolExecutionOutput> {
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            outputs.push(self.execute_tool(input).await);
        }
        outputs
    }

    /// Execute a single tool using this session's exposed-name mapping.
    pub async fn execute_tool(&self, input: ToolExecutionInput) -> ToolExecutionOutput {
        let invoked_name = input.tool_name.clone();

        if let Some(binding) = self.exposed_name_map.get(&invoked_name) {
            let resolved_tool_name = binding.resolved_tool_name.clone();
            let mut output = self
                .orchestrator
                .execute_tool_resolved(
                    ToolExecutionInput {
                        call_id: input.call_id,
                        tool_name: resolved_tool_name.clone(),
                        arguments: input.arguments,
                    },
                    &binding.server_key,
                    &binding.server_label,
                    &self.request_ctx,
                )
                .await;

            output.invoked_tool_name = Some(invoked_name.clone());
            output.resolved_tool_name = Some(resolved_tool_name);
            output.tool_name = invoked_name;
            output
        } else {
            let fallback_label = self
                .all_mcp_servers
                .first()
                .map(|b| b.label.clone())
                .unwrap_or_else(|| "mcp".to_string());
            let err = format!("Tool '{invoked_name}' is not in this session's exposed tool map");
            ToolExecutionOutput {
                call_id: input.call_id,
                tool_name: invoked_name.clone(),
                invoked_tool_name: Some(invoked_name),
                resolved_tool_name: None,
                server_key: "unknown".to_string(),
                server_label: fallback_label,
                arguments_str: input.arguments.to_string(),
                output: serde_json::json!({ "error": &err }),
                is_error: true,
                error_message: Some(err),
                response_format: ResponseFormat::Passthrough,
                duration: std::time::Duration::default(),
            }
        }
    }

    /// Resolve the user-facing server label for a tool.
    ///
    /// Uses the orchestrator inventory to find the tool's server key, then maps
    /// it to the request's MCP server label. Falls back to the first server
    /// label (or "mcp").
    pub fn resolve_tool_server_label(&self, tool_name: &str) -> String {
        let fallback_label = self
            .all_mcp_servers
            .first()
            .map(|b| b.label.as_str())
            .unwrap_or("mcp");

        self.exposed_name_map
            .get(tool_name)
            .map(|binding| binding.server_label.clone())
            .unwrap_or_else(|| fallback_label.to_string())
    }

    /// List tools for a single server key.
    ///
    /// Useful for emitting per-server `mcp_list_tools` items.
    pub fn list_tools_for_server(&self, server_key: &str) -> Vec<ToolEntry> {
        // Use the session's pre-filtered tool snapshot for consistency.
        self.mcp_tools
            .iter()
            .filter(|entry| entry.server_key() == server_key)
            .cloned()
            .collect()
    }

    /// Look up the response format for a tool.
    ///
    /// Convenience method that returns `Passthrough` if the tool is not found.
    pub fn tool_response_format(&self, tool_name: &str) -> ResponseFormat {
        self.exposed_name_map
            .get(tool_name)
            .map(|binding| binding.response_format.clone())
            .unwrap_or(ResponseFormat::Passthrough)
    }

    /// Build function-tool JSON payloads for upstream model calls.
    pub fn build_function_tools_json(&self) -> Vec<serde_json::Value> {
        build_function_tools_json_with_names(&self.mcp_tools, Some(&self.exposed_name_by_qualified))
    }

    /// Build Chat API `Tool` structs for chat completions.
    pub fn build_chat_function_tools(&self) -> Vec<openai_protocol::common::Tool> {
        build_chat_function_tools_with_names(&self.mcp_tools, Some(&self.exposed_name_by_qualified))
    }

    /// Build Responses API `ResponseTool` structs.
    pub fn build_response_tools(&self) -> Vec<ResponseTool> {
        build_response_tools_with_names(&self.mcp_tools, Some(&self.exposed_name_by_qualified))
    }

    /// Build `mcp_list_tools` JSON for a specific server.
    pub fn build_mcp_list_tools_json(
        &self,
        server_label: &str,
        server_key: &str,
    ) -> serde_json::Value {
        let tools = self.list_tools_for_server(server_key);
        build_mcp_list_tools_json(server_label, &tools)
    }

    /// Build typed `mcp_list_tools` output item for a specific server.
    pub fn build_mcp_list_tools_item(
        &self,
        server_label: &str,
        server_key: &str,
    ) -> openai_protocol::responses::ResponseOutputItem {
        let tools = self.list_tools_for_server(server_key);
        build_mcp_list_tools_item(server_label, &tools)
    }

    /// Inject MCP metadata into a response output array.
    ///
    /// Standardized ordering:
    /// 1. `mcp_list_tools` items (one per server) — prepended
    /// 2. `tool_call_items` (mcp_call / web_search_call / etc.) — after list_tools
    /// 3. Existing items (messages, etc.) — remain at end
    pub fn inject_mcp_output_items(
        &self,
        output: &mut Vec<openai_protocol::responses::ResponseOutputItem>,
        tool_call_items: Vec<openai_protocol::responses::ResponseOutputItem>,
    ) {
        let num_servers = self.mcp_servers.len();

        // 1. Prepend mcp_list_tools for each non-builtin server
        for binding in self.mcp_servers.iter().rev() {
            output.insert(
                0,
                self.build_mcp_list_tools_item(&binding.label, &binding.server_key),
            );
        }

        // 2. Insert tool call items right after mcp_list_tools
        let mut insert_pos = num_servers;
        for item in tool_call_items {
            output.insert(insert_pos, item);
            insert_pos += 1;
        }
    }

    fn build_exposed_function_tools(
        tools: &[ToolEntry],
        mcp_servers: &[McpServerBinding],
    ) -> (
        HashMap<String, ExposedToolBinding>,
        HashMap<QualifiedToolName, String>,
    ) {
        let server_labels: HashMap<&str, &str> = mcp_servers
            .iter()
            .map(|b| (b.server_key.as_str(), b.label.as_str()))
            .collect();

        let mut name_counts: HashMap<&str, usize> = HashMap::new();
        for entry in tools {
            *name_counts.entry(entry.tool_name()).or_insert(0) += 1;
        }

        let mut used_exposed_names: HashSet<String> = HashSet::with_capacity(tools.len());
        let mut name_suffixes: HashMap<String, usize> = HashMap::with_capacity(tools.len());
        let mut exposed_name_map: HashMap<String, ExposedToolBinding> =
            HashMap::with_capacity(tools.len());
        let mut exposed_name_by_qualified: HashMap<QualifiedToolName, String> =
            HashMap::with_capacity(tools.len());

        for entry in tools {
            let server_key = entry.server_key().to_string();
            let server_label = server_labels
                .get(server_key.as_str())
                .copied()
                .unwrap_or(server_key.as_str())
                .to_string();
            let resolved_tool_name = entry.tool_name().to_string();

            let base_exposed_name = if name_counts.get(entry.tool_name()).copied().unwrap_or(0) <= 1
            {
                resolved_tool_name.clone()
            } else {
                format!(
                    "mcp_{}_{}",
                    sanitize_tool_token(&server_label),
                    sanitize_tool_token(&resolved_tool_name)
                )
            };

            let suffix = name_suffixes.entry(base_exposed_name.clone()).or_insert(0);
            let mut exposed_name = if *suffix == 0 {
                base_exposed_name.clone()
            } else {
                format!("{base_exposed_name}_{suffix}")
            };
            while used_exposed_names.contains(&exposed_name) {
                *suffix += 1;
                exposed_name = format!("{base_exposed_name}_{suffix}");
            }
            used_exposed_names.insert(exposed_name.clone());

            exposed_name_by_qualified.insert(entry.qualified_name.clone(), exposed_name.clone());

            exposed_name_map.insert(
                exposed_name,
                ExposedToolBinding {
                    server_key,
                    server_label,
                    resolved_tool_name,
                    response_format: entry.response_format.clone(),
                },
            );
        }

        (exposed_name_map, exposed_name_by_qualified)
    }
}

fn sanitize_tool_token(input: &str) -> String {
    let mut out = String::with_capacity(input.len().max(1));
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    let out = out.trim_matches('_');
    if out.is_empty() {
        "tool".to_string()
    } else {
        out.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation_keeps_servers() {
        let orchestrator = McpOrchestrator::new_test();
        let mcp_servers = vec![
            McpServerBinding {
                label: "label1".to_string(),
                server_key: "key1".to_string(),
                allowed_tools: None,
            },
            McpServerBinding {
                label: "label2".to_string(),
                server_key: "key2".to_string(),
                allowed_tools: None,
            },
        ];

        let session = McpToolSession::new(&orchestrator, mcp_servers, "test-request");

        assert_eq!(session.mcp_servers().len(), 2);
        assert_eq!(session.mcp_servers()[0].label, "label1");
        assert_eq!(session.mcp_servers()[0].server_key, "key1");
    }

    #[test]
    fn test_session_empty_servers() {
        let orchestrator = McpOrchestrator::new_test();
        let session = McpToolSession::new(&orchestrator, vec![], "test-request");

        assert!(session.mcp_servers().is_empty());
        assert!(session.mcp_tools().is_empty());
    }

    #[test]
    fn test_resolve_tool_server_label_fallback() {
        let orchestrator = McpOrchestrator::new_test();
        let mcp_servers = vec![McpServerBinding {
            label: "my_label".to_string(),
            server_key: "my_key".to_string(),
            allowed_tools: None,
        }];
        let session = McpToolSession::new(&orchestrator, mcp_servers, "test-request");

        // Tool doesn't exist, should fall back to first label
        let label = session.resolve_tool_server_label("nonexistent_tool");
        assert_eq!(label, "my_label");
    }

    #[test]
    fn test_resolve_tool_server_label_no_servers() {
        let orchestrator = McpOrchestrator::new_test();
        let session = McpToolSession::new(&orchestrator, vec![], "test-request");

        // No servers, should fall back to "mcp"
        let label = session.resolve_tool_server_label("nonexistent_tool");
        assert_eq!(label, "mcp");
    }

    #[test]
    fn test_tool_response_format_default() {
        let orchestrator = McpOrchestrator::new_test();
        let session = McpToolSession::new(&orchestrator, vec![], "test-request");

        let format = session.tool_response_format("nonexistent");
        assert!(matches!(format, ResponseFormat::Passthrough));
    }

    fn create_test_tool(name: &str) -> crate::core::config::Tool {
        use std::{borrow::Cow, sync::Arc};

        crate::core::config::Tool {
            name: Cow::Owned(name.to_string()),
            title: None,
            description: Some(Cow::Owned(format!("Test tool: {name}"))),
            input_schema: Arc::new(serde_json::Map::new()),
            output_schema: None,
            annotations: None,
            icons: None,
        }
    }

    #[test]
    fn test_has_exposed_tool_with_inventory() {
        let orchestrator = McpOrchestrator::new_test();

        // Insert a tool into the inventory
        let tool = create_test_tool("test_tool");
        let entry = ToolEntry::from_server_tool("server1", tool);
        orchestrator.tool_inventory().insert_entry(entry);

        let mcp_servers = vec![McpServerBinding {
            label: "label1".to_string(),
            server_key: "server1".to_string(),
            allowed_tools: None,
        }];
        let session = McpToolSession::new(&orchestrator, mcp_servers, "test-request");

        assert!(session.has_exposed_tool("test_tool"));
        assert_eq!(session.mcp_tools().len(), 1);
    }

    #[test]
    fn test_resolve_label_with_inventory() {
        let orchestrator = McpOrchestrator::new_test();

        let tool = create_test_tool("test_tool");
        let entry = ToolEntry::from_server_tool("server1", tool);
        orchestrator.tool_inventory().insert_entry(entry);

        let mcp_servers = vec![McpServerBinding {
            label: "my_server".to_string(),
            server_key: "server1".to_string(),
            allowed_tools: None,
        }];
        let session = McpToolSession::new(&orchestrator, mcp_servers, "test-request");

        let label = session.resolve_tool_server_label("test_tool");
        assert_eq!(label, "my_server");
    }

    #[test]
    fn test_exposed_names_are_unique_for_tool_name_collisions() {
        let orchestrator = McpOrchestrator::new_test();

        let tool_a = create_test_tool("shared_tool");
        let tool_b = create_test_tool("shared_tool");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool("server1", tool_a));
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool("server2", tool_b));

        let session = McpToolSession::new(
            &orchestrator,
            vec![
                McpServerBinding {
                    label: "alpha".to_string(),
                    server_key: "server1".to_string(),
                    allowed_tools: None,
                },
                McpServerBinding {
                    label: "beta".to_string(),
                    server_key: "server2".to_string(),
                    allowed_tools: None,
                },
            ],
            "test-request",
        );

        let name_a = session
            .exposed_name_by_qualified()
            .get(&QualifiedToolName::new("server1", "shared_tool"))
            .cloned()
            .expect("missing exposed name for server1 tool");
        let name_b = session
            .exposed_name_by_qualified()
            .get(&QualifiedToolName::new("server2", "shared_tool"))
            .cloned()
            .expect("missing exposed name for server2 tool");

        assert_ne!(name_a, name_b);
        assert_ne!(name_a, "shared_tool");
        assert_ne!(name_b, "shared_tool");
        assert!(session.has_exposed_tool(&name_a));
        assert!(session.has_exposed_tool(&name_b));
    }

    #[test]
    fn test_exposed_names_handle_pre_suffixed_name_conflicts() {
        let orchestrator = McpOrchestrator::new_test();

        let tool_base = create_test_tool("foo");
        let tool_suffixed = create_test_tool("foo_1");
        let tool_dup = create_test_tool("foo");

        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool("s1", tool_base));
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool("s2", tool_suffixed));
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool("s3", tool_dup));

        let session = McpToolSession::new(
            &orchestrator,
            vec![
                McpServerBinding {
                    label: "a".to_string(),
                    server_key: "s1".to_string(),
                    allowed_tools: None,
                },
                McpServerBinding {
                    label: "b".to_string(),
                    server_key: "s2".to_string(),
                    allowed_tools: None,
                },
                McpServerBinding {
                    label: "c".to_string(),
                    server_key: "s3".to_string(),
                    allowed_tools: None,
                },
            ],
            "test-request",
        );

        let exposed_names: HashSet<String> = session
            .exposed_name_by_qualified()
            .values()
            .cloned()
            .collect();
        assert_eq!(exposed_names.len(), 3);
    }

    // --- Builtin server filtering tests ---

    fn create_builtin_orchestrator() -> McpOrchestrator {
        use crate::core::config::{BuiltinToolType, McpConfig, McpServerConfig, McpTransport};

        let config = McpConfig {
            servers: vec![
                McpServerConfig {
                    name: "brave-builtin".to_string(),
                    transport: McpTransport::Sse {
                        url: "http://localhost:8001/sse".to_string(),
                        token: None,
                        headers: HashMap::new(),
                    },
                    proxy: None,
                    required: false,
                    tools: None,
                    builtin_type: Some(BuiltinToolType::WebSearchPreview),
                    builtin_tool_name: Some("brave_web_search".to_string()),
                },
                McpServerConfig {
                    name: "regular-server".to_string(),
                    transport: McpTransport::Sse {
                        url: "http://localhost:3000/sse".to_string(),
                        token: None,
                        headers: HashMap::new(),
                    },
                    proxy: None,
                    required: false,
                    tools: None,
                    builtin_type: None,
                    builtin_tool_name: None,
                },
            ],
            ..Default::default()
        };

        McpOrchestrator::new_test_with_config(config)
    }

    #[test]
    fn test_mcp_servers_filters_builtin() {
        let orchestrator = create_builtin_orchestrator();
        let mcp_servers = vec![
            McpServerBinding {
                label: "brave".to_string(),
                server_key: "brave-builtin".to_string(),
                allowed_tools: None,
            },
            McpServerBinding {
                label: "regular".to_string(),
                server_key: "regular-server".to_string(),
                allowed_tools: None,
            },
        ];

        let session = McpToolSession::new(&orchestrator, mcp_servers, "test-request");

        // mcp_servers() should only return non-builtin servers
        let visible = session.mcp_servers();
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].label, "regular");
        assert_eq!(visible[0].server_key, "regular-server");

        // all_mcp_servers() should return everything
        assert_eq!(session.all_mcp_servers().len(), 2);
    }

    #[test]
    fn test_allowed_tools_filters_inventory_and_list_tools() {
        let orchestrator = McpOrchestrator::new_test();

        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "server1",
                create_test_tool("brave_web_search"),
            ));
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "server1",
                create_test_tool("brave_local_search"),
            ));

        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "mock".to_string(),
                server_key: "server1".to_string(),
                allowed_tools: Some(vec!["brave_web_search".to_string()]),
            }],
            "test-request",
        );

        assert!(session.has_exposed_tool("brave_web_search"));
        assert!(!session.has_exposed_tool("brave_local_search"));
        assert_eq!(session.mcp_tools().len(), 1);

        let listed = session.list_tools_for_server("server1");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].tool_name(), "brave_web_search");
    }

    #[test]
    fn test_allowed_tools_filters_only_target_server() {
        let orchestrator = McpOrchestrator::new_test();

        // server1 has two tools
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "server1",
                create_test_tool("brave_web_search"),
            ));
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "server1",
                create_test_tool("brave_local_search"),
            ));

        // server2 has two tools
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "server2",
                create_test_tool("deepwiki_search"),
            ));
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "server2",
                create_test_tool("deepwiki_read"),
            ));

        let session = McpToolSession::new(
            &orchestrator,
            vec![
                McpServerBinding {
                    label: "brave".to_string(),
                    server_key: "server1".to_string(),
                    allowed_tools: Some(vec!["brave_web_search".to_string()]),
                },
                McpServerBinding {
                    label: "deepwiki".to_string(),
                    server_key: "server2".to_string(),
                    allowed_tools: None,
                },
            ],
            "test-request",
        );

        // server1 is filtered
        assert!(session.has_exposed_tool("brave_web_search"));
        assert!(!session.has_exposed_tool("brave_local_search"));
        let listed_server1 = session.list_tools_for_server("server1");
        assert_eq!(listed_server1.len(), 1);
        assert_eq!(listed_server1[0].tool_name(), "brave_web_search");

        // server2 is unfiltered
        assert!(session.has_exposed_tool("deepwiki_search"));
        assert!(session.has_exposed_tool("deepwiki_read"));
        let listed_server2 = session.list_tools_for_server("server2");
        assert_eq!(listed_server2.len(), 2);
    }
}
