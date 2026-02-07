//! MCP Tool Session — bundles all MCP execution state for a single request.
//!
//! Instead of threading `orchestrator`, `request_ctx`, `mcp_servers`, `server_keys`,
//! and `mcp_tools` through every function, callers create one `McpToolSession` and
//! pass `&session` everywhere. When an MCP parameter changes (e.g. `mcp_servers`
//! representation), only this struct and its constructor need updating — not every
//! router function signature.

use serde_json::Value;

use super::orchestrator::{
    McpOrchestrator, McpRequestContext, ToolExecutionInput, ToolExecutionOutput,
};
use crate::{
    approval::ApprovalMode, inventory::ToolEntry, tenant::TenantContext, transform::ResponseFormat,
};

/// Bundles all MCP execution state for a single request.
///
/// Created once per request, then passed by reference to every function
/// that needs MCP infrastructure. This eliminates repeated parameter
/// threading of `orchestrator`, `request_ctx`, `mcp_servers`, `server_keys`,
/// and `mcp_tools`.
pub struct McpToolSession<'a> {
    orchestrator: &'a McpOrchestrator,
    request_ctx: McpRequestContext<'a>,
    mcp_servers: Vec<(String, String)>,
    server_keys: Vec<String>,
    mcp_tools: Vec<ToolEntry>,
}

impl<'a> McpToolSession<'a> {
    /// Create a new session by performing the setup every path currently repeats:
    /// 1. Create request context with default tenant and policy-only approval
    /// 2. Extract server_keys from mcp_servers
    /// 3. List tools for those servers
    pub fn new(
        orchestrator: &'a McpOrchestrator,
        mcp_servers: Vec<(String, String)>,
        request_id: impl Into<String>,
    ) -> Self {
        let request_ctx = orchestrator.create_request_context(
            request_id,
            TenantContext::default(),
            ApprovalMode::PolicyOnly,
        );
        let server_keys: Vec<String> = mcp_servers.iter().map(|(_, key)| key.clone()).collect();
        let mcp_tools = orchestrator.list_tools_for_servers(&server_keys);

        Self {
            orchestrator,
            request_ctx,
            mcp_servers,
            server_keys,
            mcp_tools,
        }
    }

    // --- Accessors ---

    pub fn orchestrator(&self) -> &McpOrchestrator {
        self.orchestrator
    }

    pub fn request_ctx(&self) -> &McpRequestContext<'a> {
        &self.request_ctx
    }

    pub fn mcp_servers(&self) -> &[(String, String)] {
        &self.mcp_servers
    }

    pub fn server_keys(&self) -> &[String] {
        &self.server_keys
    }

    pub fn mcp_tools(&self) -> &[ToolEntry] {
        &self.mcp_tools
    }

    // --- Delegation methods ---

    /// Execute multiple tools via the orchestrator's batch API.
    ///
    /// Delegates to `orchestrator.execute_tools()` with this session's
    /// `server_keys`, `mcp_servers`, and `request_ctx`.
    pub async fn execute_tools(&self, inputs: Vec<ToolExecutionInput>) -> Vec<ToolExecutionOutput> {
        self.orchestrator
            .execute_tools(
                inputs,
                &self.server_keys,
                &self.mcp_servers,
                &self.request_ctx,
            )
            .await
    }

    /// Find a tool entry by name within this session's allowed servers.
    pub fn find_tool_by_name(&self, tool_name: &str) -> Option<ToolEntry> {
        self.orchestrator
            .find_tool_by_name(tool_name, &self.server_keys)
    }

    /// Resolve the user-facing server label for a tool.
    ///
    /// Uses the orchestrator inventory to find the tool's server key, then maps
    /// it to the request's MCP server label. Falls back to the first server
    /// label (or "mcp").
    pub fn resolve_tool_server_label(&self, tool_name: &str) -> String {
        let fallback_label = self
            .mcp_servers
            .first()
            .map(|(label, _)| label.as_str())
            .unwrap_or("mcp");

        let Some(entry) = self.find_tool_by_name(tool_name) else {
            return fallback_label.to_string();
        };

        let server_key = entry.qualified_name.server_key();
        self.mcp_servers
            .iter()
            .find(|(_, key)| key == server_key)
            .map(|(label, _)| label.clone())
            .unwrap_or_else(|| fallback_label.to_string())
    }

    /// List tools for a single server key.
    ///
    /// Useful for emitting per-server `mcp_list_tools` items.
    pub fn list_tools_for_server(&self, server_key: &str) -> Vec<ToolEntry> {
        self.orchestrator
            .list_tools_for_servers(&[server_key.to_string()])
    }

    /// Call a single tool by name (for streaming paths that process tools individually).
    ///
    /// Returns the raw `ToolCallResult` for callers that need fine-grained control
    /// over result handling (e.g. streaming event emission).
    pub async fn call_tool_by_name(
        &self,
        tool_name: &str,
        arguments: Value,
        server_label: &str,
    ) -> crate::error::McpResult<super::orchestrator::ToolCallResult> {
        self.orchestrator
            .call_tool_by_name(
                tool_name,
                arguments,
                &self.server_keys,
                server_label,
                &self.request_ctx,
            )
            .await
    }

    /// Look up the response format for a tool.
    ///
    /// Convenience method that returns `Passthrough` if the tool is not found.
    pub fn tool_response_format(&self, tool_name: &str) -> ResponseFormat {
        self.find_tool_by_name(tool_name)
            .map(|entry| entry.response_format)
            .unwrap_or(ResponseFormat::Passthrough)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation_extracts_server_keys() {
        let orchestrator = McpOrchestrator::new_test();
        let mcp_servers = vec![
            ("label1".to_string(), "key1".to_string()),
            ("label2".to_string(), "key2".to_string()),
        ];

        let session = McpToolSession::new(&orchestrator, mcp_servers, "test-request");

        assert_eq!(session.server_keys(), &["key1", "key2"]);
        assert_eq!(session.mcp_servers().len(), 2);
        assert_eq!(
            session.mcp_servers()[0],
            ("label1".to_string(), "key1".to_string())
        );
    }

    #[test]
    fn test_session_empty_servers() {
        let orchestrator = McpOrchestrator::new_test();
        let session = McpToolSession::new(&orchestrator, vec![], "test-request");

        assert!(session.server_keys().is_empty());
        assert!(session.mcp_servers().is_empty());
        assert!(session.mcp_tools().is_empty());
    }

    #[test]
    fn test_resolve_tool_server_label_fallback() {
        let orchestrator = McpOrchestrator::new_test();
        let mcp_servers = vec![("my_label".to_string(), "my_key".to_string())];
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
    fn test_find_tool_by_name_not_found() {
        let orchestrator = McpOrchestrator::new_test();
        let session = McpToolSession::new(&orchestrator, vec![], "test-request");

        assert!(session.find_tool_by_name("nonexistent").is_none());
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
            description: Some(Cow::Owned(format!("Test tool: {}", name))),
            input_schema: Arc::new(serde_json::Map::new()),
            output_schema: None,
            annotations: None,
            icons: None,
        }
    }

    #[test]
    fn test_find_tool_with_inventory() {
        let orchestrator = McpOrchestrator::new_test();

        // Insert a tool into the inventory
        let tool = create_test_tool("test_tool");
        let entry = ToolEntry::from_server_tool("server1", tool);
        orchestrator.tool_inventory().insert_entry(entry);

        let mcp_servers = vec![("label1".to_string(), "server1".to_string())];
        let session = McpToolSession::new(&orchestrator, mcp_servers, "test-request");

        // Should find the tool
        assert!(session.find_tool_by_name("test_tool").is_some());
        assert_eq!(session.mcp_tools().len(), 1);
    }

    #[test]
    fn test_resolve_label_with_inventory() {
        let orchestrator = McpOrchestrator::new_test();

        let tool = create_test_tool("test_tool");
        let entry = ToolEntry::from_server_tool("server1", tool);
        orchestrator.tool_inventory().insert_entry(entry);

        let mcp_servers = vec![("my_server".to_string(), "server1".to_string())];
        let session = McpToolSession::new(&orchestrator, mcp_servers, "test-request");

        let label = session.resolve_tool_server_label("test_tool");
        assert_eq!(label, "my_server");
    }
}
