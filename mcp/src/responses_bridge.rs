//! Shared builders for Responses/Chat tool payloads derived from MCP tool inventory.
//!
//! This module centralizes conversion logic that was previously duplicated in routers:
//! - MCP ToolEntry -> function tool JSON (for upstream model calls)
//! - MCP ToolEntry -> chat/common Tool structs
//! - MCP ToolEntry -> Responses ResponseTool structs
//! - MCP ToolEntry list -> mcp_list_tools output item payloads

use openai_protocol::{
    common::{Function, Tool},
    responses::{generate_id, McpToolInfo, ResponseOutputItem, ResponseTool, ResponseToolType},
};
use serde_json::{json, Value};

use crate::inventory::ToolEntry;

/// Build function-tool JSON payloads from MCP tool entries.
///
/// These are used when routers expose MCP tools as function tools to upstream model APIs.
pub fn build_function_tools_json(entries: &[ToolEntry]) -> Vec<Value> {
    entries
        .iter()
        .map(|entry| {
            json!({
                "type": "function",
                "name": entry.tool.name,
                "description": entry.tool.description,
                "parameters": Value::Object((*entry.tool.input_schema).clone())
            })
        })
        .collect()
}

/// Build Chat API function tools from MCP tool entries.
pub fn build_chat_function_tools(entries: &[ToolEntry]) -> Vec<Tool> {
    entries
        .iter()
        .map(|entry| Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: entry.tool.name.to_string(),
                description: entry.tool.description.as_ref().map(|d| d.to_string()),
                parameters: Value::Object((*entry.tool.input_schema).clone()),
                strict: None,
            },
        })
        .collect()
}

/// Build Responses API MCP tools from MCP tool entries.
///
/// These tools are exposed in Responses requests where MCP tools are represented
/// as `{"type": "mcp", ...}` tool entries.
pub fn build_response_tools(entries: &[ToolEntry]) -> Vec<ResponseTool> {
    entries
        .iter()
        .map(|entry| ResponseTool {
            r#type: ResponseToolType::Mcp,
            function: Some(Function {
                name: entry.tool.name.to_string(),
                description: entry.tool.description.as_ref().map(|d| d.to_string()),
                parameters: Value::Object((*entry.tool.input_schema).clone()),
                strict: None,
            }),
            server_url: None,
            authorization: None,
            headers: None,
            server_label: Some(entry.server_key().to_string()),
            server_description: None,
            require_approval: None,
            allowed_tools: None,
        })
        .collect()
}

/// Build MCP tool infos used by `mcp_list_tools` output items.
pub fn build_mcp_tool_infos(entries: &[ToolEntry]) -> Vec<McpToolInfo> {
    entries
        .iter()
        .map(|entry| McpToolInfo {
            name: entry.tool_name().to_string(),
            description: entry.tool.description.as_ref().map(|d| d.to_string()),
            input_schema: Value::Object((*entry.tool.input_schema).clone()),
            annotations: entry
                .tool
                .annotations
                .as_ref()
                .and_then(|a| serde_json::to_value(a).ok()),
        })
        .collect()
}

/// Build a typed `mcp_list_tools` output item.
pub fn build_mcp_list_tools_item(server_label: &str, entries: &[ToolEntry]) -> ResponseOutputItem {
    ResponseOutputItem::McpListTools {
        id: generate_id("mcpl"),
        server_label: server_label.to_string(),
        tools: build_mcp_tool_infos(entries),
    }
}

/// Build a JSON `mcp_list_tools` output item payload.
///
/// Useful for routers that build/manipulate raw JSON responses.
pub fn build_mcp_list_tools_json(server_label: &str, entries: &[ToolEntry]) -> Value {
    serde_json::to_value(build_mcp_list_tools_item(server_label, entries)).unwrap_or_else(
        |_| json!({ "type": "mcp_list_tools", "server_label": server_label, "tools": [] }),
    )
}
