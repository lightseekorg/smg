//! Shared builders for Responses/Chat tool payloads derived from MCP tool inventory.
//!
//! This module centralizes conversion logic that was previously duplicated in routers:
//! - MCP ToolEntry -> function tool JSON (for upstream model calls)
//! - MCP ToolEntry -> chat/common Tool structs
//! - MCP ToolEntry -> Responses ResponseTool structs
//! - MCP ToolEntry list -> mcp_list_tools output item payloads
//!
//! # Schema cloning
//!
//! Each `ToolEntry` stores its JSON Schema as `Arc<serde_json::Map>`, which enables
//! cheap sharing across the inventory.  The downstream protocol types (`Function`,
//! `McpToolInfo`, and `serde_json::Value`) all require an *owned* `serde_json::Map`
//! inside `Value::Object`, so every builder must deep-clone the schema map once per
//! tool per call.  This is intentional -- the clone happens O(tools) times per
//! request and schema maps are typically small (a handful of properties).  If
//! profiling ever shows this to be a bottleneck, consider caching the materialised
//! `Value::Object` alongside the `Arc<Map>` in `ToolEntry`.

use openai_protocol::{
    common::{Function, Tool},
    responses::{generate_id, FunctionTool, McpToolInfo, ResponseOutputItem, ResponseTool},
};
use serde_json::{json, Value};

use crate::inventory::{QualifiedToolName, ToolEntry};

/// Materialise a `serde_json::Map` reference into an owned `Value::Object`.
///
/// This deep-clones the map.  All builder functions in this module need an
/// owned `Value` because the downstream protocol structs (`Function.parameters`,
/// `McpToolInfo.input_schema`) are `Value`-typed and will be serialised into the
/// API response.  See the module-level "Schema cloning" docs for why this is
/// unavoidable.
#[inline]
fn schema_to_value(schema: &serde_json::Map<String, Value>) -> Value {
    Value::Object(schema.clone())
}

fn resolved_name_for_entry<'a>(
    entry: &'a ToolEntry,
    exposed_names: Option<&'a std::collections::HashMap<QualifiedToolName, String>>,
) -> &'a str {
    exposed_names
        .and_then(|m| m.get(&entry.qualified_name))
        .map(|s| s.as_str())
        .unwrap_or_else(|| entry.tool_name())
}

/// Resolved (name, description, &input_schema) triples from MCP tool entries.
///
/// This is the shared extraction logic used by the JSON, Chat, and Responses
/// builder functions so that name-resolution lives in one place.  The schema is
/// returned by reference; callers clone when they need an owned `Value`.
fn resolved_tool_fields<'a>(
    entries: &'a [ToolEntry],
    exposed_names: Option<&'a std::collections::HashMap<QualifiedToolName, String>>,
) -> impl Iterator<Item = (&'a str, Option<&'a str>, &'a serde_json::Map<String, Value>)> + 'a {
    entries.iter().map(move |entry| {
        let name = resolved_name_for_entry(entry, exposed_names);
        let description = entry.tool.description.as_deref();
        (name, description, &*entry.tool.input_schema)
    })
}

/// Build function-tool JSON payloads from MCP tool entries.
///
/// These are used when routers expose MCP tools as function tools to upstream model APIs.
pub fn build_function_tools_json(entries: &[ToolEntry]) -> Vec<Value> {
    build_function_tools_json_with_names(entries, None)
}

/// Build function-tool JSON payloads from MCP tool entries with optional exposed names.
pub fn build_function_tools_json_with_names(
    entries: &[ToolEntry],
    exposed_names: Option<&std::collections::HashMap<QualifiedToolName, String>>,
) -> Vec<Value> {
    resolved_tool_fields(entries, exposed_names)
        .map(|(name, description, parameters)| {
            json!({
                "type": "function",
                "name": name,
                "description": description,
                "parameters": schema_to_value(parameters)
            })
        })
        .collect()
}

/// Build Chat API function tools from MCP tool entries.
pub fn build_chat_function_tools(entries: &[ToolEntry]) -> Vec<Tool> {
    build_chat_function_tools_with_names(entries, None)
}

/// Build Chat API function tools from MCP tool entries with optional exposed names.
pub fn build_chat_function_tools_with_names(
    entries: &[ToolEntry],
    exposed_names: Option<&std::collections::HashMap<QualifiedToolName, String>>,
) -> Vec<Tool> {
    resolved_tool_fields(entries, exposed_names)
        .map(|(name, description, parameters)| Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: description.map(|d| d.to_string()),
                parameters: schema_to_value(parameters),
                strict: None,
            },
        })
        .collect()
}

/// Build Responses API function tools from MCP tool entries.
///
/// MCP tools are exposed to the model as function tools in the Responses API,
/// so these serialize as `{"type": "function", ...}` tool entries.
pub fn build_response_tools(entries: &[ToolEntry]) -> Vec<ResponseTool> {
    build_response_tools_with_names(entries, None)
}

/// Build Responses API function tools from MCP tool entries with optional exposed names.
pub fn build_response_tools_with_names(
    entries: &[ToolEntry],
    exposed_names: Option<&std::collections::HashMap<QualifiedToolName, String>>,
) -> Vec<ResponseTool> {
    resolved_tool_fields(entries, exposed_names)
        .map(|(name, description, parameters)| {
            ResponseTool::Function(FunctionTool {
                function: Function {
                    name: name.to_string(),
                    description: description.map(|d| d.to_string()),
                    parameters: schema_to_value(parameters),
                    strict: None,
                },
            })
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
            input_schema: schema_to_value(&entry.tool.input_schema),
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
