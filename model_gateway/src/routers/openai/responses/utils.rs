//! Response patching and transformation utilities for OpenAI responses

use openai_protocol::{
    event_types::is_response_event,
    responses::{ResponseTool, ResponsesRequest},
};
use serde::Serialize;
use serde_json::{json, Map, Value};
use smg_mcp::McpToolSession;
use tracing::warn;

use super::common::parse_sse_block;

/// Check if a JSON value is missing, null, or an empty string
fn is_missing_or_empty(value: Option<&Value>) -> bool {
    match value {
        None => true,
        Some(v) => v.is_null() || v.as_str().is_some_and(|s| s.is_empty()),
    }
}

/// Insert a string value into a JSON object if the condition is met
fn insert_if<F>(obj: &mut Map<String, Value>, key: &str, value: &str, condition: F)
where
    F: FnOnce(&Map<String, Value>) -> bool,
{
    if condition(obj) {
        obj.insert(key.to_string(), Value::String(value.to_string()));
    }
}

/// Patch response JSON with metadata from original request
///
/// The upstream response may be missing fields that were in the original request.
/// This function ensures these fields are preserved in the final response:
/// - `previous_response_id` - conversation threading
/// - `instructions` - system instructions
/// - `metadata` - user-provided metadata
/// - `store` - whether to persist the response
/// - `model` - model identifier
/// - `safety_identifier` - user identifier for safety
pub(super) fn patch_response_with_request_metadata(
    response_json: &mut Value,
    original_body: &ResponsesRequest,
    original_previous_response_id: Option<&str>,
) {
    let Some(obj) = response_json.as_object_mut() else {
        return;
    };

    // Set previous_response_id if missing/empty
    if let Some(prev_id) = original_previous_response_id {
        insert_if(obj, "previous_response_id", prev_id, |o| {
            is_missing_or_empty(o.get("previous_response_id"))
        });
    }

    // Set instructions if missing/null
    if let Some(instructions) = &original_body.instructions {
        insert_if(obj, "instructions", instructions, |o| {
            is_missing_or_empty(o.get("instructions"))
        });
    }

    // Set metadata if missing/null
    if is_missing_or_empty(obj.get("metadata")) {
        if let Some(metadata) = &original_body.metadata {
            let metadata_map: Map<String, Value> = metadata
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            obj.insert("metadata".to_string(), Value::Object(metadata_map));
        }
    }

    // Always set store
    obj.insert(
        "store".to_string(),
        Value::Bool(original_body.store.unwrap_or(true)),
    );

    // Set model if missing/empty
    insert_if(obj, "model", &original_body.model, |o| {
        is_missing_or_empty(o.get("model"))
    });

    // Set safety_identifier if null (but key exists)
    if let Some(user) = &original_body.user {
        if obj
            .get("safety_identifier")
            .is_some_and(|v: &Value| v.is_null())
        {
            obj.insert("safety_identifier".to_string(), Value::String(user.clone()));
        }
    }

    // Attach conversation id for client response
    if let Some(conv_id) = &original_body.conversation {
        obj.insert("conversation".to_string(), json!({ "id": conv_id }));
    }
}

/// Rebuild SSE block with new data payload
fn rebuild_sse_block(block: &str, new_payload: &str) -> String {
    let mut rebuilt_lines = Vec::new();
    let mut data_written = false;

    for line in block.lines() {
        if line.starts_with("data:") {
            if !data_written {
                rebuilt_lines.push(format!("data: {new_payload}"));
                data_written = true;
            }
        } else {
            rebuilt_lines.push(line.to_string());
        }
    }

    if !data_written {
        rebuilt_lines.push(format!("data: {new_payload}"));
    }

    rebuilt_lines.join("\n")
}

/// Rewrite streaming SSE block to include metadata from original request
pub(super) fn rewrite_streaming_block(
    block: &str,
    original_body: &ResponsesRequest,
    original_previous_response_id: Option<&str>,
) -> Option<String> {
    let trimmed = block.trim();
    if trimmed.is_empty() {
        return None;
    }

    let (_, data) = parse_sse_block(trimmed);
    if data.is_empty() {
        return None;
    }
    let mut parsed: Value = serde_json::from_str(&data)
        .map_err(|e| warn!("Failed to parse streaming JSON payload: {}", e))
        .ok()?;

    let event_type = parsed
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or_default();

    if !is_response_event(event_type) {
        return None;
    }

    let response_obj = parsed.get_mut("response").and_then(|v| v.as_object_mut())?;
    let mut changed = false;

    // Update store value if different
    let desired_store = Value::Bool(original_body.store.unwrap_or(true));
    if response_obj.get("store") != Some(&desired_store) {
        response_obj.insert("store".to_string(), desired_store);
        changed = true;
    }

    // Set previous_response_id if missing/empty
    if let Some(prev_id) = original_previous_response_id {
        if is_missing_or_empty(response_obj.get("previous_response_id")) {
            response_obj.insert("previous_response_id".to_string(), json!(prev_id));
            changed = true;
        }
    }

    // Attach conversation id
    if let Some(conv_id) = &original_body.conversation {
        response_obj.insert("conversation".to_string(), json!({ "id": conv_id }));
        changed = true;
    }

    if !changed {
        return None;
    }

    let new_payload = serde_json::to_string(&parsed)
        .map_err(|e| warn!("Failed to serialize modified streaming payload: {}", e))
        .ok()?;

    Some(rebuild_sse_block(trimmed, &new_payload))
}

/// Helper to insert an optional serializable field into a JSON map.
pub(super) fn insert_optional_value<T: Serialize>(
    map: &mut Map<String, Value>,
    key: &str,
    value: Option<&T>,
) {
    if let Some(v) = value {
        match serde_json::to_value(v) {
            Ok(val) => {
                map.insert(key.to_string(), val);
            }
            Err(e) => {
                warn!(field = key, error = %e, "Failed to serialize optional field");
            }
        }
    }
}

/// Convert a single ResponseTool back to its original JSON representation.
///
/// Handles MCP tools (with server metadata), web_search_preview, and code_interpreter.
/// Returns None for function tools and other types that don't need restoration.
pub(super) fn response_tool_to_value(tool: &ResponseTool) -> Option<Value> {
    match tool {
        ResponseTool::Mcp(mcp) => {
            let mut m = Map::new();
            m.insert("type".to_string(), json!("mcp"));
            m.insert("server_label".to_string(), json!(&mcp.server_label));
            insert_optional_value(&mut m, "server_url", mcp.server_url.as_ref());
            insert_optional_value(
                &mut m,
                "server_description",
                mcp.server_description.as_ref(),
            );
            insert_optional_value(&mut m, "require_approval", mcp.require_approval.as_ref());
            if let Some(allowed) = &mcp.allowed_tools {
                m.insert(
                    "allowed_tools".to_string(),
                    Value::Array(allowed.iter().map(|s| json!(s)).collect()),
                );
            }
            Some(Value::Object(m))
        }
        ResponseTool::WebSearchPreview(_) => serde_json::to_value(tool).ok(),
        ResponseTool::CodeInterpreter(_) => serde_json::to_value(tool).ok(),
        ResponseTool::Function(_) => None,
    }
}

/// Restore original tools (MCP and builtin) in response for client.
///
/// The model receives function tools, but the response should mirror the original
/// request's tool format (MCP tools with server_url, builtin tools like web_search_preview).
pub(super) fn restore_original_tools(
    resp: &mut Value,
    original_body: &ResponsesRequest,
    session: Option<&McpToolSession<'_>>,
) {
    strip_internal_mcp_output_items(resp, session);
    strip_internal_mcp_tools(resp, session);

    let Some(original_tools) = original_body.tools.as_ref() else {
        return;
    };

    let mut restored_tools: Vec<Value> = original_tools
        .iter()
        .filter_map(response_tool_to_value)
        .collect();

    restored_tools.retain(|tool| !is_internal_mcp_tool_value(tool, session));

    if restored_tools.is_empty() {
        let had_restorable_original_tool = original_tools
            .iter()
            .any(|tool| response_tool_to_value(tool).is_some());
        if had_restorable_original_tool {
            if let Some(obj) = resp.as_object_mut() {
                obj.remove("tools");
            }
        }
        return;
    }

    if let Some(obj) = resp.as_object_mut() {
        obj.insert("tools".to_string(), Value::Array(restored_tools));
        obj.entry("tool_choice").or_insert(json!("auto"));
    }
}

fn strip_internal_mcp_tools(resp: &mut Value, session: Option<&McpToolSession<'_>>) {
    let Some(obj) = resp.as_object_mut() else {
        return;
    };

    let Some(tools) = obj.get_mut("tools").and_then(|value| value.as_array_mut()) else {
        return;
    };

    tools.retain(|tool| !is_internal_mcp_tool_value(tool, session));
}

fn strip_internal_mcp_output_items(resp: &mut Value, session: Option<&McpToolSession<'_>>) {
    let Some(obj) = resp.as_object_mut() else {
        return;
    };

    let Some(output) = obj.get_mut("output").and_then(|value| value.as_array_mut()) else {
        return;
    };

    output.retain(|item| !is_internal_mcp_output_item(item, session));
}

fn is_internal_mcp_tool_value(tool: &Value, session: Option<&McpToolSession<'_>>) -> bool {
    let Some(session) = session else {
        return false;
    };

    match (
        tool.get("type").and_then(|value| value.as_str()),
        tool.get("name").and_then(|value| value.as_str()),
        tool.get("server_label").and_then(|value| value.as_str()),
    ) {
        (Some("function"), Some(name), _) => session.is_internal_tool(name),
        (Some("mcp"), _, Some(server_label)) => session.is_internal_server_label(server_label),
        _ => false,
    }
}

fn is_internal_mcp_output_item(item: &Value, session: Option<&McpToolSession<'_>>) -> bool {
    let Some(session) = session else {
        return false;
    };

    match (
        item.get("type").and_then(|value| value.as_str()),
        item.get("name").and_then(|value| value.as_str()),
        item.get("server_label").and_then(|value| value.as_str()),
    ) {
        (Some("mcp_list_tools"), _, Some(server_label)) => {
            session.is_internal_server_label(server_label)
        }
        (Some("mcp_call"), Some(name), _) => session.is_internal_tool(name),
        (Some("mcp_call"), _, Some(server_label)) => session.is_internal_server_label(server_label),
        (Some("function_call"), Some(name), _) => session.is_internal_tool(name),
        (Some("function_tool_call"), Some(name), _) => session.is_internal_tool(name),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::responses::{ResponseInput, ResponsesRequest};
    use serde_json::json;
    use smg_mcp::{
        BuiltinToolType, McpConfig, McpOrchestrator, McpServerBinding, McpServerConfig,
        McpToolSession, McpTransport, Tool, ToolEntry,
    };

    use super::restore_original_tools;

    fn test_tool(name: &str) -> Tool {
        let mut schema = serde_json::Map::new();
        schema.insert("type".to_string(), json!("object"));
        schema.insert(
            "properties".to_string(),
            json!({
                "query": { "type": "string" }
            }),
        );

        Tool {
            name: name.to_string().into(),
            title: None,
            description: Some("internal".into()),
            input_schema: schema.into(),
            output_schema: None,
            icons: None,
            annotations: None,
        }
    }

    #[tokio::test]
    async fn restore_original_tools_strips_injected_internal_tool_when_request_had_no_tools() {
        let original_body = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "internal-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: true,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("internal_search"),
            ));
        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );
        let mut response = serde_json::json!({
            "tools": [{
                "type": "function",
                "name": "internal_search",
                "description": "internal",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": ["query"]
                }
            }],
            "tool_choice": "auto"
        });

        restore_original_tools(&mut response, &original_body, Some(&session));

        assert_eq!(response["tools"], serde_json::json!([]));
        assert_eq!(response["tool_choice"], "auto");
    }

    #[tokio::test]
    async fn restore_original_tools_strips_internal_mcp_output_items() {
        let original_body = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "internal-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: true,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("internal_search"),
            ));
        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );
        let mut response = serde_json::json!({
            "output": [
                {
                    "type": "mcp_list_tools",
                    "server_label": "internal-label",
                    "tools": []
                },
                {
                    "type": "mcp_call",
                    "name": "internal_search",
                    "server_label": "internal-label"
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}]
                }
            ]
        });

        restore_original_tools(&mut response, &original_body, Some(&session));

        assert_eq!(
            response["output"],
            serde_json::json!([{
                "type": "message",
                "content": [{"type": "output_text", "text": "visible"}]
            }])
        );
    }

    #[tokio::test]
    async fn restore_original_tools_strips_internal_function_tool_call_items() {
        let original_body = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "internal-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: true,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("internal_search"),
            ));
        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );
        let mut response = serde_json::json!({
            "output": [
                {
                    "type": "function_tool_call",
                    "call_id": "call_123",
                    "name": "internal_search",
                    "arguments": "{\"query\":\"secret\"}"
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}]
                }
            ]
        });

        restore_original_tools(&mut response, &original_body, Some(&session));

        assert_eq!(
            response["output"],
            serde_json::json!([{
                "type": "message",
                "content": [{"type": "output_text", "text": "visible"}]
            }])
        );
    }

    #[tokio::test]
    async fn restore_original_tools_keeps_builtin_output_items_visible() {
        let original_body = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "internal-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("brave_web_search".to_string()),
                internal: true,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("brave_web_search"),
            ));
        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );
        let mut response = serde_json::json!({
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws_call_123",
                    "server_label": "internal-label",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "private query",
                        "queries": ["private query"],
                        "sources": []
                    }
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}]
                }
            ]
        });

        restore_original_tools(&mut response, &original_body, Some(&session));

        assert_eq!(
            response["output"],
            serde_json::json!([
                {
                    "type": "web_search_call",
                    "id": "ws_call_123",
                    "server_label": "internal-label",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "private query",
                        "queries": ["private query"],
                        "sources": []
                    }
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}]
                }
            ])
        );
    }
}
