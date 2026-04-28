//! Request-side tool preparation for the OpenAI Responses router.

use openai_protocol::event_types::ItemType;
use serde_json::Value;
use smg_mcp::McpToolSession;

/// Replace MCP / builtin tools in the OpenAI request payload with the
/// function-tool view exposed by the request-scoped MCP session.
pub(crate) fn prepare_mcp_tools_as_functions(payload: &mut Value, session: &McpToolSession<'_>) {
    let Some(obj) = payload.as_object_mut() else {
        return;
    };

    let mut retained_tools: Vec<Value> = Vec::new();
    if let Some(v) = obj.get_mut("tools") {
        if let Some(arr) = v.as_array_mut() {
            retained_tools = arr
                .drain(..)
                .filter(|item| {
                    item.get("type")
                        .and_then(|v| v.as_str())
                        .is_some_and(|s| s == ItemType::FUNCTION)
                })
                .collect();
        }
    }

    let session_tools = session.build_function_tools_json();
    let mut tools_json = Vec::with_capacity(retained_tools.len() + session_tools.len());
    tools_json.append(&mut retained_tools);
    tools_json.extend(session_tools);

    if !tools_json.is_empty() {
        obj.insert("tools".to_string(), Value::Array(tools_json));
        obj.insert("tool_choice".to_string(), Value::String("auto".to_string()));
    }
}
