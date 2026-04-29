//! Minimal request/response tool-shape utilities for OpenAI responses.

use openai_protocol::responses::ResponseTool;
use serde::Serialize;
use serde_json::{json, Map, Value};
use tracing::warn;

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

/// Convert a single `ResponseTool` back to its original JSON representation.
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
            insert_optional_value(&mut m, "allowed_tools", mcp.allowed_tools.as_ref());
            insert_optional_value(&mut m, "connector_id", mcp.connector_id.as_ref());
            insert_optional_value(&mut m, "defer_loading", mcp.defer_loading.as_ref());
            Some(Value::Object(m))
        }
        ResponseTool::Function(_)
        | ResponseTool::WebSearchPreview(_)
        | ResponseTool::WebSearch(_)
        | ResponseTool::CodeInterpreter(_)
        | ResponseTool::FileSearch(_)
        | ResponseTool::ImageGeneration(_)
        | ResponseTool::Computer
        | ResponseTool::ComputerUsePreview(_)
        | ResponseTool::Custom(_)
        | ResponseTool::Namespace(_)
        | ResponseTool::Shell(_)
        | ResponseTool::ApplyPatch
        | ResponseTool::LocalShell => serde_json::to_value(tool).ok(),
    }
}
