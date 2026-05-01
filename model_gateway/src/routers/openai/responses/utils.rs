//! Minimal request/response tool-shape utilities for OpenAI responses.

use openai_protocol::responses::{response_tool_echo_value, ResponseTool};
use serde_json::Value;

/// Convert a single `ResponseTool` back to its original JSON representation.
pub(super) fn response_tool_to_value(tool: &ResponseTool) -> Value {
    response_tool_echo_value(tool)
}
