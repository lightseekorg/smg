//! OpenAI-compatible responses handling module
//!
//! This module provides comprehensive support for OpenAI Responses API with:
//! - Streaming and non-streaming response handling
//! - MCP (Model Context Protocol) tool interception and execution
//! - SSE (Server-Sent Events) parsing and forwarding
//! - Response accumulation for persistence
//! - Tool call detection and output index remapping
//! - Input history loading from conversations and response chains
//! - Storage query handlers for response retrieval

mod accumulator;
mod common;
pub(crate) mod history;
mod non_streaming;
pub(crate) mod route;
mod streaming;
mod utils;

// Re-exported for openai::mcp::tool_handler (cross-module dependency)
pub(crate) use accumulator::StreamingResponseAccumulator;
// --- Storage query handlers (extracted from router.rs) ---
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
pub(crate) use common::{extract_output_index, get_event_type};
pub use non_streaming::handle_non_streaming_response;
use openai_protocol::responses::generate_id;
use serde_json::{json, Value};
use smg_data_connector::ResponseId;
pub use streaming::handle_streaming_response;
use tracing::warn;

use super::context::ResponsesComponents;
use crate::routers::error;

/// Fetch a single stored response by ID.
pub(crate) async fn get_response(components: &ResponsesComponents, response_id: &str) -> Response {
    let id = ResponseId::from(response_id);
    match components.response_storage.get_response(&id).await {
        Ok(Some(stored)) => {
            let mut response_json = stored.raw_response;
            if let Some(obj) = response_json.as_object_mut() {
                obj.insert("id".to_string(), json!(id.0));
            }
            (StatusCode::OK, Json(response_json)).into_response()
        }
        Ok(None) => error::not_found(
            "not_found",
            format!("No response found with id '{response_id}'"),
        ),
        Err(e) => error::internal_error("storage_error", format!("Failed to get response: {e}")),
    }
}

/// List input items for a stored response.
pub(crate) async fn list_response_input_items(
    components: &ResponsesComponents,
    response_id: &str,
) -> Response {
    let resp_id = ResponseId::from(response_id);

    match components.response_storage.get_response(&resp_id).await {
        Ok(Some(stored)) => {
            let items = stored.input.as_array().cloned().unwrap_or_default();

            let items_with_ids: Vec<Value> = items
                .into_iter()
                .map(|mut item| {
                    if item.get("id").is_none() {
                        if let Some(obj) = item.as_object_mut() {
                            obj.insert("id".to_string(), json!(generate_id("msg")));
                        }
                    }
                    item
                })
                .collect();

            let response_body = json!({
                "object": "list",
                "data": items_with_ids,
                "first_id": items_with_ids.first().and_then(|v| v.get("id").and_then(|i| i.as_str())),
                "last_id": items_with_ids.last().and_then(|v| v.get("id").and_then(|i| i.as_str())),
                "has_more": false
            });

            (StatusCode::OK, Json(response_body)).into_response()
        }
        Ok(None) => error::not_found(
            "not_found",
            format!("No response found with id '{response_id}'"),
        ),
        Err(e) => {
            warn!("Failed to retrieve input items for {}: {}", response_id, e);
            error::internal_error(
                "storage_error",
                format!("Failed to retrieve input items: {e}"),
            )
        }
    }
}
