//! Surface-side helpers for regular Responses.
//!
//! - `ResponsesCallContext` — request-scoped knobs the handler bundles
//!   for both modes.
//! - `LoadedConversationHistory` — the loader's return shape.
//! - `load_conversation_history` — surface-side wrapper around the
//!   shared `load_request_history` primitive that resolves
//!   `previous_response_id` **or** `conversation` (mutually exclusive,
//!   validated upstream) and merges the lowered transcript into the
//!   request's input.

use std::collections::HashSet;

use axum::{http, response::Response};
use openai_protocol::responses::{
    self, ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponsesRequest,
};

use crate::{
    middleware::TenantRequestMeta,
    routers::{
        common::transcript_lower::{
            extract_control_items, extract_mcp_list_tools_server_labels, lower_transcript,
        },
        grpc::common::responses::{load_request_history, ResponsesContext},
    },
};

pub(super) struct LoadedConversationHistory {
    pub request: ResponsesRequest,
    pub existing_mcp_list_tools_labels: HashSet<String>,
    /// Control items collected from both history and hand-stitched input.
    pub control_items: Vec<ResponseInputOutputItem>,
}

/// Per-request parameters for chat pipeline execution. Bundles values
/// that are always threaded together through the regular responses
/// call chain.
pub(super) struct ResponsesCallContext {
    pub headers: Option<http::HeaderMap>,
    pub model_id: String,
    pub response_id: Option<String>,
    pub tenant_request_meta: TenantRequestMeta,
}

/// Resolve `previous_response_id` or `conversation` and merge the
/// resulting items in front of the current request's input. Two
/// sources for the same canonical-input contract — adapters do not
/// see the difference.
pub(super) async fn load_conversation_history(
    ctx: &ResponsesContext,
    request: &ResponsesRequest,
) -> Result<LoadedConversationHistory, Response> {
    let loaded = load_request_history(ctx, request).await?;
    let mut existing_mcp_list_tools_labels = loaded.existing_mcp_list_tools_labels;
    let mut control_items = loaded.control_items;

    let mut modified_request = request.clone();
    let mut combined = loaded.items;
    match modified_request.input {
        ResponseInput::Items(items) => {
            // Hand-stitched input: pick up `mcp_list_tools.server_label`
            // values the caller inlined directly so the agent loop does
            // not re-emit a fresh listing for the same server. Same
            // treatment for approval-pair / list_tools control items.
            existing_mcp_list_tools_labels.extend(extract_mcp_list_tools_server_labels(&items));
            control_items.extend(extract_control_items(&items));
            for item in items {
                combined.push(responses::normalize_input_item(&item));
            }
        }
        ResponseInput::Text(text) => {
            combined.push(ResponseInputOutputItem::Message {
                id: format!("msg_u_{}", uuid::Uuid::now_v7()),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText { text }],
                status: Some("completed".to_string()),
                phase: None,
            });
        }
    }

    // Lower once over the combined transcript so the regular path's
    // chat-conversion never has to know about hosted-MCP item types.
    modified_request.input = ResponseInput::Items(lower_transcript(combined));
    modified_request.previous_response_id = None;
    modified_request.conversation = None;

    Ok(LoadedConversationHistory {
        request: modified_request,
        existing_mcp_list_tools_labels,
        control_items,
    })
}
