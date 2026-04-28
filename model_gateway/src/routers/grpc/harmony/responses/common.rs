//! Shared helpers for the gRPC Harmony Responses surface.
//!
//! This module keeps the Harmony-specific history loader and
//! image-generation tool normalization.
//!
//! - `load_previous_messages` — surface-side wrapper around the shared
//!   `load_request_history` primitive that resolves
//!   `previous_response_id` **or** `conversation` (mutually exclusive,
//!   validated upstream) and merges the lowered transcript into the
//!   request's input.
//! - `strip_image_generation_from_request_tools` — harmony-specific
//!   builder workaround for the
//!   [`openai_protocol::responses::ResponseTool::ImageGeneration`] tag
//!   when an MCP server has taken ownership of `image_generation`.

use std::collections::HashSet;

use axum::response::Response;
use openai_protocol::responses::{
    ResponseInput, ResponseInputOutputItem, ResponseTool, ResponsesRequest, StringOrContentParts,
};
use smg_mcp::{McpToolSession, ResponseFormat};
use tracing::debug;

use crate::routers::{
    common::transcript_lower::{
        extract_control_items, extract_mcp_list_tools_server_labels, lower_transcript,
    },
    grpc::common::responses::{load_request_history, ResponsesContext},
};

pub(super) struct LoadedResponsesHistory {
    pub request: ResponsesRequest,
    pub existing_mcp_list_tools_labels: HashSet<String>,
    /// Control items collected from both the resolved history and the
    /// caller's hand-stitched input. Drives approval-continuation
    /// validation at the loop entry — see
    /// [`crate::routers::common::agent_loop`] for the full set of
    /// control vocabulary types.
    pub control_items: Vec<ResponseInputOutputItem>,
}

/// Resolve `previous_response_id` or `conversation` (whichever is set)
/// and merge the resulting items in front of the request's input.
pub(super) async fn load_previous_messages(
    ctx: &ResponsesContext,
    request: ResponsesRequest,
) -> Result<LoadedResponsesHistory, Response> {
    let loaded = load_request_history(ctx, &request).await?;
    let mut existing_mcp_list_tools_labels = loaded.existing_mcp_list_tools_labels;
    let mut control_items = loaded.control_items;

    let mut modified_request = request;
    let mut history_items = loaded.items;

    let combined = match modified_request.input {
        ResponseInput::Items(items) => {
            // Hand-stitched input: a caller may inline a prior turn's
            // `mcp_list_tools` item directly into `input` instead of
            // resolving it from `previous_response_id` / `conversation`.
            // Treat that the same way as history-loaded items so the
            // agent loop does not re-emit a fresh listing for the same
            // server label downstream.
            existing_mcp_list_tools_labels.extend(extract_mcp_list_tools_server_labels(&items));
            // Pick up approval-pair / list_tools control items from the
            // current request body too. Both history and hand-stitched
            // paths feed the same `control_items` vec so the driver
            // entry only has to look in one place.
            control_items.extend(extract_control_items(&items));
            history_items.extend(items);
            history_items
        }
        ResponseInput::Text(text) => {
            history_items.push(ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String(text),
                role: "user".to_string(),
                r#type: None,
                phase: None,
            });
            history_items
        }
    };

    // Lower once over the combined transcript: hand-stitched
    // current-turn items get the same projection (mcp_call →
    // function_call pair, hosted-tool metadata dropped) as items
    // loaded from prev_response_id / conversation history.
    modified_request.input = ResponseInput::Items(lower_transcript(combined));
    // Subsequent in-loop sub-calls must not re-resolve history.
    modified_request.previous_response_id = None;
    modified_request.conversation = None;

    Ok(LoadedResponsesHistory {
        request: modified_request,
        existing_mcp_list_tools_labels,
        control_items,
    })
}

/// Strip `ResponseTool::ImageGeneration` once an MCP session exposes an
/// MCP-routed replacement.
///
/// The harmony builder synthesizes a function-tool description named
/// Once an MCP server takes ownership of `image_generation`, the MCP-exposed
/// function tool is the single advertisement the model should see.
pub(super) fn strip_image_generation_from_request_tools(
    request: &mut ResponsesRequest,
    session: &McpToolSession<'_>,
) {
    let mcp_has_image_generation = session
        .mcp_tools()
        .iter()
        .any(|entry| matches!(entry.response_format, ResponseFormat::ImageGenerationCall));
    if !mcp_has_image_generation {
        return;
    }
    if let Some(tools) = request.tools.as_mut() {
        let before = tools.len();
        tools.retain(|t| !matches!(t, ResponseTool::ImageGeneration(_)));
        let after = tools.len();
        if before != after {
            debug!(
                removed = before - after,
                "Stripped ResponseTool::ImageGeneration from request.tools because the \
                 MCP session exposes an image_generation-routed dispatcher",
            );
        }
    }
}
