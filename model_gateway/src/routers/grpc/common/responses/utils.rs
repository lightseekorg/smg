//! Utility functions for /v1/responses endpoint

use std::sync::Arc;

use axum::response::Response;
use openai_protocol::{
    common::{ConversationRef, Tool, Usage},
    responses::{ResponseStatus, ResponseTool, ResponsesRequest, ResponsesResponse},
};
use serde_json::{json, to_value};
use smg_data_connector::{
    ConversationItemStorage, ConversationStorage, RequestContext as StorageRequestContext,
    ResponseStorage,
};
use tracing::{debug, warn};

use super::agent_loop_sink::GrpcResponseStreamSink;
use crate::{
    routers::{
        common::{agent_loop::RenderMode, persistence_utils::persist_conversation_items},
        error,
    },
    worker::WorkerRegistry,
};

/// Validate that workers are available for the requested model
pub(crate) fn validate_worker_availability(
    worker_registry: &Arc<WorkerRegistry>,
    model: &str,
) -> Option<Response> {
    let available_models = worker_registry.get_models();

    if !available_models.contains(&model.to_string()) {
        return Some(error::model_not_found(model));
    }

    None
}

/// Extract function tools from ResponseTools
///
/// This utility consolidates the logic for extracting tools with schemas from ResponseTools.
/// It's used by both Harmony and Regular routers for different purposes:
///
/// - **Harmony router**: Extracts function tools because MCP tools are exposed to the model as
///   function tools (via `convert_mcp_tools_to_response_tools()`), and those are used to
///   generate structural constraints in the Harmony preparation stage.
///
/// - **Regular router**: Extracts function tools during the initial conversion from
///   ResponsesRequest to ChatCompletionRequest. MCP tools are merged later by the tool loop.
pub(crate) fn extract_tools_from_response_tools(
    response_tools: Option<&[ResponseTool]>,
) -> Vec<Tool> {
    let Some(tools) = response_tools else {
        return Vec::new();
    };

    tools
        .iter()
        .filter_map(|rt| match rt {
            ResponseTool::Function(ft) => Some(Tool {
                tool_type: "function".to_string(),
                function: ft.function.clone(),
            }),
            _ => None,
        })
        .collect()
}

/// Persist response to storage if store=true
///
/// Common helper function to avoid duplication across sync and streaming paths
/// in both harmony and regular responses implementations.
pub(crate) async fn persist_response_if_needed(
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    response: &ResponsesResponse,
    original_request: &ResponsesRequest,
    request_context: Option<StorageRequestContext>,
) {
    if !original_request.store.unwrap_or(true) {
        return;
    }

    if let Ok(response_json) = to_value(response) {
        if let Err(e) = persist_conversation_items(
            conversation_storage,
            conversation_item_storage,
            response_storage,
            &response_json,
            original_request,
            request_context,
        )
        .await
        {
            warn!("Failed to persist response: {}", e);
        } else {
            debug!("Persisted response: {}", response.id);
        }
    }
}

pub(crate) struct StreamingPersistHandles {
    pub(crate) conversation_storage: Arc<dyn ConversationStorage>,
    pub(crate) conversation_item_storage: Arc<dyn ConversationItemStorage>,
    pub(crate) response_storage: Arc<dyn ResponseStorage>,
    pub(crate) request_context: Option<StorageRequestContext>,
}

/// Finalize the SSE accumulator into the response record used for persistence.
///
/// Streaming paths emit wire events incrementally, so final persistence must
/// reuse the sink emitter's accumulated item ids/order rather than rebuilding
/// from loop primitives. The request metadata patch mirrors the non-streaming
/// builder contract.
pub(crate) async fn finalize_streamed_response_for_persist(
    sink: &mut GrpcResponseStreamSink,
    usage_for_persist: Option<Usage>,
    mode: &RenderMode,
    original_request: &ResponsesRequest,
    handles: StreamingPersistHandles,
) {
    let mut final_response = sink.emitter.finalize(usage_for_persist);
    final_response
        .previous_response_id
        .clone_from(&original_request.previous_response_id);
    final_response.conversation =
        original_request
            .conversation
            .as_ref()
            .map(|c| ConversationRef::Object {
                id: c.as_id().to_string(),
            });
    final_response.store = original_request.store.unwrap_or(true);
    if let RenderMode::Incomplete { reason, .. } = mode {
        final_response.status = ResponseStatus::Incomplete;
        final_response.incomplete_details = Some(json!({ "reason": reason }));
    }

    persist_response_if_needed(
        handles.conversation_storage,
        handles.conversation_item_storage,
        handles.response_storage,
        &final_response,
        original_request,
        handles.request_context,
    )
    .await;
}
