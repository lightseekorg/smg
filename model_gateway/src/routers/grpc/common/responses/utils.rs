//! Utility functions for /v1/responses endpoint

use std::sync::Arc;

use axum::{http::HeaderMap, response::Response};
use openai_protocol::{
    common::Tool,
    responses::{ResponseTool, ResponsesRequest, ResponsesResponse},
};
use serde_json::to_value;
use smg_data_connector::{
    ConversationItemStorage, ConversationStorage, RequestContext as StorageRequestContext,
    ResponseStorage,
};
use smg_mcp::{McpOrchestrator, McpServerBinding};
use tracing::{debug, error, warn};

use crate::{
    routers::{
        common::{
            mcp_utils::ensure_request_mcp_client, persistence_utils::persist_conversation_items,
        },
        error,
    },
    worker::WorkerRegistry,
};

/// Ensure MCP connection succeeds if MCP tools or builtin tools are declared.
///
/// Checks if the request declares MCP tools or builtin tool types
/// (`web_search_preview`, `code_interpreter`, `image_generation`) and,
/// if so, validates that the MCP clients can be created and connected.
///
/// Returns Ok((has_mcp_tools, mcp_servers)) on success.
pub(crate) async fn ensure_mcp_connection(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    tools: Option<&[ResponseTool]>,
) -> Result<(bool, Vec<McpServerBinding>), Response> {
    // Check for explicit MCP tools (must error if connection fails)
    let has_explicit_mcp_tools = tools
        .map(|t| t.iter().any(|tool| matches!(tool, ResponseTool::Mcp(_))))
        .unwrap_or(false);

    // Check for builtin tools that MAY have MCP routing configured.
    //
    // `ImageGeneration` is included here because gpt-oss via the
    // harmony pipeline, and Qwen/Llama via the regular pipeline, both
    // dispatch hosted `image_generation` calls through the same MCP
    // routing path — the only difference is how the tool is advertised in
    // the prompt. Without this arm, the short-circuit below would return
    // `(false, Vec::new())`, the MCP loop would never be entered, and the
    // registered `image_generation` MCP server would receive zero
    // dispatches.
    let has_builtin_tools = tools
        .map(|t| {
            t.iter().any(|tool| {
                matches!(
                    tool,
                    ResponseTool::WebSearchPreview(_)
                        | ResponseTool::CodeInterpreter(_)
                        | ResponseTool::ImageGeneration(_)
                )
            })
        })
        .unwrap_or(false);

    // Only process if we have MCP or builtin tools
    if !has_explicit_mcp_tools && !has_builtin_tools {
        return Ok((false, Vec::new()));
    }

    if let Some(tools) = tools {
        // TODO: Thread real request headers through the gRPC responses path if/when
        // gRPC MCP flows need the same forwarded-header preservation contract.
        match ensure_request_mcp_client(mcp_orchestrator, tools).await {
            Some(mcp_servers) => {
                return Ok((true, mcp_servers));
            }
            None => {
                // No MCP servers available
                if has_explicit_mcp_tools {
                    // Explicit MCP tools MUST have working connections
                    error!(
                        function = "ensure_mcp_connection",
                        "Failed to connect to MCP servers"
                    );
                    return Err(error::failed_dependency(
                        "connect_mcp_server_failed",
                        "Failed to connect to MCP servers. Check server_url and authorization.",
                    ));
                }
                // Builtin tools without MCP routing - pass through to model
                debug!(
                    function = "ensure_mcp_connection",
                    "No MCP routing configured for builtin tools, passing through to model"
                );
                return Ok((false, Vec::new()));
            }
        }
    }

    Ok((false, Vec::new()))
}

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
#[expect(
    clippy::too_many_arguments,
    reason = "interceptor hook firing requires per-request metadata alongside storage handles"
)]
pub(crate) async fn persist_response_if_needed(
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    response: &ResponsesResponse,
    original_request: &ResponsesRequest,
    request_context: Option<StorageRequestContext>,
    interceptors: smg_extensions::InterceptorRegistry,
    headers: HeaderMap,
    request_id: String,
    tenant_id: Option<String>,
) {
    if !original_request.store.unwrap_or(true) {
        return;
    }

    if let Ok(response_json) = to_value(response) {
        let item_storage_for_hook = conversation_item_storage.clone();
        let persist_result = persist_conversation_items(
            conversation_storage,
            conversation_item_storage,
            response_storage,
            &response_json,
            original_request,
            request_context.clone(),
        )
        .await;
        match persist_result {
            Ok(()) => {
                debug!("Persisted response: {}", response.id);
                if !interceptors.is_empty() {
                    use smg_extensions::{AfterPersistCtx, RequestMetadata};

                    use crate::routers::common::turn_info::compute_turn_info;

                    let conv_id_opt: Option<smg_data_connector::ConversationId> =
                        original_request
                            .conversation
                            .as_ref()
                            .filter(|c| !c.is_empty())
                            .map(|c| smg_data_connector::ConversationId::from(c.as_id()));

                    let response_id_owned: Option<smg_data_connector::ResponseId> = response_json
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(smg_data_connector::ResponseId::from);

                    let incoming_input_value = to_value(&original_request.input).ok();
                    let turn_info = compute_turn_info(
                        item_storage_for_hook.as_ref(),
                        conv_id_opt.as_ref(),
                        incoming_input_value.as_ref(),
                    )
                    .await;

                    let metadata = RequestMetadata::build_from(
                        request_id,
                        original_request.safety_identifier.clone(),
                        tenant_id,
                        request_context,
                    );

                    let persisted_ids: &[smg_data_connector::ConversationItemId] = &[];
                    let after_ctx = AfterPersistCtx::new(
                        &headers,
                        original_request,
                        Some(&response_json),
                        response_id_owned.as_ref(),
                        conv_id_opt.as_ref(),
                        turn_info,
                        persisted_ids,
                        &metadata,
                    );
                    interceptors.run_after_persist(&after_ctx).await;
                }
            }
            Err(e) => {
                warn!("Failed to persist response: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use async_trait::async_trait;
    use axum::http::HeaderMap;
    use openai_protocol::responses::{ResponsesRequest, ResponsesResponse};
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    };
    use smg_extensions::{
        AfterPersistCtx, BeforeModelCtx, InterceptorRegistry, ResponsesInterceptor,
    };

    use super::persist_response_if_needed;

    struct CountingInterceptor {
        after_count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ResponsesInterceptor for CountingInterceptor {
        fn name(&self) -> &'static str {
            "counting-grpc"
        }
        async fn before_model(&self, _ctx: &mut BeforeModelCtx<'_>) {}
        async fn after_persist(&self, _ctx: &AfterPersistCtx<'_>) {
            self.after_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Shared gRPC after_persist site fires the registered interceptor when
    /// persistence succeeds. Mirrors the wiring used by both the harmony
    /// and regular gRPC routers.
    #[tokio::test]
    async fn persist_response_if_needed_fires_after_persist_hook() {
        let after_count = Arc::new(AtomicUsize::new(0));

        let mut builder = InterceptorRegistry::builder();
        builder.register(Arc::new(CountingInterceptor {
            after_count: after_count.clone(),
        }));
        let interceptors = builder.build();

        let conversation_storage = Arc::new(MemoryConversationStorage::new());
        let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());
        let response_storage = Arc::new(MemoryResponseStorage::new());

        let response = ResponsesResponse::builder("resp_grpc_unit", "mock-model").build();
        let request = ResponsesRequest::default();

        persist_response_if_needed(
            conversation_storage,
            conversation_item_storage,
            response_storage,
            &response,
            &request,
            None,
            interceptors,
            HeaderMap::new(),
            "req_grpc_unit".to_string(),
            Some("test-tenant".to_string()),
        )
        .await;

        assert_eq!(
            after_count.load(Ordering::SeqCst),
            1,
            "after_persist hook should fire exactly once on the gRPC shared persist path"
        );
    }

    /// Empty registry must not invoke any interceptor work.
    #[tokio::test]
    async fn persist_response_if_needed_skips_when_registry_is_empty() {
        let conversation_storage = Arc::new(MemoryConversationStorage::new());
        let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());
        let response_storage = Arc::new(MemoryResponseStorage::new());

        let response = ResponsesResponse::builder("resp_grpc_unit_empty", "mock-model").build();
        let request = ResponsesRequest::default();

        // Should be a no-op; just assert no panic and no observable side
        // effect on the storage backends from interceptor logic.
        persist_response_if_needed(
            conversation_storage,
            conversation_item_storage,
            response_storage,
            &response,
            &request,
            None,
            InterceptorRegistry::default(),
            HeaderMap::new(),
            "req_grpc_unit_empty".to_string(),
            None,
        )
        .await;
    }
}
