//! Shared helpers and state tracking for Regular Responses
//!
//! This module contains common utilities used by both streaming and non-streaming paths:
//! - ToolLoopState for tracking multi-turn tool calling
//! - Helper functions for tool preparation and extraction
//! - MCP metadata builders
//! - Conversation history loading

use axum::{http, response::Response};
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{Tool, ToolChoice, ToolChoiceValue},
    responses::{
        self, ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseOutputItem,
        ResponsesRequest,
    },
};
use smg_data_connector::{
    self as data_connector, ConversationId, ResponseId, ResponseStorageError,
};
use smg_mcp::McpToolSession;
use tracing::{debug, warn};

use crate::{
    middleware::TenantRequestMeta,
    routers::{
        common::persistence_utils::{
            count_conversation_turn_info, split_stored_message_content, ConversationTurnInfo,
        },
        error,
        grpc::common::responses::ResponsesContext,
    },
};

// ============================================================================
// Tool Loop State
// ============================================================================

/// State for tracking multi-turn tool calling loop
pub(super) struct ToolLoopState {
    pub iteration: usize,
    pub total_calls: usize,
    pub conversation_history: Vec<ResponseInputOutputItem>,
    pub original_input: ResponseInput,
    pub mcp_call_items: Vec<ResponseOutputItem>,
}

/// Per-request parameters for chat pipeline execution.
/// Bundles values that are always threaded together through the regular responses call chain.
pub(super) struct ResponsesCallContext {
    pub headers: Option<http::HeaderMap>,
    pub model_id: String,
    pub response_id: Option<String>,
    pub tenant_request_meta: TenantRequestMeta,
}

/// Loaded request bundle for Regular Responses path.
pub(super) struct LoadedRequest {
    pub request: ResponsesRequest,
    pub turn_info: Option<ConversationTurnInfo>,
}

impl ToolLoopState {
    pub fn new(original_input: ResponseInput) -> Self {
        Self {
            iteration: 0,
            total_calls: 0,
            conversation_history: Vec::new(),
            original_input,
            mcp_call_items: Vec::new(),
        }
    }

    pub fn record_call(
        &mut self,
        call_id: String,
        tool_name: String,
        args_json_str: String,
        output_str: String,
        output_item: ResponseOutputItem,
        _success: bool,
    ) {
        // Add function_tool_call item with both arguments and output
        let id = call_id.clone();
        self.conversation_history
            .push(ResponseInputOutputItem::FunctionToolCall {
                id,
                call_id,
                name: tool_name,
                arguments: args_json_str,
                output: Some(output_str),
                status: Some("completed".to_string()),
            });

        // Add transformed output item (respects tool's response_format)
        self.mcp_call_items.push(output_item);
    }
}

// ============================================================================
// Tool Preparation and Extraction
// ============================================================================

/// Merge function tools from request with MCP tools and set tool_choice based on iteration
pub(super) fn prepare_chat_tools_and_choice(
    chat_request: &mut ChatCompletionRequest,
    mcp_chat_tools: &[Tool],
    iteration: usize,
) {
    // Merge function tools from request with MCP tools
    let mut all_tools = chat_request.tools.take().unwrap_or_default();
    all_tools.extend(mcp_chat_tools.iter().cloned());
    chat_request.tools = Some(all_tools);

    // Set tool_choice based on iteration
    // - Iteration 0: Use user's tool_choice or default to auto
    // - Iteration 1+: Always use auto to avoid infinite loops
    chat_request.tool_choice = if iteration == 0 {
        chat_request
            .tool_choice
            .take()
            .or(Some(ToolChoice::Value(ToolChoiceValue::Auto)))
    } else {
        Some(ToolChoice::Value(ToolChoiceValue::Auto))
    };
}

/// Tool call extracted from a ChatCompletionResponse
#[derive(Debug, Clone)]
pub(super) struct ExtractedToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

/// Extract all tool calls from chat response (for parallel tool call support)
pub(super) fn extract_all_tool_calls_from_chat(
    response: &openai_protocol::chat::ChatCompletionResponse,
) -> Vec<ExtractedToolCall> {
    // Check if response has choices with tool calls
    let Some(choice) = response.choices.first() else {
        return Vec::new();
    };
    let message = &choice.message;

    // Look for tool_calls in the message
    if let Some(tool_calls) = &message.tool_calls {
        tool_calls
            .iter()
            .map(|tool_call| ExtractedToolCall {
                call_id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                arguments: tool_call
                    .function
                    .arguments
                    .clone()
                    .unwrap_or_else(|| "{}".to_string()),
            })
            .collect()
    } else {
        Vec::new()
    }
}

pub(super) fn convert_mcp_tools_to_chat_tools(session: &McpToolSession<'_>) -> Vec<Tool> {
    session.build_chat_function_tools()
}

// ============================================================================
// Conversation History Loading
// ============================================================================

/// Load conversation history and response chains, returning modified request
pub(super) async fn load_conversation_history(
    ctx: &ResponsesContext,
    request: &ResponsesRequest,
    stm_enabled: bool,
) -> Result<LoadedRequest, Response> {
    let mut modified_request = request.clone();
    let mut conversation_items: Option<Vec<ResponseInputOutputItem>> = None;
    // Tracks the raw DB item count (all types) for the conversation path so
    // that total_items in ConversationTurnInfo is not undercounted when
    // function_call/function_call_output/MCP items are present in storage
    // but filtered out of the inference window.
    let mut raw_stored_item_count: Option<usize> = None;

    // Handle previous_response_id by loading response chain
    if let Some(ref prev_id_str) = modified_request.previous_response_id {
        let prev_id = ResponseId::from(prev_id_str.as_str());
        match ctx
            .persistence
            .response_storage
            .get_response_chain(&prev_id, None)
            .await
        {
            Ok(chain) if !chain.responses.is_empty() => {
                let mut items = Vec::new();
                for stored in &chain.responses {
                    // Convert input items from stored input (which is now a JSON array)
                    if let Some(input_arr) = stored.input.as_array() {
                        for item in input_arr {
                            match serde_json::from_value::<ResponseInputOutputItem>(item.clone()) {
                                Ok(input_item) => {
                                    items.push(input_item);
                                }
                                Err(e) => {
                                    warn!(
                                        "Failed to deserialize stored input item: {}. Item: {}",
                                        e, item
                                    );
                                }
                            }
                        }
                    }

                    // Convert output items from stored raw_response["output"] (which is a JSON array)
                    if let Some(output_arr) =
                        stored.raw_response.get("output").and_then(|v| v.as_array())
                    {
                        for item in output_arr {
                            match serde_json::from_value::<ResponseInputOutputItem>(item.clone()) {
                                Ok(output_item) => {
                                    items.push(output_item);
                                }
                                Err(e) => {
                                    warn!(
                                        "Failed to deserialize stored output item: {}. Item: {}",
                                        e, item
                                    );
                                }
                            }
                        }
                    }
                }
                conversation_items = Some(items);
                modified_request.previous_response_id = None;
            }
            Ok(_) | Err(ResponseStorageError::ResponseNotFound(_)) => {
                return Err(error::bad_request(
                    "previous_response_not_found",
                    format!("Previous response with id '{prev_id_str}' not found."),
                ));
            }
            Err(e) => {
                return Err(error::internal_error(
                    "load_previous_response_chain_failed",
                    format!("Failed to load previous response chain for {prev_id_str}: {e}"),
                ));
            }
        }
    }

    // Handle conversation by loading conversation history
    if let Some(ref conv_ref) = request.conversation {
        let conv_id_str = conv_ref.as_id();
        let conv_id = ConversationId::from(conv_id_str);

        // Check if conversation exists - return error if not found
        let conversation = ctx
            .persistence
            .conversation_storage
            .get_conversation(&conv_id)
            .await
            .map_err(|e| {
                error::internal_error(
                    "check_conversation_failed",
                    format!("Failed to check conversation: {e}"),
                )
            })?;

        if conversation.is_none() {
            return Err(error::not_found(
                "conversation_not_found",
                format!(
                    "Conversation '{conv_id_str}' not found. Please create the conversation first using the conversations API."
                )
            ));
        }

        // Load conversation history.
        // When STMO is active we request one extra item so we can detect
        // whether the conversation has grown past max_conversation_history_items.
        // If it has, turn-count math would be wrong, so we reject early.
        let cap = ctx.max_conversation_history_items;
        let fetch_limit = if stm_enabled {
            cap.saturating_add(1)
        } else {
            cap
        };
        let params = data_connector::ListParams {
            limit: fetch_limit,
            order: data_connector::SortOrder::Asc,
            after: None,
        };
        match ctx
            .persistence
            .conversation_item_storage
            .list_items(&conv_id, params)
            .await
        {
            Ok(stored_items) => {
                if stm_enabled && stored_items.len() > cap {
                    return Err(error::bad_request(
                        "conversation_too_large",
                        format!(
                            "Conversation exceeds the configured limit of {cap} history items. \
                             Increase max_conversation_history_items in the router config \
                             or reduce conversation length before using short-term memory \
                             optimization.",
                        ),
                    ));
                }
                raw_stored_item_count = Some(stored_items.len());
                let mut items: Vec<ResponseInputOutputItem> = Vec::new();
                for item in stored_items {
                    if item.item_type == "message" {
                        // Stored content may be either the raw content array
                        // (legacy) or an object `{content: [...], phase: ...}`
                        // when the message carried a phase label (P3).
                        let (content_value, stored_phase) =
                            split_stored_message_content(item.content.clone());
                        if let Ok(content_parts) =
                            serde_json::from_value::<Vec<ResponseContentPart>>(content_value)
                        {
                            items.push(ResponseInputOutputItem::Message {
                                id: item.id.0.clone(),
                                role: item.role.clone().unwrap_or_else(|| "user".to_string()),
                                content: content_parts,
                                status: item.status.clone(),
                                phase: stored_phase,
                            });
                        }
                    }
                }

                // Append current request
                match &modified_request.input {
                    ResponseInput::Text(text) => {
                        items.push(ResponseInputOutputItem::Message {
                            id: format!("msg_u_{}", conv_id.0),
                            role: "user".to_string(),
                            content: vec![ResponseContentPart::InputText { text: text.clone() }],
                            status: Some("completed".to_string()),
                            phase: None,
                        });
                    }
                    ResponseInput::Items(current_items) => {
                        // Process all item types, converting SimpleInputMessage to Message
                        for item in current_items {
                            let normalized = responses::normalize_input_item(item);
                            items.push(normalized);
                        }
                    }
                }

                modified_request.input = ResponseInput::Items(items);
            }
            Err(e) => {
                warn!("Failed to load conversation history: {}", e);
            }
        }
    }

    // If we have conversation_items from previous_response_id, merge them
    if let Some(mut items) = conversation_items {
        // Append current request
        match &modified_request.input {
            ResponseInput::Text(text) => {
                items.push(ResponseInputOutputItem::Message {
                    id: format!(
                        "msg_u_{}",
                        request.previous_response_id.as_deref().unwrap_or("new")
                    ),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText { text: text.clone() }],
                    status: Some("completed".to_string()),
                    phase: None,
                });
            }
            ResponseInput::Items(current_items) => {
                // Process all item types, converting SimpleInputMessage to Message
                for item in current_items {
                    let normalized = responses::normalize_input_item(item);
                    items.push(normalized);
                }
            }
        }

        modified_request.input = ResponseInput::Items(items);
    }

    debug!(
        has_previous_response = request.previous_response_id.is_some(),
        has_conversation = request.conversation.is_some(),
        "Loaded conversation history"
    );

    let turn_info = if stm_enabled {
        // If a conversation was requested but list_items failed, the assembled
        // input only contains the current request — STMO turn counts would be
        // wrong. Skip STMO for this request so persistence does not enqueue a
        // job with an undercounted last_index/target_item_end.
        if request.conversation.is_some() && raw_stored_item_count.is_none() {
            None
        } else {
            let mut info = count_conversation_turn_info(&modified_request.input);
            // If we loaded from conversation storage, total_items from the
            // assembled (message-only) input undercounts — function_call,
            // function_call_output, and MCP items are in the DB but filtered
            // out of the inference window. Use the raw DB count instead so
            // target_item_end points at the correct absolute position.
            if let Some(raw_count) = raw_stored_item_count {
                // Only apply the raw-count correction when no response chain
                // was also loaded. If previous_response_id was set, the chain
                // merge ran last and overwrote modified_request.input with the
                // full replayed history — count_conversation_turn_info already
                // saw every item, so no correction is needed. (conversation and
                // previous_response_id are mutually exclusive in the API, but
                // we guard here for safety.)
                if request.previous_response_id.is_none() {
                    let current_input_count = match &request.input {
                        ResponseInput::Text(_) => 1,
                        ResponseInput::Items(items) => items.len(),
                    };
                    info.total_items = raw_count + current_input_count;
                }
            }
            Some(info)
        }
    } else {
        None
    };

    Ok(LoadedRequest {
        request: modified_request,
        turn_info,
    })
}

/// Build next request with updated conversation history
pub(super) fn build_next_request(
    state: &ToolLoopState,
    current_request: ResponsesRequest,
) -> ResponsesRequest {
    // Start with original input
    let mut input_items = match &state.original_input {
        ResponseInput::Text(text) => vec![ResponseInputOutputItem::Message {
            id: format!("msg_u_{}", state.iteration),
            role: "user".to_string(),
            content: vec![ResponseContentPart::InputText { text: text.clone() }],
            status: Some("completed".to_string()),
            phase: None,
        }],
        ResponseInput::Items(items) => items.iter().map(responses::normalize_input_item).collect(),
    };

    // Append all conversation history (function calls and outputs)
    input_items.extend_from_slice(&state.conversation_history);

    // Build new request for next iteration, moving fields from old request to avoid cloning
    ResponsesRequest {
        input: ResponseInput::Items(input_items),
        model: current_request.model,
        instructions: current_request.instructions,
        tools: current_request.tools,
        max_output_tokens: current_request.max_output_tokens,
        temperature: current_request.temperature,
        top_p: current_request.top_p,
        stream: current_request.stream,
        store: Some(false), // Don't store intermediate responses
        background: Some(false),
        max_tool_calls: current_request.max_tool_calls,
        tool_choice: current_request.tool_choice,
        parallel_tool_calls: current_request.parallel_tool_calls,
        previous_response_id: None,
        conversation: None,
        user: current_request.user,
        metadata: current_request.metadata,
        include: current_request.include,
        reasoning: current_request.reasoning,
        service_tier: current_request.service_tier,
        top_logprobs: current_request.top_logprobs,
        truncation: current_request.truncation,
        text: current_request.text,
        request_id: None,
        priority: current_request.priority,
        frequency_penalty: current_request.frequency_penalty,
        presence_penalty: current_request.presence_penalty,
        stop: current_request.stop,
        // Responses API top-level fields (P2): propagate per-request knobs so
        // multi-turn tool-loop continuations keep the same prompt template,
        // cache key, safety identifier, streaming options, and context-
        // management config as the original request.
        prompt: current_request.prompt,
        prompt_cache_key: current_request.prompt_cache_key,
        prompt_cache_retention: current_request.prompt_cache_retention,
        safety_identifier: current_request.safety_identifier,
        stream_options: current_request.stream_options,
        context_management: current_request.context_management,
        top_k: current_request.top_k,
        min_p: current_request.min_p,
        repetition_penalty: current_request.repetition_penalty,
    }
}
