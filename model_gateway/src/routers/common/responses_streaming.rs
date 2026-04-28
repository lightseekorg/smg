//! Shared OpenAI-compatible SSE infrastructure for `/v1/responses`.

use axum::{body::Body, http::StatusCode, response::Response};
use bytes::Bytes;
use http::header::{HeaderValue, CONTENT_TYPE};
use openai_protocol::{
    chat::ChatCompletionStreamResponse,
    common::Usage,
    event_types::{
        CodeInterpreterCallEvent, ContentPartEvent, FileSearchCallEvent, FunctionCallEvent,
        ImageGenerationCallEvent, McpEvent, OutputItemEvent, OutputTextEvent, ResponseEvent,
        WebSearchCallEvent,
    },
    responses::{
        InputTokensDetails, OutputTokensDetails, ResponseOutputItem, ResponseStatus, ResponseUsage,
        ResponsesRequest, ResponsesResponse, ResponsesUsage,
    },
};
use serde_json::{json, Value};
use smg_mcp::{self as mcp};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::warn;
use uuid::Uuid;

use crate::routers::common::agent_loop::OutputFamily;

pub(crate) enum OutputItemType {
    Message,
    McpListTools,
    McpCall,
    McpApprovalRequest,
    FunctionCall,
    Reasoning,
    WebSearchCall,
    CodeInterpreterCall,
    FileSearchCall,
    ImageGenerationCall,
}

/// Status of an output item
#[derive(Debug, Clone, PartialEq)]
enum ItemStatus {
    InProgress,
    Completed,
}

/// State tracking for a single output item
#[derive(Debug, Clone)]
struct OutputItemState {
    output_index: usize,
    status: ItemStatus,
    item_data: Option<Value>,
}

/// OpenAI-compatible event emitter for /v1/responses streaming
///
/// Manages state and sequence numbers to emit proper event types:
/// - response.created
/// - response.in_progress
/// - response.output_item.added
/// - response.output_item.done
/// - response.completed
/// - response.content_part.added
/// - response.content_part.done
/// - response.output_text.delta
/// - response.output_text.done
/// - response.mcp_list_tools.in_progress
/// - response.mcp_list_tools.completed
/// - response.mcp_call.in_progress
/// - response.mcp_call_arguments.delta
/// - response.mcp_call_arguments.done
/// - response.mcp_call.completed
/// - response.mcp_call.failed
/// - response.web_search_call.in_progress
/// - response.web_search_call.searching
/// - response.web_search_call.completed
/// - response.function_call_arguments.delta
/// - response.function_call_arguments.done
pub(crate) struct ResponseStreamEventEmitter {
    sequence_number: u64,
    pub response_id: String,
    model: String,
    created_at: u64,
    message_id: String,
    accumulated_text: String,
    has_emitted_output_item_added: bool,
    has_emitted_content_part_added: bool,
    // Output item tracking
    output_items: Vec<OutputItemState>,
    next_output_index: usize,
    current_message_output_index: Option<usize>,
    current_item_id: Option<String>,
    current_reasoning_output_index: Option<usize>,
    current_reasoning_item_id: Option<String>,
    accumulated_reasoning_text: String,
    has_emitted_reasoning_summary_part_added: bool,
    original_request: Option<ResponsesRequest>,
}

// Streaming adapters share these emitter primitives and each uses a
// different subset.
#[expect(
    dead_code,
    reason = "primitives kept on the impl for future surface adapters"
)]
impl ResponseStreamEventEmitter {
    pub fn new(response_id: String, model: String, created_at: u64) -> Self {
        let message_id = format!("msg_{}", Uuid::now_v7());

        Self {
            sequence_number: 0,
            response_id,
            model,
            created_at,
            message_id,
            accumulated_text: String::new(),
            has_emitted_output_item_added: false,
            has_emitted_content_part_added: false,
            output_items: Vec::new(),
            next_output_index: 0,
            current_message_output_index: None,
            current_item_id: None,
            current_reasoning_output_index: None,
            current_reasoning_item_id: None,
            accumulated_reasoning_text: String::new(),
            has_emitted_reasoning_summary_part_added: false,
            original_request: None,
        }
    }

    /// Set the original request for including all fields in response.completed
    pub fn set_original_request(&mut self, request: ResponsesRequest) {
        self.original_request = Some(request);
    }

    pub fn next_sequence(&mut self) -> u64 {
        let seq = self.sequence_number;
        self.sequence_number += 1;
        seq
    }

    pub fn emit_created(&mut self) -> Value {
        json!({
            "type": ResponseEvent::CREATED,
            "sequence_number": self.next_sequence(),
            "response": {
                "id": self.response_id,
                "object": "response",
                "created_at": self.created_at,
                "status": "in_progress",
                "model": self.model,
                "output": []
            }
        })
    }

    pub fn emit_in_progress(&mut self) -> Value {
        json!({
            "type": ResponseEvent::IN_PROGRESS,
            "sequence_number": self.next_sequence(),
            "response": {
                "id": self.response_id,
                "object": "response",
                "status": "in_progress"
            }
        })
    }

    pub fn emit_content_part_added(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> Value {
        self.has_emitted_content_part_added = true;
        json!({
            "type": ContentPartEvent::ADDED,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "part": {
                "type": "output_text",
                "text": ""
            }
        })
    }

    pub fn emit_text_delta(
        &mut self,
        delta: &str,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> Value {
        self.accumulated_text.push_str(delta);
        json!({
            "type": OutputTextEvent::DELTA,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "delta": delta,
            "obfuscation": null
        })
    }

    pub fn emit_text_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> Value {
        json!({
            "type": OutputTextEvent::DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "text": self.accumulated_text.clone()
        })
    }

    pub fn emit_content_part_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> Value {
        json!({
            "type": ContentPartEvent::DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "part": {
                "type": "output_text",
                "text": self.accumulated_text.clone()
            }
        })
    }

    // INVARIANT: this method is terminal — it drains internal state via `take()`
    // and must only be called once per emitter lifetime.
    pub fn emit_completed(&mut self, usage: Option<&Value>) -> Value {
        // Build output array from tracked items
        let output: Vec<Value> = self
            .output_items
            .iter_mut()
            .filter_map(|item| {
                if item.status == ItemStatus::Completed {
                    item.item_data.take()
                } else {
                    None
                }
            })
            .collect();

        // If no items were tracked, fall back to a generic message.
        let output = if output.is_empty() {
            vec![json!({
                "id": std::mem::take(&mut self.message_id),
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": std::mem::take(&mut self.accumulated_text)
                }]
            })]
        } else {
            output
        };

        // Build base response object
        let mut response_obj = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "completed",
            "model": self.model,
            "output": output
        });

        // Add usage if provided
        if let Some(usage_val) = usage {
            response_obj["usage"] = usage_val.clone();
        }

        // Mirror the non-streaming `ResponsesResponse` shape so streaming
        // `response.completed.response` echoes the same canonical-field set.
        if let Some(ref req) = self.original_request {
            response_obj["instructions"] = json!(req.instructions);
            response_obj["max_output_tokens"] = json!(req.max_output_tokens);
            response_obj["max_tool_calls"] = json!(req.max_tool_calls);
            response_obj["previous_response_id"] = json!(req.previous_response_id);
            // OpenAI Responses always echoes `conversation` as
            // `{ "id": "conv_..." }`. The request's `ConversationRef`
            // accepts either a bare string or the `Object` form via
            // untagged serde — normalize the echo to the canonical
            // object shape so the wire response is shape-stable
            // regardless of which input form the client sent.
            response_obj["conversation"] = match req.conversation.as_ref() {
                Some(conv) => json!({ "id": conv.as_id() }),
                None => Value::Null,
            };
            response_obj["reasoning"] = json!(req.reasoning);
            response_obj["temperature"] = json!(req.temperature);
            response_obj["top_p"] = json!(req.top_p);
            response_obj["truncation"] = json!(req.truncation);
            response_obj["user"] = json!(req.user);
            response_obj["background"] = json!(req.background);
            response_obj["frequency_penalty"] = json!(req.frequency_penalty);
            response_obj["presence_penalty"] = json!(req.presence_penalty);
            response_obj["service_tier"] = json!(req.service_tier);
            response_obj["prompt_cache_key"] = json!(req.prompt_cache_key);
            response_obj["prompt_cache_retention"] = json!(req.prompt_cache_retention);
            response_obj["top_logprobs"] = json!(req.top_logprobs);
            response_obj["text"] = json!(req.text);
            response_obj["safety_identifier"] = json!(req.safety_identifier);

            response_obj["parallel_tool_calls"] = json!(req.parallel_tool_calls.unwrap_or(true));
            response_obj["store"] = json!(req.store.unwrap_or(true));
            let empty_tools = vec![];
            let empty_metadata = Default::default();
            response_obj["tools"] = json!(req.tools.as_ref().unwrap_or(&empty_tools));
            response_obj["metadata"] = json!(req.metadata.as_ref().unwrap_or(&empty_metadata));

            // tool_choice: serialize if present, otherwise use "auto"
            if let Some(ref tc) = req.tool_choice {
                response_obj["tool_choice"] = json!(tc);
            } else {
                response_obj["tool_choice"] = json!("auto");
            }
        }

        // Pure response-side fields the gateway does not populate today
        // but that the OpenAI Responses NS body always carries as `null`.
        // Emit them here too so streaming stays shape-aligned with NS.
        response_obj["billing"] = Value::Null;
        response_obj["moderation"] = Value::Null;
        response_obj["completed_at"] = Value::Null;
        response_obj["error"] = Value::Null;
        response_obj["incomplete_details"] = Value::Null;

        json!({
            "type": ResponseEvent::COMPLETED,
            "sequence_number": self.next_sequence(),
            "response": response_obj
        })
    }

    /// Convert tool entries to JSON values using the shared `build_mcp_tool_infos` bridge.
    fn tool_entries_to_json(tools: &[mcp::ToolEntry]) -> Result<Vec<Value>, serde_json::Error> {
        mcp::build_mcp_tool_infos(tools)
            .into_iter()
            .map(serde_json::to_value)
            .collect()
    }

    // ========================================================================
    // MCP Event Emission Methods
    // ========================================================================

    pub fn emit_mcp_list_tools_in_progress(&mut self, output_index: usize, item_id: &str) -> Value {
        json!({
            "type": McpEvent::LIST_TOOLS_IN_PROGRESS,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id
        })
    }

    pub fn emit_mcp_list_tools_completed(
        &mut self,
        output_index: usize,
        item_id: &str,
        tool_items: &[Value],
    ) -> Value {
        json!({
            "type": McpEvent::LIST_TOOLS_COMPLETED,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "tools": tool_items
        })
    }

    pub fn emit_mcp_call_arguments_delta(
        &mut self,
        output_index: usize,
        item_id: &str,
        delta: &str,
    ) -> Value {
        json!({
            "type": McpEvent::CALL_ARGUMENTS_DELTA,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "delta": delta,
            "obfuscation": null
        })
    }

    pub fn emit_mcp_call_arguments_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        arguments: &str,
    ) -> Value {
        json!({
            "type": McpEvent::CALL_ARGUMENTS_DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "arguments": arguments
        })
    }

    pub fn emit_mcp_call_failed(
        &mut self,
        output_index: usize,
        item_id: &str,
        error: &str,
    ) -> Value {
        json!({
            "type": McpEvent::CALL_FAILED,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "error": error
        })
    }

    // ========================================================================
    // Generic Tool Call Event Emission (based on OutputFamily)
    // ========================================================================

    /// Emit a tool call event with the specified event type.
    /// This is the internal helper used by all tool call event methods.
    fn emit_tool_event(
        &mut self,
        event_type: &'static str,
        output_index: usize,
        item_id: &str,
    ) -> Value {
        json!({
            "type": event_type,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id
        })
    }

    /// Emit the appropriate in_progress event based on output family.
    ///
    /// Returns `None` for `Function` (caller-declared function tools)
    /// because the OpenAI wire spec carries `function_call`'s
    /// in-progress state only on `output_item.added`'s `status`
    /// field, not as a separate event. Hosted-builtin and `McpCall`
    /// families return `Some(event)`.
    pub fn emit_tool_call_in_progress(
        &mut self,
        output_index: usize,
        item_id: &str,
        family: OutputFamily,
    ) -> Option<Value> {
        let event_type = match family {
            OutputFamily::WebSearchCall => WebSearchCallEvent::IN_PROGRESS,
            OutputFamily::CodeInterpreterCall => CodeInterpreterCallEvent::IN_PROGRESS,
            OutputFamily::FileSearchCall => FileSearchCallEvent::IN_PROGRESS,
            OutputFamily::ImageGenerationCall => ImageGenerationCallEvent::IN_PROGRESS,
            OutputFamily::McpCall => McpEvent::CALL_IN_PROGRESS,
            OutputFamily::Function => return None,
        };
        Some(self.emit_tool_event(event_type, output_index, item_id))
    }

    /// Emit the searching/interpreting/generating event for builtin tool calls (no-op for passthrough).
    ///
    /// For `image_generation_call` this emits the `generating` event. The
    /// partial-image event is emitted separately via `emit_image_generation_partial_image`
    /// because it carries additional payload (the partial b64 bytes) and is
    /// optional per the `partial_images` request field.
    pub fn emit_tool_call_searching(
        &mut self,
        output_index: usize,
        item_id: &str,
        family: OutputFamily,
    ) -> Option<Value> {
        let event_type = match family {
            OutputFamily::WebSearchCall => WebSearchCallEvent::SEARCHING,
            OutputFamily::CodeInterpreterCall => CodeInterpreterCallEvent::INTERPRETING,
            OutputFamily::FileSearchCall => FileSearchCallEvent::SEARCHING,
            OutputFamily::ImageGenerationCall => ImageGenerationCallEvent::GENERATING,
            OutputFamily::McpCall | OutputFamily::Function => return None,
        };
        Some(self.emit_tool_event(event_type, output_index, item_id))
    }

    /// Emit a `response.image_generation_call.partial_image` event.
    ///
    /// Returns `None` when `response_format` is anything other than
    /// [`OutputFamily::ImageGenerationCall`], mirroring how
    /// `emit_tool_call_searching` gates on format. The payload carries the
    /// base64-encoded partial image bytes plus a 0-based partial image index.
    ///
    /// Per-router wiring is responsible for deciding when to call this and
    /// how to source the partial-image bytes.
    #[expect(
        dead_code,
        reason = "partial_image emission is wired by per-router integrations"
    )]
    pub fn emit_image_generation_partial_image(
        &mut self,
        output_index: usize,
        item_id: &str,
        family: OutputFamily,
        partial_image_index: u32,
        partial_image_b64: &str,
    ) -> Option<Value> {
        if !matches!(family, OutputFamily::ImageGenerationCall) {
            return None;
        }
        Some(json!({
            "type": ImageGenerationCallEvent::PARTIAL_IMAGE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "partial_image_index": partial_image_index,
            "partial_image_b64": partial_image_b64
        }))
    }

    /// Emit the appropriate completed event based on output family.
    ///
    /// Returns `None` for `Function` because `function_call` has no
    /// `*_completed` event — its completion is signalled only by
    /// `output_item.done`. Hosted-builtin and `McpCall` return
    /// `Some(event)`.
    pub fn emit_tool_call_completed(
        &mut self,
        output_index: usize,
        item_id: &str,
        family: OutputFamily,
    ) -> Option<Value> {
        let event_type = match family {
            OutputFamily::WebSearchCall => WebSearchCallEvent::COMPLETED,
            OutputFamily::CodeInterpreterCall => CodeInterpreterCallEvent::COMPLETED,
            OutputFamily::FileSearchCall => FileSearchCallEvent::COMPLETED,
            OutputFamily::ImageGenerationCall => ImageGenerationCallEvent::COMPLETED,
            OutputFamily::McpCall => McpEvent::CALL_COMPLETED,
            OutputFamily::Function => return None,
        };
        Some(self.emit_tool_event(event_type, output_index, item_id))
    }

    /// Emit a streaming `*_arguments.delta` event keyed on the
    /// caller's [`OutputFamily`]. Returns `None` for families that
    /// don't stream arguments on the wire (hosted builtins surface
    /// progress through structured `*.in_progress` / `*.searching`
    /// events instead).
    pub fn emit_tool_call_arguments_delta(
        &mut self,
        output_index: usize,
        item_id: &str,
        delta: &str,
        family: OutputFamily,
    ) -> Option<Value> {
        match family {
            OutputFamily::Function => {
                Some(self.emit_function_call_arguments_delta(output_index, item_id, delta))
            }
            OutputFamily::McpCall => {
                Some(self.emit_mcp_call_arguments_delta(output_index, item_id, delta))
            }
            OutputFamily::WebSearchCall
            | OutputFamily::CodeInterpreterCall
            | OutputFamily::FileSearchCall
            | OutputFamily::ImageGenerationCall => None,
        }
    }

    /// Emit the closing `*_arguments.done` event keyed on
    /// [`OutputFamily`]. Mirrors [`emit_tool_call_arguments_delta`].
    pub fn emit_tool_call_arguments_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        arguments: &str,
        family: OutputFamily,
    ) -> Option<Value> {
        match family {
            OutputFamily::Function => {
                Some(self.emit_function_call_arguments_done(output_index, item_id, arguments))
            }
            OutputFamily::McpCall => {
                Some(self.emit_mcp_call_arguments_done(output_index, item_id, arguments))
            }
            OutputFamily::WebSearchCall
            | OutputFamily::CodeInterpreterCall
            | OutputFamily::FileSearchCall
            | OutputFamily::ImageGenerationCall => None,
        }
    }

    // ========================================================================
    // Helper Methods for OutputFamily
    // ========================================================================

    /// Map an [`OutputFamily`] (or absence) to the [`OutputItemType`]
    /// the per-index allocator uses. Falls back to `FunctionCall` when
    /// the caller has no family hint — caller-declared function tools
    /// are the only family that legitimately reach this fallback path.
    pub fn output_item_type_for_family(family: Option<OutputFamily>) -> OutputItemType {
        match family {
            Some(OutputFamily::WebSearchCall) => OutputItemType::WebSearchCall,
            Some(OutputFamily::CodeInterpreterCall) => OutputItemType::CodeInterpreterCall,
            Some(OutputFamily::FileSearchCall) => OutputItemType::FileSearchCall,
            Some(OutputFamily::ImageGenerationCall) => OutputItemType::ImageGenerationCall,
            Some(OutputFamily::McpCall) => OutputItemType::McpCall,
            Some(OutputFamily::Function) | None => OutputItemType::FunctionCall,
        }
    }

    // ========================================================================
    // Function Call Event Emission Methods
    // ========================================================================

    pub fn emit_function_call_arguments_delta(
        &mut self,
        output_index: usize,
        item_id: &str,
        delta: &str,
    ) -> Value {
        json!({
            "type": FunctionCallEvent::ARGUMENTS_DELTA,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "delta": delta,
            "obfuscation": null
        })
    }

    pub fn emit_function_call_arguments_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        arguments: &str,
    ) -> Value {
        json!({
            "type": FunctionCallEvent::ARGUMENTS_DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "arguments": arguments
        })
    }

    // ========================================================================
    // Output Item Wrapper Events
    // ========================================================================

    /// Emit response.output_item.added event
    pub fn emit_output_item_added(&mut self, output_index: usize, item: &Value) -> Value {
        json!({
            "type": OutputItemEvent::ADDED,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item": item
        })
    }

    /// Emit response.output_item.done event
    pub fn emit_output_item_done(&mut self, output_index: usize, item: &Value) -> Value {
        // Store the item data for later use in emit_completed
        self.store_output_item_data(output_index, item.clone());

        json!({
            "type": OutputItemEvent::DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item": item
        })
    }

    /// Generate unique ID for item type
    fn generate_item_id(prefix: &str) -> String {
        format!("{}_{}", prefix, Uuid::now_v7().simple())
    }

    /// Allocate next output index and track item
    pub fn allocate_output_index(&mut self, item_type: OutputItemType) -> (usize, String) {
        let index = self.next_output_index;
        self.next_output_index += 1;

        let id_prefix = match &item_type {
            OutputItemType::McpListTools => "mcpl",
            OutputItemType::McpCall => "mcp",
            // `mcpr_` matches OpenAI's documented prefix for
            // `mcp_approval_request` items.
            OutputItemType::McpApprovalRequest => "mcpr",
            OutputItemType::FunctionCall => "fc",
            OutputItemType::Message => "msg",
            OutputItemType::Reasoning => "rs",
            OutputItemType::WebSearchCall => "ws",
            OutputItemType::CodeInterpreterCall => "ci",
            OutputItemType::FileSearchCall => "fs",
            OutputItemType::ImageGenerationCall => "ig",
        };

        let id = Self::generate_item_id(id_prefix);

        self.output_items.push(OutputItemState {
            output_index: index,
            status: ItemStatus::InProgress,
            item_data: None,
        });

        (index, id)
    }

    pub fn next_output_index(&self) -> usize {
        self.next_output_index
    }

    pub fn advance_next_output_index_to(&mut self, next_output_index: usize) {
        self.next_output_index = self.next_output_index.max(next_output_index);
    }

    /// Mark output item as completed and store its data
    pub fn complete_output_item(&mut self, output_index: usize) {
        if let Some(item) = self
            .output_items
            .iter_mut()
            .find(|i| i.output_index == output_index)
        {
            item.status = ItemStatus::Completed;
        }
    }

    /// Store output item data when emitting output_item.done
    pub fn store_output_item_data(&mut self, output_index: usize, item_data: Value) {
        if let Some(item) = self
            .output_items
            .iter_mut()
            .find(|i| i.output_index == output_index)
        {
            item.item_data = Some(item_data);
        }
    }

    /// Finalize and return the complete ResponsesResponse
    ///
    /// This constructs the final ResponsesResponse from all accumulated output items
    /// for persistence. Should be called after streaming is complete.
    /// Reads non-destructively so `emit_completed()` can still drain state afterwards.
    pub fn finalize(&self, usage: Option<Usage>) -> ResponsesResponse {
        // Build output array from tracked items (clone — emit_completed drains later)
        let output: Vec<ResponseOutputItem> = self
            .output_items
            .iter()
            .filter_map(|item| {
                item.item_data
                    .as_ref()
                    .and_then(|data| serde_json::from_value(data.clone()).ok())
            })
            .collect();

        // Convert Usage to ResponsesUsage
        let responses_usage = usage.map(|u| {
            ResponsesUsage::Modern(ResponseUsage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                input_tokens_details: u
                    .prompt_tokens_details
                    .as_ref()
                    .map(InputTokensDetails::from),
                output_tokens_details: u.completion_tokens_details.as_ref().and_then(|d| {
                    d.reasoning_tokens.map(|tokens| OutputTokensDetails {
                        reasoning_tokens: tokens,
                    })
                }),
            })
        });

        // Build response using builder
        ResponsesResponse::builder(&self.response_id, &self.model)
            .created_at(self.created_at as i64)
            .status(ResponseStatus::Completed)
            .output(output)
            .maybe_copy_from_request(self.original_request.as_ref())
            .maybe_usage(responses_usage)
            .build()
    }

    pub fn process_reasoning_delta(
        &mut self,
        delta: &str,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        if delta.is_empty() {
            return Ok(());
        }

        if self.current_reasoning_item_id.is_none() {
            let (output_index, item_id) = self.allocate_output_index(OutputItemType::Reasoning);
            self.current_reasoning_output_index = Some(output_index);
            self.current_reasoning_item_id = Some(item_id.clone());

            // Match the OpenAI Responses cloud reasoning item shape:
            // only `id`, `type`, and `summary` are emitted (verified by
            // hitting the cloud across effort=minimal/medium/high and
            // summary=auto/detailed; `content` / `encrypted_content` /
            // `status` never appear unless the client opts into
            // `include: ["reasoning.encrypted_content"]`, which the
            // gateway does not relay anyway).
            let item = json!({
                "id": item_id,
                "type": "reasoning",
                "summary": [],
            });
            let event = self.emit_output_item_added(output_index, &item);
            self.send_event(&event, tx)?;
        }

        let Some(output_index) = self.current_reasoning_output_index else {
            return Ok(());
        };
        let Some(item_id) = self.current_reasoning_item_id.clone() else {
            return Ok(());
        };

        if !self.has_emitted_reasoning_summary_part_added {
            let event = json!({
                "type": "response.reasoning_summary_part.added",
                "sequence_number": self.next_sequence(),
                "output_index": output_index,
                "item_id": item_id,
                "summary_index": 0,
                "part": { "type": "summary_text", "text": "" }
            });
            self.send_event(&event, tx)?;
            self.has_emitted_reasoning_summary_part_added = true;
        }

        self.accumulated_reasoning_text.push_str(delta);
        // Cloud emits reasoning summary deltas under
        // `response.reasoning_summary_text.{delta,done}` keyed by
        // `summary_index`, not `response.reasoning_text.*` /
        // `content_index`. Use the cloud-aligned event family so
        // OpenAI-SDK clients pick up the deltas as standard reasoning
        // summary stream, and final `summary[]` is populated below in
        // `finish_reasoning_item` with the same accumulated text.
        let event = json!({
            "type": "response.reasoning_summary_text.delta",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "summary_index": 0,
            "delta": delta,
            "obfuscation": null
        });
        self.send_event(&event, tx)
    }

    pub fn finish_reasoning_item(
        &mut self,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        let (Some(output_index), Some(item_id)) = (
            self.current_reasoning_output_index,
            self.current_reasoning_item_id.clone(),
        ) else {
            return Ok(());
        };

        let text = std::mem::take(&mut self.accumulated_reasoning_text);
        let event = json!({
            "type": "response.reasoning_summary_text.done",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "summary_index": 0,
            "text": text.clone()
        });
        self.send_event(&event, tx)?;

        if self.has_emitted_reasoning_summary_part_added {
            let event = json!({
                "type": "response.reasoning_summary_part.done",
                "sequence_number": self.next_sequence(),
                "output_index": output_index,
                "item_id": item_id,
                "summary_index": 0,
                "part": { "type": "summary_text", "text": "" }
            });
            self.send_event(&event, tx)?;
        }

        // Final reasoning item carries the accumulated text inside
        // `summary[]` as a `summary_text` part, matching the cloud
        // shape (verified across multiple effort/summary configs).
        let item = json!({
            "id": item_id,
            "type": "reasoning",
            "summary": [
                { "type": "summary_text", "text": text }
            ],
        });
        let event = self.emit_output_item_done(output_index, &item);
        self.send_event(&event, tx)?;
        self.complete_output_item(output_index);

        self.current_reasoning_output_index = None;
        self.current_reasoning_item_id = None;
        self.has_emitted_reasoning_summary_part_added = false;
        Ok(())
    }

    /// Process a chunk and emit appropriate events
    pub fn process_chunk(
        &mut self,
        chunk: &ChatCompletionStreamResponse,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        // Process content if present
        if let Some(choice) = chunk.choices.first() {
            if let Some(reasoning) = &choice.delta.reasoning_content {
                self.process_reasoning_delta(reasoning, tx)?;
            }

            if let Some(content) = &choice.delta.content {
                if !content.is_empty() {
                    // Allocate output_index and item_id for this message item (once per message)
                    if self.current_item_id.is_none() {
                        let (output_index, item_id) =
                            self.allocate_output_index(OutputItemType::Message);

                        // Build message item structure
                        let item = json!({
                            "id": item_id,
                            "type": "message",
                            "role": "assistant",
                            "content": []
                        });

                        // Emit output_item.added
                        let event = self.emit_output_item_added(output_index, &item);
                        self.send_event(&event, tx)?;
                        self.has_emitted_output_item_added = true;

                        // Store for subsequent events
                        self.current_item_id = Some(item_id);
                        self.current_message_output_index = Some(output_index);
                    }

                    // output_index and item_id are always set in the block above
                    // when current_item_id was None and we allocated new ones
                    if let (Some(output_index), Some(item_id)) = (
                        self.current_message_output_index,
                        self.current_item_id.clone(),
                    ) {
                        let content_index = 0; // Single content part for now

                        // Emit content_part.added before first delta
                        if !self.has_emitted_content_part_added {
                            let event =
                                self.emit_content_part_added(output_index, &item_id, content_index);
                            self.send_event(&event, tx)?;
                            self.has_emitted_content_part_added = true;
                        }

                        // Emit text delta
                        let event =
                            self.emit_text_delta(content, output_index, &item_id, content_index);
                        self.send_event(&event, tx)?;
                    }
                }
            }

            // Check for finish_reason to emit completion events
            if let Some(reason) = &choice.finish_reason {
                self.finish_reasoning_item(tx)?;
                if reason == "stop" || reason == "length" {
                    if let (Some(output_index), Some(item_id)) = (
                        self.current_message_output_index,
                        self.current_item_id.clone(),
                    ) {
                        let content_index = 0;

                        // Emit closing events
                        if self.has_emitted_content_part_added {
                            let event = self.emit_text_done(output_index, &item_id, content_index);
                            self.send_event(&event, tx)?;
                            let event =
                                self.emit_content_part_done(output_index, &item_id, content_index);
                            self.send_event(&event, tx)?;
                        }

                        if self.has_emitted_output_item_added {
                            // Build complete message item for output_item.done
                            let item = json!({
                                "id": item_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "output_text",
                                    "text": std::mem::take(&mut self.accumulated_text)
                                }]
                            });
                            let event = self.emit_output_item_done(output_index, &item);
                            self.send_event(&event, tx)?;
                        }

                        // Mark item as completed
                        self.complete_output_item(output_index);
                    }
                }
            }
        }

        Ok(())
    }

    #[expect(
        clippy::unused_self,
        reason = "method on emitter for API consistency with send_event_best_effort"
    )]
    pub fn send_event(
        &self,
        event: &Value,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        let event_json =
            serde_json::to_string(event).map_err(|e| format!("Failed to serialize event: {e}"))?;

        // Extract event type from the JSON for SSE event field
        let event_type = event
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("message");

        // Format as SSE with event: field
        let sse_message = format!("event: {event_type}\ndata: {event_json}\n\n");

        if tx.send(Ok(Bytes::from(sse_message))).is_err() {
            return Err("Client disconnected".to_string());
        }

        Ok(())
    }

    /// Send event and log any errors (typically client disconnect)
    ///
    /// This is a convenience method for streaming scenarios where client
    /// disconnection is expected and should be logged but not fail the operation.
    /// Returns true if sent successfully, false if client disconnected.
    pub fn send_event_best_effort(
        &self,
        event: &Value,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    ) -> bool {
        match self.send_event(event, tx) {
            Ok(()) => true,
            Err(e) => {
                tracing::debug!("Failed to send event (likely client disconnect): {}", e);
                false
            }
        }
    }

    /// Emit an error event
    ///
    /// Creates and sends an error event with the given error message.
    /// Uses OpenAI's error event format.
    /// Use this for terminal errors that should abort the streaming response.
    pub fn emit_error(
        &mut self,
        error_msg: &str,
        error_code: Option<&str>,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    ) {
        let event = json!({
            "type": "error",
            "code": error_code.unwrap_or("internal_error"),
            "message": error_msg,
            "param": null,
            "sequence_number": self.next_sequence()
        });
        let sse_data = match serde_json::to_string(&event) {
            Ok(json) => format!("data: {json}\n\n"),
            Err(_) => "data: {\"type\":\"error\",\"code\":\"internal_error\",\"message\":\"serialization failed\",\"param\":null}\n\n".to_string(),
        };
        let _ = tx.send(Ok(Bytes::from(sse_data)));
    }

    /// Emit the full mcp_list_tools output-item sequence.
    ///
    /// Allocates an output index, builds the tool-list JSON, then emits the four
    /// standard events (output_item.added, mcp_list_tools.in_progress,
    /// mcp_list_tools.completed, output_item.done) and marks the item complete.
    ///
    /// `server_label` is taken as an explicit parameter so callers can iterate
    /// over multiple MCP servers without mutating the emitter's own label.
    pub fn emit_mcp_list_tools_sequence(
        &mut self,
        server_label: &str,
        tools: &[mcp::ToolEntry],
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        let (output_index, item_id) = self.allocate_output_index(OutputItemType::McpListTools);

        // Build per-tool JSON items
        let tool_items = Self::tool_entries_to_json(tools).unwrap_or_else(|e| {
            warn!("Failed to serialize McpToolInfo to JSON: {e}");
            Vec::new()
        });

        // In-progress item (empty tools)
        let item_in_progress = json!({
            "id": item_id,
            "type": "mcp_list_tools",
            "server_label": server_label,
            "status": "in_progress",
            "tools": []
        });

        // Emit output_item.added
        let event = self.emit_output_item_added(output_index, &item_in_progress);
        self.send_event(&event, tx)?;

        // Emit mcp_list_tools.in_progress
        let event = self.emit_mcp_list_tools_in_progress(output_index, &item_id);
        self.send_event(&event, tx)?;

        // Emit mcp_list_tools.completed
        let event = self.emit_mcp_list_tools_completed(output_index, &item_id, &tool_items);
        self.send_event(&event, tx)?;

        // Completed item (with tools populated)
        let item_done = json!({
            "id": item_id,
            "type": "mcp_list_tools",
            "server_label": server_label,
            "status": "completed",
            "tools": tool_items
        });

        // Emit output_item.done (also stores item data internally)
        let event = self.emit_output_item_done(output_index, &item_done);
        self.send_event(&event, tx)?;

        self.complete_output_item(output_index);

        Ok(())
    }
}

/// Build a Server-Sent Events (SSE) response
///
/// Creates a Response with proper SSE headers and streaming body.
#[expect(
    clippy::expect_used,
    reason = "Response::builder with static headers and valid status code is infallible"
)]
pub(crate) fn build_sse_response(
    rx: mpsc::UnboundedReceiver<Result<Bytes, std::io::Error>>,
) -> Response {
    let stream = UnboundedReceiverStream::new(rx);
    Response::builder()
        .status(StatusCode::OK)
        .header(
            CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream; charset=utf-8"),
        )
        .header("Cache-Control", HeaderValue::from_static("no-cache"))
        .header("Connection", HeaderValue::from_static("keep-alive"))
        .body(Body::from_stream(stream))
        .expect("infallible: static headers and valid status code")
}
