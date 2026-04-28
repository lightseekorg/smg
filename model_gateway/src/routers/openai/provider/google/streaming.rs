use std::any::Any;

use serde_json::{json, Value};

use super::GoogleProvider;
use crate::worker::Endpoint;

#[derive(Default, Clone)]
struct PendingFunctionCall {
    id: String,
    name: String,
    arguments: String,
    output_index: usize,
}

#[derive(Default)]
pub(super) struct GoogleStreamState {
    has_emitted_initial_events: bool,
    has_emitted_content_part_added: bool,
    assistant_message_id: String,
    text_content_part_id: String,
    response_id: String,
    accumulated_text: String,
    function_calls: Vec<PendingFunctionCall>,
    output_index: usize,
    message_item_added: bool,
}

impl GoogleStreamState {
    pub(super) fn new() -> Self {
        Self {
            has_emitted_initial_events: false,
            has_emitted_content_part_added: false,
            assistant_message_id: format!("msg_{}", uuid::Uuid::now_v7()),
            text_content_part_id: format!("part_{}", uuid::Uuid::now_v7()),
            response_id: format!("resp_{}", uuid::Uuid::now_v7()),
            accumulated_text: String::new(),
            function_calls: Vec::new(),
            output_index: 0,
            message_item_added: false,
        }
    }
}

impl GoogleProvider {
    fn map_stream_finish_reason(reason: Option<&str>) -> (&'static str, Option<&'static str>) {
        match reason.unwrap_or_default().to_ascii_uppercase().as_str() {
            "STOP" => ("completed", None),
            "MAX_TOKENS" => ("incomplete", Some("max_output_tokens")),
            "SAFETY" | "RECITATION" => ("incomplete", Some("content_filter")),
            _ => ("incomplete", Some("other")),
        }
    }

    fn transform_stream_chunk(event: &Value, state: &mut GoogleStreamState) -> Vec<Value> {
        let mut out = Vec::new();
        let Some(candidates) = event.get("candidates").and_then(Value::as_array) else {
            return out;
        };
        let Some(candidate) = candidates.first() else {
            return out;
        };

        if !state.has_emitted_initial_events {
            out.push(json!({
                "type": "response.created",
                "response": {
                    "id": state.response_id,
                    "object": "response",
                    "status": "in_progress",
                    "output": []
                }
            }));
            out.push(json!({
                "type": "response.in_progress",
                "response": {
                    "status": "in_progress"
                }
            }));
            state.has_emitted_initial_events = true;
        }

        if let Some(parts) = candidate
            .get("content")
            .and_then(|v| v.get("parts"))
            .and_then(Value::as_array)
        {
            for p in parts {
                if let Some(function_call) = p.get("functionCall").and_then(Value::as_object) {
                    let call_id =
                        format!("call_{}", uuid::Uuid::now_v7().to_string().replace('-', ""));
                    let function_name = function_call
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown_tool")
                        .to_string();
                    let args = function_call
                        .get("args")
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    let args_json =
                        serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
                    let base_idx = if state.message_item_added {
                        state.output_index + 1
                    } else {
                        state.output_index
                    };
                    let idx = base_idx + state.function_calls.len();
                    out.push(json!({
                        "type": "response.output_item.added",
                        "output_index": idx,
                        "item": {
                            "id": call_id,
                            "type": "function_call",
                            "name": function_name,
                            "arguments": "",
                            "call_id": call_id
                        }
                    }));
                    out.push(json!({
                        "type": "response.function_call_arguments.delta",
                        "output_index": idx,
                        "item_id": call_id,
                        "delta": args_json
                    }));
                    state.function_calls.push(PendingFunctionCall {
                        id: call_id,
                        name: function_name,
                        arguments: args_json,
                        output_index: idx,
                    });
                }

                if let Some(text) = p.get("text").and_then(Value::as_str) {
                    if text.is_empty() {
                        continue;
                    }
                    if !state.has_emitted_content_part_added {
                        out.push(json!({
                            "type": "response.output_item.added",
                            "output_index": state.output_index,
                            "item": {
                                "id": state.assistant_message_id,
                                "type": "message",
                                "role": "assistant",
                                "content": []
                            }
                        }));
                        out.push(json!({
                            "type": "response.content_part.added",
                            "output_index": state.output_index,
                            "content_index": 0,
                            "item_id": state.assistant_message_id,
                            "part": {
                                "id": state.text_content_part_id,
                                "type": "output_text",
                                "text": "",
                                "logprobs": []
                            }
                        }));
                        state.has_emitted_content_part_added = true;
                        state.message_item_added = true;
                    }
                    out.push(json!({
                        "type": "response.output_text.delta",
                        "output_index": state.output_index,
                        "item_id": state.assistant_message_id,
                        "content_index": 0,
                        "delta": text
                    }));
                    state.accumulated_text.push_str(text);
                }
            }
        }

        let finish_reason = candidate.get("finishReason").and_then(Value::as_str);
        if finish_reason.is_none() {
            return out;
        }

        for call in &state.function_calls {
            out.push(json!({
                "type": "response.function_call_arguments.done",
                "output_index": call.output_index,
                "item_id": call.id,
                "arguments": call.arguments
            }));
            out.push(json!({
                "type": "response.output_item.done",
                "output_index": call.output_index,
                "item": {
                    "id": call.id,
                    "type": "function_call",
                    "name": call.name,
                    "arguments": call.arguments,
                    "call_id": call.id,
                    "status": "completed"
                }
            }));
        }

        if !state.accumulated_text.is_empty() {
            out.push(json!({
                "type": "response.output_text.done",
                "output_index": state.output_index,
                "item_id": state.assistant_message_id,
                "content_index": 0,
                "text": state.accumulated_text
            }));
            out.push(json!({
                "type": "response.content_part.done",
                "output_index": state.output_index,
                "content_index": 0,
                "item_id": state.assistant_message_id,
                "part": {
                    "id": state.text_content_part_id,
                    "type": "output_text",
                    "text": state.accumulated_text,
                    "logprobs": []
                }
            }));
            out.push(json!({
                "type": "response.output_item.done",
                "output_index": state.output_index,
                "item": {
                    "id": state.assistant_message_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": state.accumulated_text,
                        "annotations": [],
                        "logprobs": []
                    }]
                }
            }));
        }

        let (status, incomplete_reason) = Self::map_stream_finish_reason(finish_reason);
        let mut response = json!({
            "id": state.response_id,
            "object": "response",
            "status": status,
        });

        let mut output = Vec::new();
        for call in &state.function_calls {
            output.push(json!({
                "id": call.id,
                "type": "function_call",
                "name": call.name,
                "arguments": call.arguments,
                "call_id": call.id,
                "status": "completed"
            }));
        }
        if !state.accumulated_text.is_empty() {
            output.push(json!({
                "id": state.assistant_message_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": state.accumulated_text,
                    "annotations": [],
                    "logprobs": []
                }]
            }));
        }
        response["output"] = Value::Array(output);

        if let Some(reason) = incomplete_reason {
            response["incomplete_details"] = json!({
                "reason": reason
            });
        }

        if let Some(usage) = event.get("usageMetadata").and_then(Value::as_object) {
            let input_tokens = usage
                .get("promptTokenCount")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let candidates_tokens = usage
                .get("candidatesTokenCount")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let thoughts_tokens = usage
                .get("thoughtsTokenCount")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let output_tokens = candidates_tokens + thoughts_tokens;
            let mut usage_out = serde_json::Map::new();
            usage_out.insert("input_tokens".to_string(), json!(input_tokens));
            usage_out.insert("output_tokens".to_string(), json!(output_tokens));
            if usage.contains_key("thoughtsTokenCount") {
                usage_out.insert(
                    "output_tokens_details".to_string(),
                    json!({ "reasoning_tokens": thoughts_tokens }),
                );
            }
            if let Some(total_tokens) = usage.get("totalTokenCount").and_then(Value::as_i64) {
                usage_out.insert("total_tokens".to_string(), json!(total_tokens));
            }
            response["usage"] = Value::Object(usage_out);
        }

        out.push(json!({
            "type": if incomplete_reason.is_some() { "response.incomplete" } else { "response.completed" },
            "response": response
        }));

        out
    }
    pub(super) fn transform_stream_event_with_state_impl(
        event: &Value,
        state: Option<&mut (dyn Any + Send)>,
        endpoint: Endpoint,
    ) -> Vec<Value> {
        if endpoint != Endpoint::Responses || event.get("type").is_some() {
            return vec![event.clone()];
        }
        if let Some(state) = state.and_then(|s| s.downcast_mut::<GoogleStreamState>()) {
            return Self::transform_stream_chunk(event, state);
        }
        let mut fallback = GoogleStreamState::new();
        Self::transform_stream_chunk(event, &mut fallback)
    }
}
