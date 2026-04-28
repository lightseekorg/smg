use std::any::Any;

use serde_json::{json, Value};

use super::GoogleProvider;
use crate::worker::Endpoint;

#[derive(Default)]
pub(super) struct GoogleStreamState {
    has_emitted_initial_events: bool,
    has_emitted_content_part_added: bool,
    assistant_message_id: String,
    text_content_part_id: String,
    response_id: String,
    function_call_id: Option<String>,
    current_function_name: Option<String>,
    accumulated_text: String,
    accumulated_function_args: String,
    has_function_call: bool,
    output_index: usize,
    function_call_output_index: Option<usize>,
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
            function_call_id: None,
            current_function_name: None,
            accumulated_text: String::new(),
            accumulated_function_args: String::new(),
            has_function_call: false,
            output_index: 0,
            function_call_output_index: None,
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
                    let call_id = state
                        .function_call_id
                        .get_or_insert_with(|| {
                            format!("call_{}", uuid::Uuid::now_v7().to_string().replace('-', ""))
                        })
                        .clone();
                    state.has_function_call = true;
                    let function_name = function_call
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown_tool")
                        .to_string();
                    state.current_function_name = Some(function_name.clone());
                    let args = function_call
                        .get("args")
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    let args_json =
                        serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
                    if state.function_call_output_index.is_none() {
                        let idx = if state.message_item_added {
                            state.output_index + 1
                        } else {
                            state.output_index
                        };
                        state.function_call_output_index = Some(idx);
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
                    }
                    if let Some(idx) = state.function_call_output_index {
                        out.push(json!({
                            "type": "response.function_call_arguments.delta",
                            "output_index": idx,
                            "item_id": call_id,
                            "delta": args_json
                        }));
                    }
                    state.accumulated_function_args.push_str(&args_json);
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

        if state.has_function_call {
            let call_id = state.function_call_id.clone().unwrap_or_else(|| {
                format!("call_{}", uuid::Uuid::now_v7().to_string().replace('-', ""))
            });
            let call_name = state
                .current_function_name
                .clone()
                .unwrap_or_else(|| "unknown_tool".to_string());
            let idx = state
                .function_call_output_index
                .unwrap_or(state.output_index);
            out.push(json!({
                "type": "response.function_call_arguments.done",
                "output_index": idx,
                "item_id": call_id,
                "arguments": state.accumulated_function_args
            }));
            out.push(json!({
                "type": "response.output_item.done",
                "output_index": idx,
                "item": {
                    "id": call_id,
                    "type": "function_call",
                    "name": call_name,
                    "arguments": state.accumulated_function_args,
                    "call_id": call_id,
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
        if state.has_function_call {
            output.push(json!({
                "id": state.function_call_id.clone().unwrap_or_default(),
                "type": "function_call",
                "name": state.current_function_name.clone().unwrap_or_else(|| "unknown_tool".to_string()),
                "arguments": state.accumulated_function_args,
                "call_id": state.function_call_id.clone().unwrap_or_default(),
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
