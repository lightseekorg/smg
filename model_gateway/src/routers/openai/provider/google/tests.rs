use serde_json::json;

use super::{super::Provider, GoogleProvider};
use crate::worker::Endpoint;

#[test]
fn responses_transform_does_not_emit_top_level_stream_for_google() {
    let provider = GoogleProvider;
    let mut payload = json!({
        "model": "google.gemini-2.5-flash",
        "stream": true,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "hello"}
                ]
            }
        ]
    });

    provider
        .transform_request(&mut payload, Endpoint::Responses)
        .expect("transform should succeed");

    assert!(
        payload.get("stream").is_none(),
        "google generateContent payload must not contain top-level stream"
    );
    assert!(payload.get("contents").is_some());
}

#[test]
fn responses_transform_maps_tool_choice_required_to_any_mode() {
    let provider = GoogleProvider;
    let mut payload = json!({
        "model": "google.gemini-2.5-flash",
        "input": [{"role":"user","content":[{"type":"input_text","text":"hi"}]}],
        "tool_choice": "required"
    });

    provider
        .transform_request(&mut payload, Endpoint::Responses)
        .expect("transform should succeed");

    assert_eq!(
        payload
            .get("toolConfig")
            .and_then(|v| v.get("functionCallingConfig"))
            .and_then(|v| v.get("mode"))
            .and_then(|v| v.as_str()),
        Some("ANY")
    );
}

#[test]
fn gemini_upstream_url_requires_dp_supplied_url() {
    let provider = GoogleProvider;
    let err = provider
        .upstream_url("http://unused-worker", None)
        .expect_err("gemini must reject missing upstream URL");
    assert!(
        err.to_string().contains("upstream URL is required"),
        "unexpected error: {err}"
    );
}

#[test]
fn gemini_upstream_url_uses_dp_supplied_url_as_is() {
    let provider = GoogleProvider;
    let url = provider
        .upstream_url(
            "http://unused-worker",
            Some("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"),
        )
        .expect("gemini URL should be accepted");
    assert_eq!(
        url,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    );
}

#[test]
fn stream_transform_generates_text_delta_and_completion_event() {
    let provider = GoogleProvider;
    let event = json!({
        "candidates": [{
            "content": {"parts": [{"text": "hello"}]},
            "finishReason": "STOP"
        }]
    });

    let mapped = provider
        .transform_stream_event(&event, None, Endpoint::Responses)
        .expect("stream transform should succeed");

    assert!(mapped
        .iter()
        .any(|e| e.get("type").and_then(|v| v.as_str()) == Some("response.output_text.delta")));
    assert!(mapped
        .iter()
        .any(|e| e.get("type").and_then(|v| v.as_str()) == Some("response.completed")));
}

#[test]
fn stream_transform_emits_lifecycle_and_done_events() {
    let provider = GoogleProvider;
    let event = json!({
        "candidates": [{
            "content": {"parts": [{"text": "world"}]},
            "finishReason": "STOP"
        }]
    });

    let mapped = provider
        .transform_stream_event(&event, None, Endpoint::Responses)
        .expect("stream transform should succeed");

    assert!(mapped
        .iter()
        .any(|e| e.get("type").and_then(|v| v.as_str()) == Some("response.created")));
    assert!(mapped
        .iter()
        .any(|e| e.get("type").and_then(|v| v.as_str()) == Some("response.in_progress")));
    assert!(mapped
        .iter()
        .any(|e| e.get("type").and_then(|v| v.as_str()) == Some("response.output_item.added")));
    assert!(mapped
        .iter()
        .any(|e| e.get("type").and_then(|v| v.as_str()) == Some("response.content_part.added")));
    assert!(mapped
        .iter()
        .any(|e| e.get("type").and_then(|v| v.as_str()) == Some("response.output_text.done")));
    assert!(mapped
        .iter()
        .any(|e| e.get("type").and_then(|v| v.as_str()) == Some("response.output_item.done")));
}

#[test]
fn response_transform_prioritizes_content_filter_over_max_tokens() {
    let provider = GoogleProvider;
    let mut response = json!({
        "modelVersion": "gemini-2.5-flash",
        "createTime": "2026-04-20T10:20:30Z",
        "candidates": [
            {"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "x"}] }},
            {"finishReason": "SAFETY", "content": {"parts": [{"text": "y"}] }}
        ],
        "usageMetadata": {
          "promptTokenCount": 10,
          "candidatesTokenCount": 2,
          "thoughtsTokenCount": 0,
          "totalTokenCount": 12
        }
    });

    provider
        .transform_response(&mut response, Endpoint::Responses)
        .expect("response transform should succeed");

    assert_eq!(
        response.get("status").and_then(|v| v.as_str()),
        Some("incomplete")
    );
    assert_eq!(
        response
            .get("incomplete_details")
            .and_then(|v| v.get("reason"))
            .and_then(|v| v.as_str()),
        Some("content_filter")
    );
}

#[test]
fn response_transform_marks_failed_when_candidates_missing() {
    let provider = GoogleProvider;
    let mut response = json!({
        "modelVersion": "gemini-2.5-flash",
        "createTime": "2026-04-20T10:20:30Z",
        "usageMetadata": {
          "promptTokenCount": 10,
          "candidatesTokenCount": 0,
          "thoughtsTokenCount": 0,
          "totalTokenCount": 10
        }
    });

    provider
        .transform_response(&mut response, Endpoint::Responses)
        .expect("response transform should succeed");

    assert_eq!(
        response.get("status").and_then(|v| v.as_str()),
        Some("failed")
    );
}

#[test]
fn request_transform_ignores_input_file_with_file_id_only() {
    let provider = GoogleProvider;
    let mut payload = json!({
        "model": "google.gemini-2.5-flash",
        "input": [{
            "role": "user",
            "content": [
                {"type":"input_file","file_id":"file_only_123"}
            ]
        }]
    });

    provider
        .transform_request(&mut payload, Endpoint::Responses)
        .expect("transform should succeed");

    let contents = payload
        .get("contents")
        .and_then(|v| v.as_array())
        .expect("contents should exist");
    assert!(
        contents.is_empty(),
        "file_id-only input_file should be ignored (no implicit fetch in SMG)"
    );
}

#[test]
fn request_transform_defaults_inline_file_mime_type_when_missing() {
    let provider = GoogleProvider;
    let mut payload = json!({
        "model": "google.gemini-2.5-flash",
        "input": [{
            "role": "user",
            "content": [
                {"type":"input_file","file_data":"dGVzdA=="}
            ]
        }]
    });

    provider
        .transform_request(&mut payload, Endpoint::Responses)
        .expect("transform should succeed");

    assert_eq!(
        payload["contents"][0]["parts"][0]["inlineData"]["mimeType"],
        json!("application/octet-stream")
    );
    assert_eq!(
        payload["contents"][0]["parts"][0]["inlineData"]["data"],
        json!("dGVzdA==")
    );
}

#[test]
fn response_transform_maps_function_response_to_call_id_and_output() {
    let provider = GoogleProvider;
    let mut response = json!({
        "modelVersion": "gemini-2.5-flash",
        "createTime": "2026-04-20T10:20:30Z",
        "candidates": [{
            "finishReason": "STOP",
            "content": {
                "parts": [{
                    "functionResponse": {
                        "id": "call_abc123",
                        "response": {"temp": 72}
                    }
                }]
            }
        }],
        "usageMetadata": {
          "promptTokenCount": 10,
          "candidatesTokenCount": 1,
          "thoughtsTokenCount": 0,
          "totalTokenCount": 11
        }
    });

    provider
        .transform_response(&mut response, Endpoint::Responses)
        .expect("response transform should succeed");

    let output = response["output"].as_array().expect("output array");
    let function_output = output
        .iter()
        .find(|item| item.get("type").and_then(|v| v.as_str()) == Some("function_call_output"))
        .expect("function_call_output item");

    assert_eq!(function_output["call_id"], json!("call_abc123"));
    assert_eq!(function_output["output"], json!({"temp": 72}));
    assert!(function_output.get("tool_call_id").is_none());
    assert!(function_output.get("content").is_none());
}

#[test]
fn request_transform_maps_tool_response_to_function_response() {
    let provider = GoogleProvider;
    let mut payload = json!({
        "model": "google.gemini-2.5-flash",
        "input": [{
            "type": "tool_response",
            "role": "tool",
            "name": "weather_api",
            "content": {"result": "sunny"}
        }]
    });

    provider
        .transform_request(&mut payload, Endpoint::Responses)
        .expect("transform should succeed");

    let part = &payload["contents"][0]["parts"][0]["functionResponse"];
    assert_eq!(part["name"], json!("weather_api"));
    assert_eq!(part["response"], json!({"result": "sunny"}));
}

#[test]
fn request_transform_parses_json_strings_for_function_call_and_output() {
    let provider = GoogleProvider;
    let mut payload = json!({
        "model": "google.gemini-2.5-flash",
        "input": [
            {"type":"function_call","name":"weather","arguments":"{\"city\":\"sf\"}"},
            {"type":"function_call_output","name":"weather","output":"{\"temp\":72}"}
        ]
    });

    provider
        .transform_request(&mut payload, Endpoint::Responses)
        .expect("transform should succeed");

    assert_eq!(
        payload["contents"][0]["parts"][0]["functionCall"]["args"],
        json!({"city":"sf"})
    );
    assert_eq!(
        payload["contents"][1]["parts"][0]["functionResponse"]["response"],
        json!({"temp":72})
    );
}

#[test]
fn request_transform_preserves_non_assistant_roles() {
    let provider = GoogleProvider;
    let mut payload = json!({
        "model": "google.gemini-2.5-flash",
        "input": [{
            "role": "tool",
            "content": [{"type":"output_text","text":"ack"}]
        }]
    });

    provider
        .transform_request(&mut payload, Endpoint::Responses)
        .expect("transform should succeed");

    assert_eq!(payload["contents"][0]["role"], json!("tool"));
}

#[test]
fn request_transform_accepts_single_object_input() {
    let provider = GoogleProvider;
    let mut payload = json!({
        "model": "google.gemini-2.5-flash",
        "input": {
            "role": "user",
            "content": [{"type":"input_text","text":"hello"}]
        }
    });

    provider
        .transform_request(&mut payload, Endpoint::Responses)
        .expect("transform should succeed");

    assert_eq!(payload["contents"][0]["role"], json!("user"));
    assert_eq!(payload["contents"][0]["parts"][0]["text"], json!("hello"));
}

#[test]
fn request_transform_accepts_plain_string_input() {
    let provider = GoogleProvider;
    let mut payload = json!({
        "model": "google.gemini-2.5-flash-lite",
        "input": "Tell me a three sentence bedtime story about a unicorn.",
        "store": false
    });

    provider
        .transform_request(&mut payload, Endpoint::Responses)
        .expect("transform should succeed");

    assert_eq!(payload["model"], json!("gemini-2.5-flash-lite"));
    assert_eq!(payload["contents"][0]["role"], json!("user"));
    assert_eq!(
        payload["contents"][0]["parts"][0]["text"],
        json!("Tell me a three sentence bedtime story about a unicorn.")
    );
    assert!(payload.get("store").is_none());
}

#[test]
fn response_transform_sets_completed_at_and_falls_back_total_tokens_from_components() {
    let provider = GoogleProvider;
    let mut response = json!({
        "modelVersion": "gemini-2.5-flash",
        "createTime": "2026-04-20T10:20:30Z",
        "candidates": [{"finishReason":"STOP","content":{"parts":[{"text":"ok"}]}}],
        "usageMetadata": {
          "promptTokenCount": 10,
          "candidatesTokenCount": 2,
          "thoughtsTokenCount": 1
        }
    });

    provider
        .transform_response(&mut response, Endpoint::Responses)
        .expect("response transform should succeed");

    assert!(response.get("completed_at").is_some());
    assert_eq!(
        response["usage"]["total_tokens"].as_i64(),
        Some(13),
        "missing totalTokenCount should fall back to input + output tokens"
    );
}

#[test]
fn stream_transform_stateful_text_accumulation_and_single_lifecycle() {
    let provider = GoogleProvider;
    let mut state = provider.new_stream_state().expect("google stream state");

    let chunk1 = json!({
        "candidates": [{
            "content": {"role":"model","parts":[{"text":"Hello"}]}
        }]
    });
    let events1 = provider
        .transform_stream_event(&chunk1, Some(state.as_mut()), Endpoint::Responses)
        .expect("chunk1 transform");
    assert_eq!(
        events1
            .iter()
            .filter(|e| e.get("type").and_then(|v| v.as_str()) == Some("response.created"))
            .count(),
        1
    );
    assert!(events1
        .iter()
        .any(|e| e.get("type").and_then(|v| v.as_str()) == Some("response.output_text.delta")));

    let chunk2 = json!({
        "candidates": [{
            "content": {"role":"model","parts":[{"text":" world"}]}
        }]
    });
    let events2 = provider
        .transform_stream_event(&chunk2, Some(state.as_mut()), Endpoint::Responses)
        .expect("chunk2 transform");
    assert_eq!(
        events2
            .iter()
            .filter(|e| e.get("type").and_then(|v| v.as_str()) == Some("response.created"))
            .count(),
        0
    );
    assert_eq!(events2.len(), 1);

    let final_chunk = json!({
        "candidates": [{
            "content": {"role":"model","parts":[{"text":""}]},
            "finishReason":"STOP"
        }]
    });
    let final_events = provider
        .transform_stream_event(&final_chunk, Some(state.as_mut()), Endpoint::Responses)
        .expect("final transform");
    let completed = final_events
        .iter()
        .find(|e| e.get("type").and_then(|v| v.as_str()) == Some("response.completed"))
        .expect("completed event");
    let text_done = final_events
        .iter()
        .find(|e| e.get("type").and_then(|v| v.as_str()) == Some("response.output_text.done"))
        .expect("text done event");
    assert_eq!(text_done["text"], json!("Hello world"));
    assert_eq!(completed["response"]["status"], json!("completed"));
}

#[test]
fn stream_transform_stateful_function_call_only_parity() {
    let provider = GoogleProvider;
    let mut state = provider.new_stream_state().expect("google stream state");
    let function_chunk = json!({
        "candidates": [{
            "content": {"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"location":"SF"}}}]}
        }]
    });

    let events = provider
        .transform_stream_event(&function_chunk, Some(state.as_mut()), Endpoint::Responses)
        .expect("function chunk transform");
    assert_eq!(events.len(), 4);
    assert_eq!(events[2]["output_index"], json!(0));
    assert_eq!(events[3]["output_index"], json!(0));

    let completion = json!({
        "candidates": [{
            "content": {"role":"model","parts":[{"text":""}]},
            "finishReason":"STOP"
        }]
    });
    let done_events = provider
        .transform_stream_event(&completion, Some(state.as_mut()), Endpoint::Responses)
        .expect("completion transform");
    assert_eq!(done_events.len(), 3);
    assert_eq!(
        done_events[0]["type"],
        json!("response.function_call_arguments.done")
    );
    assert_eq!(done_events[1]["type"], json!("response.output_item.done"));
    assert_eq!(done_events[2]["type"], json!("response.completed"));
    assert_eq!(
        done_events[2]["response"]["output"][0]["type"],
        json!("function_call")
    );
}

#[test]
fn stream_transform_stateful_text_then_function_output_indices() {
    let provider = GoogleProvider;
    let mut state = provider.new_stream_state().expect("google stream state");

    let text_chunk = json!({
        "candidates": [{
            "content": {"role":"model","parts":[{"text":"let me check"}]}
        }]
    });
    let _ = provider
        .transform_stream_event(&text_chunk, Some(state.as_mut()), Endpoint::Responses)
        .expect("text transform");

    let function_chunk = json!({
        "candidates": [{
            "content": {"role":"model","parts":[{"functionCall":{"name":"tool","args":{"k":"v"}}}]}
        }]
    });
    let function_events = provider
        .transform_stream_event(&function_chunk, Some(state.as_mut()), Endpoint::Responses)
        .expect("function transform");
    assert_eq!(function_events[0]["output_index"], json!(1));
    assert_eq!(function_events[1]["output_index"], json!(1));
}

#[test]
fn stream_transform_incomplete_details_contains_reason_only() {
    let provider = GoogleProvider;
    let mut state = provider.new_stream_state().expect("google stream state");
    let final_chunk = json!({
        "candidates": [{
            "content": {"role":"model","parts":[{"text":"partial"}]},
            "finishReason":"MAX_TOKENS"
        }]
    });

    let final_events = provider
        .transform_stream_event(&final_chunk, Some(state.as_mut()), Endpoint::Responses)
        .expect("final transform");
    let incomplete = final_events
        .iter()
        .find(|e| e.get("type").and_then(|v| v.as_str()) == Some("response.incomplete"))
        .expect("incomplete event");

    assert_eq!(
        incomplete["response"]["incomplete_details"]["reason"],
        json!("max_output_tokens")
    );
    assert!(
        incomplete["response"]["incomplete_details"]
            .get("type")
            .is_none(),
        "incomplete_details should only contain reason"
    );
}
