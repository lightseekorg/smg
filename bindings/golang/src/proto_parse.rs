//! Shared proto JSON parsing utilities

use serde_json::Value;
use smg::grpc_client::sglang_proto as proto;

/// Parse a JSON value into a proto::GenerateResponse
pub fn parse_proto_response(json_value: &Value) -> Result<proto::GenerateResponse, &'static str> {
    let mut proto_response = proto::GenerateResponse {
        request_id: json_value
            .get("request_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        response: None,
    };

    if let Some(chunk_json) = json_value.get("chunk") {
        proto_response.response = Some(proto::generate_response::Response::Chunk(parse_chunk(
            chunk_json,
        )));
    } else if let Some(complete_json) = json_value.get("complete") {
        proto_response.response = Some(proto::generate_response::Response::Complete(
            parse_complete(complete_json),
        ));
    } else if let Some(error_json) = json_value.get("error") {
        proto_response.response = Some(proto::generate_response::Response::Error(parse_error(
            error_json,
        )));
    } else {
        return Err("Response JSON must contain 'chunk', 'complete', or 'error' field");
    }

    Ok(proto_response)
}

/// Check if a proto response is a terminal response (complete or error)
pub fn is_terminal_response(json_value: &Value) -> bool {
    json_value.get("complete").is_some() || json_value.get("error").is_some()
}

fn parse_chunk(json: &Value) -> proto::GenerateStreamChunk {
    proto::GenerateStreamChunk {
        token_ids: json
            .get("token_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as u32))
                    .collect()
            })
            .unwrap_or_default(),
        prompt_tokens: json
            .get("prompt_tokens")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(0),
        completion_tokens: json
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(0),
        cached_tokens: json
            .get("cached_tokens")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(0),
        output_logprobs: None,
        hidden_states: vec![],
        input_logprobs: None,
        index: 0,
    }
}

fn parse_complete(json: &Value) -> proto::GenerateComplete {
    proto::GenerateComplete {
        output_ids: json
            .get("output_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as u32))
                    .collect()
            })
            .unwrap_or_default(),
        finish_reason: json
            .get("finish_reason")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        prompt_tokens: json
            .get("prompt_tokens")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(0),
        completion_tokens: json
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(0),
        cached_tokens: json
            .get("cached_tokens")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(0),
        output_logprobs: None,
        all_hidden_states: vec![],
        input_logprobs: None,
        matched_stop: None,
        index: 0,
    }
}

fn parse_error(json: &Value) -> proto::GenerateError {
    proto::GenerateError {
        message: json
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        http_status_code: json
            .get("http_status_code")
            .and_then(|v| v.as_str())
            .unwrap_or("500")
            .to_string(),
        details: json
            .get("details")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
    }
}
