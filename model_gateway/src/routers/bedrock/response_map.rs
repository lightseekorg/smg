use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConverseResponse {
    #[serde(default)]
    output: Option<ConverseOutput>,
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ConverseOutput {
    message: ConverseMessage,
}

#[derive(Debug, Deserialize)]
struct ConverseMessage {
    #[serde(default)]
    content: Vec<ConverseContent>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConverseContent {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    tool_use: Option<ToolUse>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolUse {
    tool_use_id: String,
    name: String,
    #[serde(default)]
    input: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Usage {
    #[serde(default)]
    input_tokens: Option<u32>,
    #[serde(default)]
    output_tokens: Option<u32>,
    #[serde(default)]
    total_tokens: Option<u32>,
}

pub(crate) fn map_non_stream_response(raw: &[u8], model: &str) -> Result<Value, serde_json::Error> {
    let parsed: ConverseResponse = serde_json::from_slice(raw)?;
    let created = now_epoch();

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    if let Some(output) = parsed.output.as_ref() {
        for block in &output.message.content {
            if let Some(t) = block.text.as_deref() {
                text.push_str(t);
            }
            if let Some(tu) = block.tool_use.as_ref() {
                // OpenAI expects `function.arguments` as a JSON-encoded string.
                let arguments =
                    serde_json::to_string(&tu.input).unwrap_or_else(|_| "{}".to_string());
                tool_calls.push(json!({
                    "id": tu.tool_use_id,
                    "type": "function",
                    "function": {
                        "name": tu.name,
                        "arguments": arguments,
                    },
                }));
            }
        }
    }

    let usage = parsed.usage.unwrap_or(Usage {
        input_tokens: Some(0),
        output_tokens: Some(0),
        total_tokens: Some(0),
    });
    let prompt_tokens = usage.input_tokens.unwrap_or(0);
    let completion_tokens = usage.output_tokens.unwrap_or(0);
    let total_tokens = usage
        .total_tokens
        .unwrap_or_else(|| prompt_tokens.saturating_add(completion_tokens));

    let mut message = json!({
        "role": "assistant",
        "content": if text.is_empty() && !tool_calls.is_empty() {
            Value::Null
        } else {
            Value::String(text)
        },
    });
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }

    Ok(json!({
        "id": format!("chatcmpl-bedrock-{}", created),
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": map_stop_reason(parsed.stop_reason.as_deref().unwrap_or("end_turn")),
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    }))
}

fn map_stop_reason(stop: &str) -> &'static str {
    match stop {
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        "stop_sequence" | "end_turn" => "stop",
        "content_filtered" | "guardrail_intervened" => "content_filter",
        _ => "stop",
    }
}

fn now_epoch() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
