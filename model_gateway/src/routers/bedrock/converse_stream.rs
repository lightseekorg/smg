//! Map Bedrock `ConverseStream` AWS event-stream frames to OpenAI-style SSE lines.

use std::time::{SystemTime, UNIX_EPOCH};

use bytes::{Bytes, BytesMut};
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use super::event_stream::{pop_next_event, DecodeError, StreamEvent};

fn now_epoch() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn map_bedrock_stop_to_openai(stop: &str) -> &'static str {
    match stop {
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        "stop_sequence" => "stop",
        "end_turn" => "stop",
        "content_filtered" => "content_filter",
        "guardrail_intervened" => "content_filter",
        _ => "stop",
    }
}

fn openai_chunk(
    id: &str,
    created: i64,
    model: &str,
    delta: Value,
    finish_reason: Value,
    usage: Option<Value>,
) -> String {
    let mut obj = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }]
    });
    if let Some(u) = usage {
        if let Some(map) = obj.as_object_mut() {
            map.insert("usage".to_string(), u);
        }
    }
    format!("data: {obj}\n\n")
}

fn extract_delta_text(payload: &Value) -> Option<String> {
    payload
        .get("contentBlockDelta")
        .and_then(|c| c.get("delta"))
        .and_then(|d| d.get("text"))
        .and_then(|t| t.as_str())
        .map(str::to_owned)
        .or_else(|| {
            payload
                .get("delta")
                .and_then(|d| d.get("text"))
                .and_then(|t| t.as_str())
                .map(str::to_owned)
        })
}

fn extract_message_stop(payload: &Value) -> Option<&str> {
    payload
        .get("messageStop")
        .and_then(|m| m.get("stopReason"))
        .and_then(Value::as_str)
        .or_else(|| payload.get("stopReason").and_then(Value::as_str))
}

fn extract_metadata_usage(payload: &Value) -> Option<Value> {
    let usage = payload
        .get("metadata")
        .and_then(|m| m.get("usage"))
        .or_else(|| payload.get("usage"))?;
    let num = |k: &str| -> u64 {
        usage
            .get(k)
            .and_then(|v| {
                v.as_u64()
                    .or_else(|| v.as_i64().and_then(|i| u64::try_from(i).ok()))
            })
            .unwrap_or(0)
    };
    let input = num("inputTokens");
    let output = num("outputTokens");
    let total = num("totalTokens").max(input.saturating_add(output));
    Some(json!({
        "prompt_tokens": input,
        "completion_tokens": output,
        "total_tokens": total,
    }))
}

fn extract_assistant_role(payload: &Value) -> bool {
    payload
        .get("messageStart")
        .and_then(|m| m.get("role"))
        .and_then(Value::as_str)
        == Some("assistant")
        || payload.get("role").and_then(Value::as_str) == Some("assistant")
}

fn aws_error_sse(payload: &[u8]) -> String {
    let msg = serde_json::from_slice::<Value>(payload)
        .ok()
        .and_then(|v| {
            v.get("message")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .or_else(|| v.get("__type").and_then(Value::as_str).map(str::to_owned))
        })
        .unwrap_or_else(|| String::from_utf8_lossy(payload).into_owned());
    let err = json!({"message": msg, "type": "bedrock_stream_error"});
    format!("data: {err}\n\n")
}

fn send_sse(tx: &mpsc::UnboundedSender<Result<Bytes, String>>, line: String) {
    let _ = tx.send(Ok(Bytes::from(line)));
}

/// Drain Bedrock `application/vnd.amazon.eventstream` bytes and emit OpenAI-style `data: …` lines.
pub(crate) async fn forward_bedrock_converse_stream_as_sse<S>(
    mut upstream: S,
    model_id: String,
    tx: mpsc::UnboundedSender<Result<Bytes, String>>,
) where
    S: futures_util::Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    let created = now_epoch();
    let id = format!("chatcmpl-bedrock-{created}");
    let mut buf = BytesMut::new();
    let mut sent_role = false;
    let mut pending_finish: Option<&'static str> = None;

    while let Some(chunk) = upstream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.send(Err(format!("Bedrock stream read failed: {e}")));
                return;
            }
        };
        buf.extend_from_slice(&chunk);

        loop {
            match pop_next_event(&mut buf) {
                Ok(ev) => {
                    if let Err(e) = handle_stream_event(
                        &ev,
                        &model_id,
                        &id,
                        created,
                        &mut sent_role,
                        &mut pending_finish,
                        &tx,
                    ) {
                        let _ = tx.send(Err(e));
                        return;
                    }
                }
                Err(DecodeError::Truncated) => break,
                Err(DecodeError::Invalid(reason)) => {
                    let _ = tx.send(Err(format!("Bedrock event stream decode error: {reason}")));
                    return;
                }
            }
        }
    }

    // If the upstream ended while a partial frame remained, surface this as an
    // error rather than emitting `[DONE]`. Silently completing would let clients
    // treat a truncated response as a successful one and corrupt downstream
    // tool-call / conversation state.
    if !buf.is_empty() {
        let _ = tx.send(Err(format!(
            "Bedrock stream ended with {} unread byte(s); upstream terminated mid-frame",
            buf.len()
        )));
        return;
    }

    if let Some(fr) = pending_finish.take() {
        send_sse(
            &tx,
            openai_chunk(&id, created, &model_id, json!({}), json!(fr), None),
        );
    }

    send_sse(&tx, "data: [DONE]\n\n".to_string());
}

fn handle_stream_event(
    ev: &StreamEvent,
    model_id: &str,
    id: &str,
    created: i64,
    sent_role: &mut bool,
    pending_finish: &mut Option<&'static str>,
    tx: &mpsc::UnboundedSender<Result<Bytes, String>>,
) -> Result<(), String> {
    let payload: Value = serde_json::from_slice(&ev.payload)
        .map_err(|e| format!("invalid Bedrock stream JSON: {e}"))?;

    if ev.event_type.ends_with("Exception") || ev.event_type.ends_with("Fault") {
        send_sse(tx, aws_error_sse(&ev.payload));
        return Err("bedrock upstream exception".to_string());
    }

    if payload.get("__type").is_some() && payload.get("message").and_then(|m| m.as_str()).is_some()
    {
        send_sse(tx, aws_error_sse(&ev.payload));
        return Err("bedrock upstream error".to_string());
    }

    if extract_assistant_role(&payload) && !*sent_role {
        *sent_role = true;
        send_sse(
            tx,
            openai_chunk(
                id,
                created,
                model_id,
                json!({"role": "assistant"}),
                Value::Null,
                None,
            ),
        );
    }

    if let Some(text) = extract_delta_text(&payload) {
        if !text.is_empty() {
            send_sse(
                tx,
                openai_chunk(
                    id,
                    created,
                    model_id,
                    json!({"content": text}),
                    Value::Null,
                    None,
                ),
            );
        }
    }

    if let Some(stop) = extract_message_stop(&payload) {
        *pending_finish = Some(map_bedrock_stop_to_openai(stop));
    }

    if let Some(usage) = extract_metadata_usage(&payload) {
        let finish = pending_finish
            .take()
            .map(|s| json!(s))
            .unwrap_or(Value::Null);
        send_sse(
            tx,
            openai_chunk(id, created, model_id, json!({}), finish, Some(usage)),
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_stop_reasons() {
        assert_eq!(map_bedrock_stop_to_openai("end_turn"), "stop");
        assert_eq!(map_bedrock_stop_to_openai("max_tokens"), "length");
        assert_eq!(map_bedrock_stop_to_openai("tool_use"), "tool_calls");
    }

    #[test]
    fn extracts_nested_delta_text() {
        let v: Value = serde_json::from_str(
            r#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"Hello"}}}"#,
        )
        .unwrap();
        assert_eq!(extract_delta_text(&v).as_deref(), Some("Hello"));
    }
}
