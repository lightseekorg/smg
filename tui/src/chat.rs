use futures::StreamExt;
use tokio::sync::mpsc;

use crate::client::SmgClient;

/// A chat message in the conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Which API endpoint to use for chat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatEndpoint {
    #[default]
    Chat,
    Responses,
}

impl ChatEndpoint {
    pub fn label(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Responses => "responses",
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            Self::Chat => Self::Responses,
            Self::Responses => Self::Chat,
        }
    }
}

/// Stream a chat from the SMG gateway, sending tokens via `tx`.
/// Special signals: "\n[DONE]", "\n[ERROR]...", "\n[RESPONSE_ID]..."
pub async fn stream_chat(
    client: &SmgClient,
    model: &str,
    messages: &[serde_json::Value],
    endpoint: ChatEndpoint,
    previous_response_id: Option<String>,
    tx: mpsc::UnboundedSender<String>,
) {
    match endpoint {
        ChatEndpoint::Chat => stream_chat_completions(client, model, messages, tx).await,
        ChatEndpoint::Responses => {
            stream_responses(client, model, messages, previous_response_id, tx).await;
        }
    }
}

async fn stream_chat_completions(
    client: &SmgClient,
    model: &str,
    messages: &[serde_json::Value],
    tx: mpsc::UnboundedSender<String>,
) {
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
    });

    let resp = match client.stream_request("/v1/chat/completions", &body).await {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.send(format!("\n[ERROR]{e}"));
            return;
        }
    };

    process_sse_stream(resp, tx).await;
}

async fn stream_responses(
    client: &SmgClient,
    model: &str,
    messages: &[serde_json::Value],
    previous_response_id: Option<String>,
    tx: mpsc::UnboundedSender<String>,
) {
    // For multi-turn: use previous_response_id + only the latest user message
    let body = if let Some(ref prev_id) = previous_response_id {
        // Only send the latest user message with previous_response_id
        let latest_input = messages
            .last()
            .map(|m| m["content"].as_str().unwrap_or(""))
            .unwrap_or("");
        serde_json::json!({
            "model": model,
            "input": latest_input,
            "previous_response_id": prev_id,
            "stream": true,
        })
    } else {
        // First turn: send all messages as input
        let input: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m["role"],
                    "content": m["content"],
                })
            })
            .collect();
        serde_json::json!({
            "model": model,
            "input": input,
            "stream": true,
        })
    };

    let resp = match client.stream_request("/v1/responses", &body).await {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.send(format!("\n[ERROR]{e}"));
            return;
        }
    };

    // Process responses API streaming format
    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();
    let mut got_deltas = false;

    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.send(format!("\n[ERROR]{e}"));
                return;
            }
        };

        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].to_string();
            buffer.drain(..pos + 1);

            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if line == "data: [DONE]" {
                let _ = tx.send("\n[DONE]".to_string());
                return;
            }

            if let Some(data) = line.strip_prefix("data: ") {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                    let event_type = parsed["type"].as_str().unwrap_or("");
                    match event_type {
                        "response.created" => {
                            // Extract response ID for multi-turn
                            if let Some(id) = parsed["response"]["id"].as_str() {
                                let _ = tx.send(format!("\n[RESPONSE_ID]{id}"));
                            }
                        }
                        "response.output_text.delta" => {
                            if let Some(delta) = parsed["delta"].as_str() {
                                got_deltas = true;
                                let _ = tx.send(delta.to_string());
                            }
                        }
                        "response.completed" | "response.done" => {
                            // Only extract text if no deltas were received (sglang sends full text here)
                            if !got_deltas {
                                if let Some(outputs) = parsed["response"]["output"].as_array() {
                                    for output in outputs {
                                        if let Some(contents) = output["content"].as_array() {
                                            for content in contents {
                                                if let Some(text) = content["text"].as_str() {
                                                    let _ = tx.send(text.to_string());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            let _ = tx.send("\n[DONE]".to_string());
                            return;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let _ = tx.send("\n[DONE]".to_string());
}

async fn process_sse_stream(resp: reqwest::Response, tx: mpsc::UnboundedSender<String>) {
    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.send(format!("\n[ERROR]{e}"));
                return;
            }
        };

        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].to_string();
            buffer.drain(..pos + 1);

            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if line == "data: [DONE]" {
                let _ = tx.send("\n[DONE]".to_string());
                return;
            }

            if let Some(data) = line.strip_prefix("data: ") {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                    // Surface mid-stream errors
                    if let Some(err_msg) = parsed["error"]["message"].as_str() {
                        let _ = tx.send(format!("\n[ERROR]{err_msg}"));
                        return;
                    }
                    if let Some(delta) = parsed["choices"][0]["delta"]["content"].as_str() {
                        let _ = tx.send(delta.to_string());
                    }
                }
            }
        }
    }

    let _ = tx.send("\n[DONE]".to_string());
}
