//! SSE infrastructure for Anthropic streaming responses
//!
//! Provides SSE frame parsing, event formatting, stream wrappers,
//! and the core stream consumption logic used by the streaming processor.

use std::{
    io,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use axum::{
    body::Body,
    http::{header, HeaderMap, StatusCode},
    response::Response,
};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use openai_protocol::messages::{ContentBlock, MessageDeltaUsage, StopReason, ToolUseBlock};
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{debug, error, warn};

use super::mcp::{IterationResult, McpToolCall};
use crate::core::Worker;

// ============================================================================
// Constants
// ============================================================================

/// Maximum SSE buffer size (1 MB) to prevent DoS from upstream workers
/// that send data without frame delimiters.
const MAX_SSE_BUFFER_SIZE: usize = 1024 * 1024;

/// Maximum content block index accepted from an upstream worker.
/// Prevents OOM from a malicious worker sending an extremely large index.
const MAX_UPSTREAM_BLOCK_INDEX: u32 = 1024;

/// Maximum accumulated size for a single content block's text/JSON (10 MB).
/// Prevents unbounded memory growth from a stream of deltas.
const MAX_BLOCK_ACCUMULATION_SIZE: usize = 10 * 1024 * 1024;

// ============================================================================
// Public types
// ============================================================================

/// Result from consuming an upstream SSE stream.
pub(crate) struct StreamConsumeResult {
    pub iteration: IterationResult,
    pub usage: Option<MessageDeltaUsage>,
}

// ============================================================================
// SSE Response Builder
// ============================================================================

/// Build an SSE response from a status, upstream headers, and body stream.
pub(crate) fn build_sse_response(
    status: StatusCode,
    upstream_headers: HeaderMap,
    body: Body,
) -> Response {
    let mut builder = Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive");

    for (key, value) in upstream_headers.iter() {
        let key_str = key.as_str();
        if !matches!(
            key_str,
            "content-type"
                | "cache-control"
                | "connection"
                | "transfer-encoding"
                | "content-length"
        ) {
            builder = builder.header(key, value);
        }
    }

    builder.body(body).unwrap_or_else(|e| {
        error!("Failed to build streaming response: {}", e);
        crate::routers::error::internal_error("response_build_failed", "Failed to build response")
    })
}

// ============================================================================
// Load Tracking Stream Wrapper
// ============================================================================

/// Stream wrapper that tracks worker load and circuit breaker outcome.
///
/// Decrements worker load when the stream completes or is dropped, and records
/// circuit breaker outcome based on whether the stream completed successfully.
pub(crate) struct LoadTrackingStream<S> {
    inner: Pin<Box<S>>,
    /// Worker is wrapped in `Option` so `Drop` can `.take()` it exactly once.
    /// It is always `Some` during the stream's lifetime.
    worker: Option<Arc<dyn Worker>>,
    completed_successfully: bool,
    encountered_error: bool,
    /// If true, always record failure regardless of stream completion
    force_failure: bool,
}

impl<S> LoadTrackingStream<S> {
    pub(crate) fn new(inner: S, worker: Arc<dyn Worker>) -> Self {
        Self {
            inner: Box::pin(inner),
            worker: Some(worker),
            completed_successfully: false,
            encountered_error: false,
            force_failure: false,
        }
    }

    pub(crate) fn new_force_failure(inner: S, worker: Arc<dyn Worker>) -> Self {
        Self {
            inner: Box::pin(inner),
            worker: Some(worker),
            completed_successfully: false,
            encountered_error: false,
            force_failure: true,
        }
    }
}

impl<S> Stream for LoadTrackingStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>>,
{
    type Item = Result<Bytes, io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => Poll::Ready(Some(Ok(bytes))),
            Poll::Ready(Some(Err(e))) => {
                self.encountered_error = true;
                Poll::Ready(Some(Err(io::Error::other(e.to_string()))))
            }
            Poll::Ready(None) => {
                self.completed_successfully = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<S> Drop for LoadTrackingStream<S> {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.decrement_load();

            if self.force_failure {
                worker.record_outcome(false);
                debug!(
                    completed = %self.completed_successfully,
                    "LoadTrackingStream (force_failure) completed, recorded failure"
                );
            } else if self.completed_successfully && !self.encountered_error {
                worker.record_outcome(true);
                debug!("LoadTrackingStream completed successfully, recorded success");
            } else {
                worker.record_outcome(false);
                debug!(
                    completed = %self.completed_successfully,
                    error = %self.encountered_error,
                    "LoadTrackingStream interrupted or errored, recorded failure"
                );
            }
        }
    }
}

// ============================================================================
// SSE event formatting and sending
// ============================================================================

/// Format and send an SSE event through the channel.
///
/// Returns `true` if the send succeeded, `false` if the receiver was dropped.
pub(crate) async fn send_event(
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
    event_type: &str,
    data: &Value,
) -> bool {
    let bytes = format_sse_event(event_type, data);
    tx.send(Ok(bytes)).await.is_ok()
}

/// Format a `MessageStreamEvent` as SSE bytes: `event: <type>\ndata: <json>\n\n`
fn format_sse_event(event_type: &str, data: &Value) -> Bytes {
    let json = serde_json::to_string(data).unwrap_or_else(|_| "{}".to_string());
    Bytes::from(format!("event: {}\ndata: {}\n\n", event_type, json))
}

/// Send an SSE error event.
pub(crate) async fn send_error(tx: &mpsc::Sender<Result<Bytes, io::Error>>, message: &str) -> bool {
    let data = serde_json::json!({
        "type": "error",
        "error": {
            "type": "api_error",
            "message": message
        }
    });
    send_event(tx, "error", &data).await
}

/// Emit `content_block_start` + `content_block_stop` events for an
/// `mcp_tool_result` block.
pub(crate) async fn emit_mcp_tool_result(
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
    call: &McpToolCall,
    global_index: &mut u32,
) -> bool {
    let index = *global_index;

    // content_block_start with mcp_tool_result
    let block_start = serde_json::json!({
        "type": "content_block_start",
        "index": index,
        "content_block": {
            "type": "mcp_tool_result",
            "tool_use_id": call.mcp_id,
            "is_error": call.is_error,
            "content": [{
                "type": "text",
                "text": call.result_content
            }]
        }
    });

    if !send_event(tx, "content_block_start", &block_start).await {
        return false;
    }

    // content_block_stop
    let block_stop = serde_json::json!({
        "type": "content_block_stop",
        "index": index
    });

    if !send_event(tx, "content_block_stop", &block_stop).await {
        return false;
    }

    *global_index += 1;
    true
}

/// Emit the final `message_delta` and `message_stop` events.
pub(crate) async fn emit_final(
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
    stop_reason: Option<&StopReason>,
    total_input_tokens: u32,
    total_output_tokens: u32,
) {
    let stop_reason_val = stop_reason
        .map(|r| serde_json::to_value(r).unwrap_or(Value::Null))
        .unwrap_or(Value::Null);

    let message_delta = serde_json::json!({
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason_val,
            "stop_sequence": null
        },
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens
        }
    });

    if !send_event(tx, "message_delta", &message_delta).await {
        debug!("Failed to send final message_delta — channel closed");
    }

    let message_stop = serde_json::json!({
        "type": "message_stop"
    });
    if !send_event(tx, "message_stop", &message_stop).await {
        debug!("Failed to send message_stop — channel closed");
    }
}

// ============================================================================
// Stream consumption
// ============================================================================

/// Consume an upstream SSE byte stream, parsing events and forwarding them
/// (with transformations) to the client.
///
/// The `resolve_server_name` closure maps a tool name to its MCP server label,
/// decoupling this function from `McpToolSession`.
pub(crate) async fn consume_and_forward<F>(
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
    response: reqwest::Response,
    global_index: &mut u32,
    is_first_iteration: bool,
    resolve_server_name: F,
) -> Result<StreamConsumeResult, String>
where
    F: Fn(&str) -> String,
{
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut processor =
        EventProcessor::new(tx, global_index, is_first_iteration, resolve_server_name);

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream read error: {}", e))?;
        let text = String::from_utf8_lossy(&chunk);
        buffer.push_str(&text);

        // Guard against unbounded buffer growth (DoS protection)
        if buffer.len() > MAX_SSE_BUFFER_SIZE {
            return Err(format!(
                "SSE buffer exceeded maximum size ({} bytes) — possible malformed upstream stream",
                MAX_SSE_BUFFER_SIZE
            ));
        }

        // Process complete SSE frames (delimited by double newline)
        while let Some(frame_end) = buffer.find("\n\n") {
            let frame: String = buffer.drain(..frame_end + 2).collect();
            let frame = &frame[..frame.len() - 2]; // strip trailing \n\n

            if let Some((event_type, data)) = parse_sse_frame(frame) {
                processor.process(&event_type, &data).await?;
            }
        }
    }

    // Process any remaining data in buffer
    if !buffer.trim().is_empty() {
        if let Some((event_type, data)) = parse_sse_frame(&buffer) {
            processor.process(&event_type, &data).await?;
        }
    }

    Ok(processor.into_result())
}

// ============================================================================
// Internal: Block accumulator
// ============================================================================

/// Accumulator for a content block being streamed.
enum BlockAccumulator {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input_json: String,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
}

impl BlockAccumulator {
    /// Create a new accumulator for the given content block type.
    fn for_type(block_type: &str) -> Self {
        match block_type {
            "thinking" => Self::Thinking {
                thinking: String::new(),
                signature: String::new(),
            },
            _ => Self::Text {
                text: String::new(),
            },
        }
    }

    /// Accumulate a streaming delta into this block.
    fn accumulate_delta(&mut self, delta: &Value) {
        let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match self {
            Self::Text { text } if delta_type == "text_delta" => {
                if let Some(t) = delta.get("text").and_then(|v| v.as_str()) {
                    if text.len() + t.len() <= MAX_BLOCK_ACCUMULATION_SIZE {
                        text.push_str(t);
                    }
                }
            }
            Self::ToolUse { input_json, .. } if delta_type == "input_json_delta" => {
                if let Some(json) = delta.get("partial_json").and_then(|v| v.as_str()) {
                    if input_json.len() + json.len() <= MAX_BLOCK_ACCUMULATION_SIZE {
                        input_json.push_str(json);
                    }
                }
            }
            Self::Thinking {
                thinking,
                signature,
            } => {
                if delta_type == "thinking_delta" {
                    if let Some(t) = delta.get("thinking").and_then(|v| v.as_str()) {
                        if thinking.len() + t.len() <= MAX_BLOCK_ACCUMULATION_SIZE {
                            thinking.push_str(t);
                        }
                    }
                } else if delta_type == "signature_delta" {
                    if let Some(s) = delta.get("signature").and_then(|v| v.as_str()) {
                        if signature.len() + s.len() <= MAX_BLOCK_ACCUMULATION_SIZE {
                            signature.push_str(s);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Finalize this accumulator into a `ContentBlock` and optional `ToolUseBlock`.
    fn finalize(&self) -> (ContentBlock, Option<ToolUseBlock>) {
        match self {
            Self::Text { text } => (
                ContentBlock::Text {
                    text: text.clone(),
                    citations: None,
                },
                None,
            ),
            Self::ToolUse {
                id,
                name,
                input_json,
            } => {
                let input: Value = serde_json::from_str(input_json).unwrap_or_else(|e| {
                    warn!(
                        error = %e,
                        json = %input_json,
                        "Failed to parse tool input JSON, using empty object"
                    );
                    Value::Object(serde_json::Map::new())
                });
                let tool_use = ToolUseBlock {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    cache_control: None,
                };
                (
                    ContentBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input,
                    },
                    Some(tool_use),
                )
            }
            Self::Thinking {
                thinking,
                signature,
            } => (
                ContentBlock::Thinking {
                    thinking: thinking.clone(),
                    signature: signature.clone(),
                },
                None,
            ),
        }
    }
}

// ============================================================================
// Internal: SSE event processor
// ============================================================================

/// Processes SSE events from the upstream worker, transforming and forwarding
/// them to the client with index remapping and tool_use → mcp_tool_use conversion.
///
/// Generic over `F` to decouple from `McpToolSession` — the closure resolves
/// tool names to MCP server labels.
struct EventProcessor<'a, F> {
    tx: &'a mpsc::Sender<Result<Bytes, io::Error>>,
    global_index: &'a mut u32,
    index_base: u32,
    is_first_iteration: bool,
    resolve_server_name: F,
    result: IterationResult,
    usage: Option<MessageDeltaUsage>,
    upstream_blocks: Vec<BlockAccumulator>,
}

impl<'a, F> EventProcessor<'a, F>
where
    F: Fn(&str) -> String,
{
    fn new(
        tx: &'a mpsc::Sender<Result<Bytes, io::Error>>,
        global_index: &'a mut u32,
        is_first_iteration: bool,
        resolve_server_name: F,
    ) -> Self {
        let index_base = *global_index;
        Self {
            tx,
            global_index,
            index_base,
            is_first_iteration,
            resolve_server_name,
            result: IterationResult {
                content_blocks: Vec::new(),
                tool_use_blocks: Vec::new(),
                stop_reason: None,
            },
            usage: None,
            upstream_blocks: Vec::new(),
        }
    }

    /// Consume the accumulated result.
    fn into_result(self) -> StreamConsumeResult {
        StreamConsumeResult {
            iteration: self.result,
            usage: self.usage,
        }
    }

    /// Process a single SSE event from the upstream worker.
    async fn process(&mut self, event_type: &str, data: &str) -> Result<(), String> {
        let mut parsed: Value =
            serde_json::from_str(data).map_err(|e| format!("Failed to parse SSE data: {}", e))?;

        match event_type {
            "message_start" => {
                if self.is_first_iteration {
                    send_event(self.tx, "message_start", &parsed).await;
                }
            }
            "content_block_start" => self.handle_block_start(&mut parsed).await?,
            "content_block_delta" => self.handle_block_delta(&mut parsed).await,
            "content_block_stop" => self.handle_block_stop(&parsed).await,
            "message_delta" => self.handle_message_delta(&parsed),
            "message_stop" => { /* Don't forward — we emit our own at the end */ }
            "ping" => {
                send_event(self.tx, "ping", &serde_json::json!({"type": "ping"})).await;
            }
            "error" => {
                send_event(self.tx, "error", &parsed).await;
            }
            _ => {
                debug!(event_type = %event_type, "Forwarding unknown SSE event type");
                send_event(self.tx, event_type, &parsed).await;
            }
        }

        Ok(())
    }

    /// Handle a `content_block_start` event: transform tool_use → mcp_tool_use,
    /// remap index, and initialize the block accumulator.
    async fn handle_block_start(&mut self, parsed: &mut Value) -> Result<(), String> {
        let upstream_index = parsed.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

        if upstream_index > MAX_UPSTREAM_BLOCK_INDEX {
            return Err(format!(
                "Upstream content block index {} exceeds maximum ({})",
                upstream_index, MAX_UPSTREAM_BLOCK_INDEX
            ));
        }

        let block_type = parsed
            .get("content_block")
            .and_then(|cb| cb.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let client_index = self.index_base + upstream_index;

        while self.upstream_blocks.len() <= upstream_index as usize {
            self.upstream_blocks.push(BlockAccumulator::Text {
                text: String::new(),
            });
        }

        if block_type == "tool_use" {
            let content_block = parsed.get("content_block").cloned().unwrap_or(Value::Null);
            self.emit_mcp_tool_use_start(&content_block, upstream_index, client_index)
                .await;
        } else {
            // Initialize accumulator before mutating parsed (block_type borrows parsed)
            self.upstream_blocks[upstream_index as usize] = BlockAccumulator::for_type(block_type);
            parsed["index"] = Value::from(client_index);
            send_event(self.tx, "content_block_start", parsed).await;
        }

        Ok(())
    }

    /// Transform an upstream `tool_use` block into `mcp_tool_use` and emit it.
    async fn emit_mcp_tool_use_start(
        &mut self,
        content_block: &Value,
        upstream_index: u32,
        client_index: u32,
    ) {
        let id = content_block
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let name = content_block
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let mcp_id = format!("mcptoolu_{}", id.trim_start_matches("toolu_"));
        let server_name = (self.resolve_server_name)(&name);

        let event = serde_json::json!({
            "type": "content_block_start",
            "index": client_index,
            "content_block": {
                "type": "mcp_tool_use",
                "id": mcp_id,
                "name": name,
                "server_name": server_name,
                "input": {}
            }
        });
        send_event(self.tx, "content_block_start", &event).await;

        self.upstream_blocks[upstream_index as usize] = BlockAccumulator::ToolUse {
            id,
            name,
            input_json: String::new(),
        };
    }

    /// Handle a `content_block_delta` event: accumulate content and forward
    /// with remapped index.
    async fn handle_block_delta(&mut self, parsed: &mut Value) {
        let upstream_index = parsed.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let client_index = self.index_base + upstream_index;

        // Accumulate before mutating (we need to read delta first)
        if let Some(delta) = parsed.get("delta") {
            if let Some(block) = self.upstream_blocks.get_mut(upstream_index as usize) {
                block.accumulate_delta(delta);
            }
        }

        parsed["index"] = Value::from(client_index);
        send_event(self.tx, "content_block_delta", parsed).await;
    }

    /// Handle a `content_block_stop` event: finalize the accumulated block
    /// and update the global index.
    async fn handle_block_stop(&mut self, parsed: &Value) {
        let upstream_index = parsed.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let client_index = self.index_base + upstream_index;

        let event = serde_json::json!({
            "type": "content_block_stop",
            "index": client_index
        });
        send_event(self.tx, "content_block_stop", &event).await;

        if let Some(block) = self.upstream_blocks.get(upstream_index as usize) {
            let (content_block, tool_use) = block.finalize();
            self.result.content_blocks.push(content_block);
            if let Some(tool_use) = tool_use {
                self.result.tool_use_blocks.push(tool_use);
            }
        }

        *self.global_index = (*self.global_index).max(client_index + 1);
    }

    /// Handle a `message_delta` event: capture stop_reason and usage
    /// (not forwarded — we emit our own combined delta at the end).
    fn handle_message_delta(&mut self, parsed: &Value) {
        if let Some(delta) = parsed.get("delta") {
            if let Some(stop_str) = delta.get("stop_reason").and_then(|v| v.as_str()) {
                self.result.stop_reason =
                    serde_json::from_value(Value::String(stop_str.to_string())).ok();
            }
        }
        if let Some(usage) = parsed.get("usage") {
            self.usage = serde_json::from_value(usage.clone()).ok();
        }
    }
}

// ============================================================================
// SSE frame parsing
// ============================================================================

/// Parse a raw SSE frame into `(event_type, data)`.
///
/// SSE frames look like:
/// ```text
/// event: content_block_start
/// data: {"type":"content_block_start",...}
/// ```
fn parse_sse_frame(frame: &str) -> Option<(String, String)> {
    let mut event_type = String::new();
    let mut data_lines = Vec::new();

    for line in frame.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(value) = line.strip_prefix("event:") {
            event_type = value.trim().to_string();
        } else if let Some(value) = line.strip_prefix("data:") {
            data_lines.push(value.trim().to_string());
        }
    }

    if data_lines.is_empty() {
        return None;
    }

    let data = data_lines.join("\n");

    // If no event type specified, try to infer from data
    if event_type.is_empty() {
        if let Ok(parsed) = serde_json::from_str::<Value>(&data) {
            if let Some(t) = parsed.get("type").and_then(|v| v.as_str()) {
                event_type = t.to_string();
            }
        }
    }

    if event_type.is_empty() {
        return None;
    }

    Some((event_type, data))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sse_frame_basic() {
        let frame = "event: message_start\ndata: {\"type\":\"message_start\"}";
        let (event_type, data) = parse_sse_frame(frame).unwrap();
        assert_eq!(event_type, "message_start");
        assert_eq!(data, "{\"type\":\"message_start\"}");
    }

    #[test]
    fn test_parse_sse_frame_content_block() {
        let frame = "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}";
        let (event_type, data) = parse_sse_frame(frame).unwrap();
        assert_eq!(event_type, "content_block_start");
        let parsed: Value = serde_json::from_str(&data).unwrap();
        assert_eq!(parsed["index"], 0);
    }

    #[test]
    fn test_parse_sse_frame_no_event_type_infers() {
        let frame = "data: {\"type\":\"ping\"}";
        let (event_type, _data) = parse_sse_frame(frame).unwrap();
        assert_eq!(event_type, "ping");
    }

    #[test]
    fn test_parse_sse_frame_empty() {
        assert!(parse_sse_frame("").is_none());
        assert!(parse_sse_frame("event: foo").is_none());
    }

    #[test]
    fn test_format_sse_event() {
        let data = serde_json::json!({"type": "ping"});
        let bytes = format_sse_event("ping", &data);
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.starts_with("event: ping\n"));
        assert!(text.contains("data: "));
        assert!(text.ends_with("\n\n"));
    }

    #[test]
    fn test_parse_sse_frame_with_extra_whitespace() {
        let frame = "  event: content_block_delta  \n  data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}  ";
        let (event_type, data) = parse_sse_frame(frame).unwrap();
        assert_eq!(event_type, "content_block_delta");
        let parsed: Value = serde_json::from_str(&data).unwrap();
        assert_eq!(parsed["index"], 1);
    }
}
