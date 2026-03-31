//! Shared SSE (Server-Sent Events) codec for encoding and decoding SSE streams.
//!
//! Consolidates duplicate SSE parsing and formatting logic from across the gateway
//! into a single, well-tested module. Two main components:
//!
//! - [`SseEncoder`]: Produces SSE-framed bytes for sending to clients.
//!   Serializes JSON directly into a reusable `Vec<u8>` buffer via
//!   `serde_json::to_writer`, eliminating intermediate `String` allocations.
//!
//! - [`SseDecoder`]: Consumes incoming SSE byte streams from upstream workers.
//!   Uses cursor tracking and borrow-based frame parsing to minimize allocations.

use std::borrow::Cow;

use axum::body::Body;
use bytes::Bytes;
use http::{
    header::{HeaderValue, CONTENT_TYPE},
    StatusCode,
};
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

// ============================================================================
// SseEncoder
// ============================================================================

/// Reusable SSE encoder. The internal `Vec<u8>` grows to high-water mark
/// and retains its capacity across chunks via `.clear()`.
pub struct SseEncoder {
    buf: Vec<u8>,
}

impl SseEncoder {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(4096),
        }
    }

    /// Encode `data: {json}\n\n`.
    ///
    /// Serializes directly into the reusable buffer via `serde_json::to_writer`
    /// (no intermediate `String`), then copies into `Bytes` for the channel.
    pub fn encode_data<T: Serialize>(&mut self, value: &T) -> Result<Bytes, serde_json::Error> {
        self.buf.clear();
        self.buf.extend_from_slice(b"data: ");
        serde_json::to_writer(&mut self.buf, value)?;
        self.buf.extend_from_slice(b"\n\n");
        Ok(Bytes::copy_from_slice(&self.buf))
    }

    /// Encode `event: {type}\ndata: {json}\n\n` (Anthropic/Responses format).
    pub fn encode_event<T: Serialize>(
        &mut self,
        event_type: &str,
        value: &T,
    ) -> Result<Bytes, serde_json::Error> {
        self.buf.clear();
        self.buf.extend_from_slice(b"event: ");
        self.buf.extend_from_slice(event_type.as_bytes());
        self.buf.extend_from_slice(b"\ndata: ");
        serde_json::to_writer(&mut self.buf, value)?;
        self.buf.extend_from_slice(b"\n\n");
        Ok(Bytes::copy_from_slice(&self.buf))
    }

    /// `data: [DONE]\n\n` — static bytes, zero allocation.
    pub fn done() -> Bytes {
        Bytes::from_static(b"data: [DONE]\n\n")
    }
}

impl Default for SseEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SseDecoder
// ============================================================================

/// Maximum buffer size before rejecting (DoS protection).
const DEFAULT_MAX_BUFFER_SIZE: usize = 1024 * 1024; // 1 MB

/// Buffers incoming bytes and yields complete SSE frames.
///
/// - Defers UTF-8 validation to complete frames (handles multi-byte splits)
/// - Uses cursor tracking to avoid per-frame memmove
/// - Returns borrowed `SseFrame` references — zero allocation for single-line data
///
/// # Lifetime constraints
///
/// `SseFrame` borrows from the decoder's internal buffer. You must drop
/// each frame before calling `next_frame()` or `compact()` again. The
/// compiler enforces this statically.
pub struct SseDecoder {
    buf: Vec<u8>,
    consumed: usize,
    max_size: usize,
}

/// A parsed SSE frame. Borrows from the decoder's buffer where possible.
pub struct SseFrame<'a> {
    /// The `event:` field value, if present. Borrowed from buffer.
    pub event_type: Option<&'a str>,
    /// The `data:` field value. Borrowed for single-line (common), owned for multi-line join.
    pub data: Cow<'a, str>,
}

impl SseFrame<'_> {
    /// Deserialize the data as JSON. Called on-demand (lazy deserialization).
    pub fn decode_data<T: serde::de::DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str(&self.data)
    }

    /// Check for `[DONE]` sentinel.
    pub fn is_done(&self) -> bool {
        self.data.as_ref() == "[DONE]"
    }
}

/// Errors that can occur during SSE decoding.
#[derive(Debug)]
pub enum SseDecodeError {
    /// Buffer exceeded the configured maximum size.
    BufferOverflow,
    /// A complete frame contained invalid UTF-8.
    InvalidUtf8(std::str::Utf8Error),
}

impl std::fmt::Display for SseDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SseDecodeError::BufferOverflow => write!(f, "SSE buffer overflow"),
            SseDecodeError::InvalidUtf8(e) => write!(f, "invalid UTF-8 in SSE frame: {e}"),
        }
    }
}

impl std::error::Error for SseDecodeError {}

impl SseDecoder {
    pub fn new() -> Self {
        Self::with_max_size(DEFAULT_MAX_BUFFER_SIZE)
    }

    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            buf: Vec::with_capacity(4096),
            consumed: 0,
            max_size,
        }
    }

    /// Append bytes. Returns error if buffer exceeds max_size.
    /// Check BEFORE extending (DoS protection).
    pub fn push(&mut self, chunk: &[u8]) -> Result<(), SseDecodeError> {
        let unconsumed = self.buf.len() - self.consumed;
        if unconsumed + chunk.len() > self.max_size {
            return Err(SseDecodeError::BufferOverflow);
        }
        self.buf.extend_from_slice(chunk);
        Ok(())
    }

    /// Yield the next complete SSE frame.
    ///
    /// Returns borrowed references into the internal buffer.
    /// Call in a loop until `None`, then call `compact()`.
    pub fn next_frame(&mut self) -> Option<Result<SseFrame<'_>, SseDecodeError>> {
        let remaining = &self.buf[self.consumed..];
        let pos = find_double_newline(remaining)?;
        let frame_bytes = &remaining[..pos];

        let frame_str = match std::str::from_utf8(frame_bytes) {
            Ok(s) => s,
            Err(e) => {
                self.consumed += pos + 2; // skip past \n\n
                return Some(Err(SseDecodeError::InvalidUtf8(e)));
            }
        };

        let result = parse_frame(frame_str);
        self.consumed += pos + 2;
        Some(Ok(result))
    }

    /// Compact: shift unconsumed bytes to front. Call after draining frames.
    /// Single O(n) memmove per batch instead of per-frame.
    pub fn compact(&mut self) {
        if self.consumed > 0 {
            self.buf.drain(..self.consumed);
            self.consumed = 0;
        }
    }

    /// Flush remaining data at end of stream.
    pub fn flush(&mut self) -> Option<Result<SseFrame<'_>, SseDecodeError>> {
        let remaining = &self.buf[self.consumed..];
        if remaining.is_empty() {
            return None;
        }
        let frame_str = match std::str::from_utf8(remaining) {
            Ok(s) => s,
            Err(e) => return Some(Err(SseDecodeError::InvalidUtf8(e))),
        };
        let trimmed = frame_str.trim();
        if trimmed.is_empty() {
            return None;
        }
        self.consumed = self.buf.len();
        Some(Ok(parse_frame(trimmed)))
    }

    /// Unconsumed byte count (for DoS monitoring).
    pub fn buffered_len(&self) -> usize {
        self.buf.len() - self.consumed
    }
}

impl Default for SseDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Standalone block parsing
// ============================================================================

/// Parse a single SSE block (already extracted from the stream) into an [`SseFrame`].
///
/// Use this when you already have a complete SSE block as a string (e.g. from
/// an accumulator or rewrite path) and don't need the full [`SseDecoder`] machinery.
pub fn parse_block(block: &str) -> SseFrame<'_> {
    parse_frame(block.trim())
}

// ============================================================================
// Shared helpers
// ============================================================================

/// Find `\n\n` boundary. Returns byte offset of first `\n` in the pair.
#[inline]
fn find_double_newline(buf: &[u8]) -> Option<usize> {
    // memchr::memmem is ~10x faster than windows(2) for large buffers.
    // memchr is already a direct dependency of model_gateway.
    memchr::memmem::find(buf, b"\n\n")
}

/// Parse a complete SSE frame into event type and data.
///
/// Returns borrowed references for zero-allocation in the common case
/// (single `data:` line, `event:` present or absent).
fn parse_frame(frame: &str) -> SseFrame<'_> {
    let mut event_type: Option<&str> = None;
    let mut first_data: Option<&str> = None;
    let mut extra_data: Option<Vec<&str>> = None;

    for line in frame.lines() {
        // CRLF normalization: strip trailing \r if present (spec requires this)
        let line = line.strip_suffix('\r').unwrap_or(line);

        if line.is_empty() {
            continue;
        }

        // Comments (lines starting with :) — ignore per spec
        if line.starts_with(':') {
            continue;
        }

        if let Some(value) = line.strip_prefix("data:") {
            let value = value.strip_prefix(' ').unwrap_or(value); // spec: strip ONE leading space
            match first_data {
                None => first_data = Some(value),
                Some(_) => {
                    // Multi-line data: rare in practice, but spec-compliant
                    extra_data.get_or_insert_with(Vec::new).push(value);
                }
            }
        } else if let Some(value) = line.strip_prefix("event:") {
            event_type = Some(value.strip_prefix(' ').unwrap_or(value));
        }
        // id:, retry: — silently ignored (not used by LLM APIs)
    }

    let data = match (first_data, extra_data) {
        (None, _) => Cow::Borrowed(""),
        (Some(first), None) => Cow::Borrowed(first),
        (Some(first), Some(rest)) => {
            let mut joined = String::with_capacity(
                first.len() + rest.iter().map(|s| s.len() + 1).sum::<usize>(),
            );
            joined.push_str(first);
            for line in rest {
                joined.push('\n');
                joined.push_str(line);
            }
            Cow::Owned(joined)
        }
    };

    SseFrame { event_type, data }
}

// ============================================================================
// Build SSE Response
// ============================================================================

/// Build an HTTP response with SSE headers and streaming body from an unbounded receiver.
#[expect(
    clippy::expect_used,
    reason = "Response::builder with static headers and valid status code is infallible"
)]
pub fn build_sse_response(
    rx: mpsc::UnboundedReceiver<Result<Bytes, std::io::Error>>,
) -> axum::response::Response {
    let stream = UnboundedReceiverStream::new(rx);
    axum::response::Response::builder()
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- SseEncoder tests ---

    #[test]
    fn test_encode_data_simple() {
        let mut enc = SseEncoder::new();
        let val = serde_json::json!({"type": "ping"});
        let bytes = enc.encode_data(&val).unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.starts_with("data: "));
        assert!(text.ends_with("\n\n"));
        assert!(text.contains("\"type\":\"ping\""));
    }

    #[test]
    fn test_encode_event_with_type() {
        let mut enc = SseEncoder::new();
        let val = serde_json::json!({"index": 0, "delta": "hi"});
        let bytes = enc.encode_event("content_block_delta", &val).unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.starts_with("event: content_block_delta\n"));
        assert!(text.contains("data: "));
        assert!(text.ends_with("\n\n"));
    }

    #[test]
    fn test_encode_done_sentinel() {
        let bytes = SseEncoder::done();
        assert_eq!(&bytes[..], b"data: [DONE]\n\n");
    }

    #[test]
    fn test_encoder_buffer_reuse() {
        let mut enc = SseEncoder::new();
        for i in 0..20 {
            let val = serde_json::json!({"i": i});
            let _ = enc.encode_data(&val).unwrap();
        }
        // Buffer should have grown to hold the largest event and stayed there.
        // Capacity should be >= initial 4096 (never shrinks).
        assert!(enc.buf.capacity() >= 4096);
    }

    #[test]
    fn test_encode_data_matches_format_pattern() {
        // Verify output matches `format!("data: {json}\n\n")` for compatibility.
        let mut enc = SseEncoder::new();
        let val = serde_json::json!({"type":"message_start","message":{"id":"msg_1"}});
        let bytes = enc.encode_data(&val).unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();

        let json_str = serde_json::to_string(&val).unwrap();
        let expected = format!("data: {json_str}\n\n");
        assert_eq!(text, expected);
    }

    #[test]
    fn test_encode_event_matches_format_pattern() {
        // Verify output matches `format!("event: {type}\ndata: {json}\n\n")`.
        let mut enc = SseEncoder::new();
        let val = serde_json::json!({"type":"content_block_start","index":0});
        let bytes = enc.encode_event("content_block_start", &val).unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();

        let json_str = serde_json::to_string(&val).unwrap();
        let expected = format!("event: content_block_start\ndata: {json_str}\n\n");
        assert_eq!(text, expected);
    }

    // --- SseDecoder tests ---

    #[test]
    fn test_decode_single_data_frame() {
        let mut dec = SseDecoder::new();
        dec.push(b"data: {\"type\":\"ping\"}\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.event_type, None);
        assert_eq!(frame.data.as_ref(), "{\"type\":\"ping\"}");
        assert!(matches!(frame.data, Cow::Borrowed(_)));
    }

    #[test]
    fn test_decode_event_and_data() {
        let mut dec = SseDecoder::new();
        dec.push(b"event: message_start\ndata: {\"type\":\"message_start\"}\n\n")
            .unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.event_type, Some("message_start"));
        assert_eq!(frame.data.as_ref(), "{\"type\":\"message_start\"}");
    }

    #[test]
    fn test_decode_multiple_frames() {
        let mut dec = SseDecoder::new();
        dec.push(b"data: first\n\ndata: second\n\n").unwrap();

        let f1 = dec.next_frame().unwrap().unwrap();
        assert_eq!(f1.data.as_ref(), "first");
        drop(f1);

        let f2 = dec.next_frame().unwrap().unwrap();
        assert_eq!(f2.data.as_ref(), "second");
        drop(f2);

        assert!(dec.next_frame().is_none());
    }

    #[test]
    fn test_decode_split_across_chunks() {
        let mut dec = SseDecoder::new();
        dec.push(b"data: hel").unwrap();
        assert!(dec.next_frame().is_none()); // incomplete
        dec.push(b"lo\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.data.as_ref(), "hello");
    }

    #[test]
    fn test_decode_multiline_data() {
        let mut dec = SseDecoder::new();
        dec.push(b"data: line1\ndata: line2\ndata: line3\n\n")
            .unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.data.as_ref(), "line1\nline2\nline3");
        assert!(matches!(frame.data, Cow::Owned(_)));
    }

    #[test]
    fn test_decode_crlf_within_frame() {
        // If the frame *is* delimited by \n\n but lines have trailing \r
        let mut dec = SseDecoder::new();
        dec.push(b"event: ping\r\ndata: {}\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.event_type, Some("ping"));
        assert_eq!(frame.data.as_ref(), "{}");
    }

    #[test]
    fn test_decode_comments_ignored() {
        let mut dec = SseDecoder::new();
        dec.push(b": this is a comment\ndata: hello\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.data.as_ref(), "hello");
        assert_eq!(frame.event_type, None);
    }

    #[test]
    fn test_decode_empty_data() {
        let mut dec = SseDecoder::new();
        dec.push(b"event: ping\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.event_type, Some("ping"));
        assert_eq!(frame.data.as_ref(), "");
    }

    #[test]
    fn test_decode_done_sentinel() {
        let mut dec = SseDecoder::new();
        dec.push(b"data: [DONE]\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert!(frame.is_done());
    }

    #[test]
    fn test_decode_json_deserialization() {
        let mut dec = SseDecoder::new();
        dec.push(b"data: {\"value\":42}\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        let val: serde_json::Value = frame.decode_data().unwrap();
        assert_eq!(val["value"], 42);
    }

    #[test]
    fn test_decode_buffer_overflow() {
        let mut dec = SseDecoder::with_max_size(16);
        let result = dec.push(b"data: this is way too long for the buffer\n\n");
        assert!(matches!(result, Err(SseDecodeError::BufferOverflow)));
    }

    #[test]
    fn test_decode_compact() {
        let mut dec = SseDecoder::new();
        dec.push(b"data: first\n\ndata: second\n\npartial").unwrap();

        let f1 = dec.next_frame().unwrap().unwrap();
        assert_eq!(f1.data.as_ref(), "first");
        drop(f1);

        let f2 = dec.next_frame().unwrap().unwrap();
        assert_eq!(f2.data.as_ref(), "second");
        drop(f2);

        assert!(dec.next_frame().is_none());
        assert_eq!(dec.buffered_len(), 7); // "partial"

        dec.compact();
        assert_eq!(dec.consumed, 0);
        assert_eq!(dec.buf, b"partial");
    }

    #[test]
    fn test_decode_flush() {
        let mut dec = SseDecoder::new();
        dec.push(b"data: final").unwrap();
        assert!(dec.next_frame().is_none());
        let frame = dec.flush().unwrap().unwrap();
        assert_eq!(frame.data.as_ref(), "final");
    }

    #[test]
    fn test_decode_flush_empty() {
        let mut dec = SseDecoder::new();
        assert!(dec.flush().is_none());
    }

    #[test]
    fn test_decode_flush_whitespace_only() {
        let mut dec = SseDecoder::new();
        dec.push(b"  \n  ").unwrap();
        assert!(dec.flush().is_none());
    }

    #[test]
    fn test_decode_unknown_fields_ignored() {
        let mut dec = SseDecoder::new();
        dec.push(b"id: 123\nretry: 5000\ndata: hello\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.data.as_ref(), "hello");
        assert_eq!(frame.event_type, None);
    }

    #[test]
    fn test_decode_leading_space_strip() {
        let mut dec = SseDecoder::new();
        // Per spec: strip exactly one leading space after the colon
        dec.push(b"data:  two spaces\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        // Strip one space, leaving " two spaces" -> " two spaces"
        assert_eq!(frame.data.as_ref(), " two spaces");
    }

    #[test]
    fn test_decode_no_space_after_colon() {
        let mut dec = SseDecoder::new();
        dec.push(b"data:nospace\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.data.as_ref(), "nospace");
    }

    #[test]
    fn test_decode_multi_byte_utf8_split() {
        let mut dec = SseDecoder::new();
        // "data: 日本" in UTF-8
        let full = "data: 日本\n\n";
        let bytes = full.as_bytes();
        // Split in the middle of a multi-byte character
        let mid = 8; // "data: 日" is "data: " (6) + 日 (3 bytes) = 9, split at 8
        dec.push(&bytes[..mid]).unwrap();
        assert!(dec.next_frame().is_none()); // incomplete frame
        dec.push(&bytes[mid..]).unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.data.as_ref(), "日本");
    }

    #[test]
    fn test_decode_empty_lines_between_fields() {
        let mut dec = SseDecoder::new();
        dec.push(b"event: test\n\ndata: value\n\n").unwrap();
        // First frame: just event, no data
        let f1 = dec.next_frame().unwrap().unwrap();
        assert_eq!(f1.event_type, Some("test"));
        assert_eq!(f1.data.as_ref(), "");
        drop(f1);
        // Second frame: just data
        let f2 = dec.next_frame().unwrap().unwrap();
        assert_eq!(f2.data.as_ref(), "value");
    }

    // --- Roundtrip tests ---

    #[test]
    fn test_roundtrip_data_only() {
        let mut enc = SseEncoder::new();
        let original = serde_json::json!({"type":"ping","id":42});
        let bytes = enc.encode_data(&original).unwrap();

        let mut dec = SseDecoder::new();
        dec.push(&bytes).unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        let decoded: serde_json::Value = frame.decode_data().unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_event_with_type() {
        let mut enc = SseEncoder::new();
        let original = serde_json::json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}});
        let bytes = enc.encode_event("content_block_delta", &original).unwrap();

        let mut dec = SseDecoder::new();
        dec.push(&bytes).unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.event_type, Some("content_block_delta"));
        let decoded: serde_json::Value = frame.decode_data().unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_done() {
        let bytes = SseEncoder::done();
        let mut dec = SseDecoder::new();
        dec.push(&bytes).unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert!(frame.is_done());
    }

    #[test]
    fn test_roundtrip_multiple_events() {
        let mut enc = SseEncoder::new();
        let events = vec![
            ("message_start", serde_json::json!({"type":"message_start"})),
            (
                "content_block_start",
                serde_json::json!({"type":"content_block_start","index":0}),
            ),
            (
                "content_block_delta",
                serde_json::json!({"type":"content_block_delta","index":0,"delta":{"text":"hi"}}),
            ),
            (
                "content_block_stop",
                serde_json::json!({"type":"content_block_stop","index":0}),
            ),
            ("message_stop", serde_json::json!({"type":"message_stop"})),
        ];

        let mut all_bytes = Vec::new();
        for (event_type, val) in &events {
            let bytes = enc.encode_event(event_type, val).unwrap();
            all_bytes.extend_from_slice(&bytes);
        }

        let mut dec = SseDecoder::new();
        dec.push(&all_bytes).unwrap();

        for (expected_type, expected_val) in &events {
            let frame = dec.next_frame().unwrap().unwrap();
            assert_eq!(frame.event_type, Some(*expected_type));
            let decoded: serde_json::Value = frame.decode_data().unwrap();
            assert_eq!(&decoded, expected_val);
            drop(frame);
        }

        assert!(dec.next_frame().is_none());
    }

    // --- parse_frame unit tests ---

    #[test]
    fn test_parse_frame_basic() {
        let frame = parse_frame("event: message_start\ndata: {\"type\":\"message_start\"}");
        assert_eq!(frame.event_type, Some("message_start"));
        assert_eq!(frame.data.as_ref(), "{\"type\":\"message_start\"}");
    }

    #[test]
    fn test_parse_frame_data_only() {
        let frame = parse_frame("data: hello");
        assert_eq!(frame.event_type, None);
        assert_eq!(frame.data.as_ref(), "hello");
    }

    #[test]
    fn test_parse_frame_empty() {
        let frame = parse_frame("");
        assert_eq!(frame.event_type, None);
        assert_eq!(frame.data.as_ref(), "");
    }

    #[test]
    fn test_parse_frame_multiline_data() {
        let frame = parse_frame("data: line1\ndata: line2");
        assert_eq!(frame.data.as_ref(), "line1\nline2");
    }

    #[test]
    fn test_parse_frame_comment_only() {
        let frame = parse_frame(": heartbeat");
        assert_eq!(frame.event_type, None);
        assert_eq!(frame.data.as_ref(), "");
    }
}
