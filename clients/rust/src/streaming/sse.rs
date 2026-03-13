use std::pin::Pin;

use bytes::Bytes;
use futures::stream::Stream;
use tokio_stream::StreamExt;

/// A parsed SSE event.
#[derive(Debug, Clone)]
pub struct SseEvent {
    /// The `event:` field (e.g. "message_start" for Anthropic).
    pub event: Option<String>,
    /// The `data:` field (typically JSON).
    pub data: String,
}

/// Transform a byte stream from reqwest into a stream of [`SseEvent`]s.
///
/// Handles the three SSE protocol variants:
/// - OpenAI: `data: {...}\n\n` terminated by `data: [DONE]`
/// - Anthropic: `event: type\ndata: {...}\n\n`
/// - Responses API: `event: type\ndata: {...}\n\n` + `data: [DONE]`
pub(crate) fn sse_stream(
    byte_stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + Unpin + 'static,
) -> Pin<Box<dyn Stream<Item = Result<SseEvent, crate::SmgError>> + Send>> {
    let state = SseParserState::default();
    // Holds trailing bytes from an incomplete UTF-8 sequence split across chunks.
    let incomplete: Vec<u8> = Vec::new();
    Box::pin(futures::stream::unfold(
        (byte_stream, state, String::new(), incomplete),
        |(mut stream, mut state, mut buffer, mut incomplete)| async move {
            loop {
                // Try to parse a complete event from the buffer.
                if let Some(event) = state.try_parse(&mut buffer) {
                    return Some((Ok(event), (stream, state, buffer, incomplete)));
                }

                // [DONE] was received — terminate the stream.
                if state.done {
                    return None;
                }

                // Need more data from the stream.
                match stream.next().await {
                    Some(Ok(chunk)) => {
                        // Prepend any leftover bytes from a split UTF-8 sequence.
                        let bytes = if incomplete.is_empty() {
                            chunk.to_vec()
                        } else {
                            let mut combined = std::mem::take(&mut incomplete);
                            combined.extend_from_slice(&chunk);
                            combined
                        };
                        match String::from_utf8(bytes) {
                            Ok(s) => buffer.push_str(&s),
                            Err(e) => {
                                let valid_up_to = e.utf8_error().valid_up_to();
                                let raw = e.into_bytes();
                                // SAFETY: valid_up_to is guaranteed valid UTF-8.
                                buffer.push_str(
                                    std::str::from_utf8(&raw[..valid_up_to]).unwrap_or_default(),
                                );
                                // Stash the trailing incomplete sequence for the next chunk.
                                incomplete = raw[valid_up_to..].to_vec();
                            }
                        }
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(crate::SmgError::Connection(e)),
                            (stream, state, buffer, incomplete),
                        ));
                    }
                    None => {
                        // Stream ended. Try to flush any remaining event.
                        if let Some(event) = state.flush(&mut buffer) {
                            return Some((Ok(event), (stream, state, buffer, incomplete)));
                        }
                        return None;
                    }
                }
            }
        },
    ))
}

#[derive(Default)]
struct SseParserState {
    current_event: Option<String>,
    current_data: Vec<String>,
    /// Set when `data: [DONE]` is received, signaling the stream should end.
    done: bool,
}

impl SseParserState {
    /// Try to parse a complete SSE event from the buffer.
    /// Returns `Some(event)` if a complete event was found and consumed.
    fn try_parse(&mut self, buffer: &mut String) -> Option<SseEvent> {
        loop {
            // Find the next line boundary.
            let newline_pos = buffer.find('\n')?;
            let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
            // Consume the line + newline from buffer.
            buffer.drain(..=newline_pos);

            if line.is_empty() {
                // Empty line = event boundary.
                if !self.current_data.is_empty() {
                    let data = self.current_data.join("\n");
                    let event_type = self.current_event.take();
                    self.current_data.clear();

                    // `[DONE]` signals the end of the stream (OpenAI/Responses API).
                    if data == "[DONE]" {
                        self.done = true;
                        return None;
                    }

                    return Some(SseEvent {
                        event: event_type,
                        data,
                    });
                }
                // Empty line with no data — discard any stale event type
                // so it doesn't leak into the next event.
                self.current_event = None;
                continue;
            }

            if let Some(value) = line
                .strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"))
            {
                self.current_data.push(value.to_string());
            } else if let Some(value) = line
                .strip_prefix("event: ")
                .or_else(|| line.strip_prefix("event:"))
            {
                self.current_event = Some(value.to_string());
            }
            // Ignore other fields (id:, retry:, comments starting with :).
        }
    }

    /// Flush any remaining event data when the stream ends.
    /// Processes any partial line left in the buffer before assembling the event.
    fn flush(&mut self, buffer: &mut String) -> Option<SseEvent> {
        // Process any remaining partial line in the buffer.
        if !buffer.is_empty() {
            let line = buffer.trim_end_matches(['\r', '\n']);
            if let Some(value) = line
                .strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"))
            {
                self.current_data.push(value.to_string());
            } else if let Some(value) = line
                .strip_prefix("event: ")
                .or_else(|| line.strip_prefix("event:"))
            {
                self.current_event = Some(value.to_string());
            }
            buffer.clear();
        }

        if !self.current_data.is_empty() {
            let data = self.current_data.join("\n");
            let event_type = self.current_event.take();
            self.current_data.clear();

            if data == "[DONE]" {
                return None;
            }

            return Some(SseEvent {
                event: event_type,
                data,
            });
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;

    /// Helper to create a byte stream from string chunks (simulating an HTTP response body).
    fn bytes_stream(
        chunks: Vec<&str>,
    ) -> impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + Unpin + 'static {
        let owned: Vec<Result<Bytes, reqwest::Error>> = chunks
            .into_iter()
            .map(|s| Ok(Bytes::from(s.to_string())))
            .collect();
        futures::stream::iter(owned)
    }

    #[tokio::test]
    async fn test_sse_parses_openai_format() {
        let raw = "data: {\"id\":\"1\",\"choices\":[]}\n\ndata: {\"id\":\"2\",\"choices\":[]}\n\ndata: [DONE]\n\n";
        let stream = sse_stream(bytes_stream(vec![raw]));
        let results: Vec<_> = stream.collect().await;
        let events: Vec<SseEvent> = results
            .into_iter()
            .map(|r| r.expect("unexpected SSE parse error"))
            .collect();
        assert_eq!(events.len(), 2);
        assert!(events[0].data.contains("\"id\":\"1\""));
        assert!(events[1].data.contains("\"id\":\"2\""));
        assert!(events[0].event.is_none());
    }

    #[tokio::test]
    async fn test_sse_parses_anthropic_format() {
        let raw = "event: message_start\ndata: {\"message\":{\"id\":\"msg_1\"}}\n\nevent: content_block_delta\ndata: {\"delta\":{\"text\":\"hi\"}}\n\n";
        let stream = sse_stream(bytes_stream(vec![raw]));
        let results: Vec<_> = stream.collect().await;
        let events: Vec<SseEvent> = results
            .into_iter()
            .map(|r| r.expect("unexpected SSE parse error"))
            .collect();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event.as_deref(), Some("message_start"));
        assert_eq!(events[1].event.as_deref(), Some("content_block_delta"));
    }

    #[tokio::test]
    async fn test_sse_multi_chunk_split() {
        // Split the payload mid-line to exercise cross-chunk buffering.
        let stream = sse_stream(bytes_stream(vec![
            "data: {\"id\":",
            "\"1\",\"choices\":[]}\n\ndata: [DONE]\n\n",
        ]));
        let results: Vec<_> = stream.collect().await;
        let events: Vec<SseEvent> = results
            .into_iter()
            .map(|r| r.expect("unexpected SSE parse error"))
            .collect();
        assert_eq!(events.len(), 1);
        assert!(events[0].data.contains("\"id\":\"1\""));
    }

    #[tokio::test]
    async fn test_sse_done_terminates_stream() {
        // After [DONE], the stream should end even if the transport hasn't closed.
        let raw = "data: {\"id\":\"1\"}\n\ndata: [DONE]\n\ndata: {\"id\":\"2\"}\n\n";
        let stream = sse_stream(bytes_stream(vec![raw]));
        let results: Vec<_> = stream.collect().await;
        let events: Vec<SseEvent> = results
            .into_iter()
            .map(|r| r.expect("unexpected SSE parse error"))
            .collect();
        // Only the first event should be yielded; [DONE] terminates the stream.
        assert_eq!(events.len(), 1);
        assert!(events[0].data.contains("\"id\":\"1\""));
    }

    #[tokio::test]
    async fn test_sse_stale_event_type_not_leaked() {
        // An event: line followed by an empty boundary (no data) should not
        // leak the event type into the next real event.
        let raw = "event: stale\n\ndata: {\"id\":\"1\"}\n\n";
        let stream = sse_stream(bytes_stream(vec![raw]));
        let results: Vec<_> = stream.collect().await;
        let events: Vec<SseEvent> = results
            .into_iter()
            .map(|r| r.expect("unexpected SSE parse error"))
            .collect();
        assert_eq!(events.len(), 1);
        assert!(events[0].data.contains("\"id\":\"1\""));
        // The stale "stale" event type must NOT be attached to this event.
        assert!(events[0].event.is_none());
    }

    #[tokio::test]
    async fn test_sse_multibyte_utf8_split_across_chunks() {
        // Split a multibyte UTF-8 character (é = 0xC3 0xA9) across two chunks
        // to verify the incremental decoder stitches it back together.
        let prefix = b"data: {\"text\":\"caf";
        let e_acute_byte1: &[u8] = &[0xC3]; // first byte of 'é'
        let e_acute_byte2: &[u8] = &[0xA9]; // second byte of 'é'
        let suffix = b"\"}\n\ndata: [DONE]\n\n";

        let chunk1: Vec<u8> = [prefix.as_slice(), e_acute_byte1].concat();
        let chunk2: Vec<u8> = [e_acute_byte2, suffix.as_slice()].concat();

        let owned: Vec<Result<Bytes, reqwest::Error>> =
            vec![Ok(Bytes::from(chunk1)), Ok(Bytes::from(chunk2))];
        let byte_stream = futures::stream::iter(owned);
        let stream = sse_stream(byte_stream);
        let results: Vec<_> = stream.collect().await;
        let events: Vec<SseEvent> = results
            .into_iter()
            .map(|r| r.expect("unexpected SSE parse error"))
            .collect();
        assert_eq!(events.len(), 1);
        assert!(events[0].data.contains("café"));
    }
}
