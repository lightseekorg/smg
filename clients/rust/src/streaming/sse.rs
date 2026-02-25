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
    Box::pin(futures::stream::unfold(
        (byte_stream, state, String::new()),
        |(mut stream, mut state, mut buffer)| async move {
            loop {
                // Try to parse a complete event from the buffer.
                if let Some(event) = state.try_parse(&mut buffer) {
                    return Some((Ok(event), (stream, state, buffer)));
                }

                // Need more data from the stream.
                match stream.next().await {
                    Some(Ok(chunk)) => {
                        // This is safe because SSE is a text protocol; servers
                        // produce valid UTF-8. Use lossy to be resilient.
                        buffer.push_str(&String::from_utf8_lossy(&chunk));
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(crate::SmgError::Connection(e)),
                            (stream, state, buffer),
                        ));
                    }
                    None => {
                        // Stream ended. Try to flush any remaining event.
                        if let Some(event) = state.flush(&mut buffer) {
                            return Some((Ok(event), (stream, state, buffer)));
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

                    // Skip [DONE] sentinel.
                    if data == "[DONE]" {
                        return None;
                    }

                    return Some(SseEvent {
                        event: event_type,
                        data,
                    });
                }
                // Empty line with no data — skip (keep-alive).
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
    fn flush(&mut self, _buffer: &mut String) -> Option<SseEvent> {
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
