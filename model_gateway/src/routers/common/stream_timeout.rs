//! Helpers for enforcing a wall-clock cap on long-lived streaming responses.

use std::{fmt, future::Future, time::Duration};

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use tokio::time::{self, Instant};

pub const MAX_STREAM_ERROR_BODY_SIZE: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct StreamDeadline {
    total_deadline: Instant,
    total_timeout: Duration,
    idle_timeout: Duration,
}

impl StreamDeadline {
    pub fn new(total_timeout: Duration, idle_timeout: Duration) -> Self {
        let now = Instant::now();
        Self {
            total_deadline: now.checked_add(total_timeout).unwrap_or(now),
            total_timeout,
            idle_timeout,
        }
    }

    pub fn is_total_elapsed(&self) -> bool {
        Instant::now() >= self.total_deadline
    }

    pub fn message(&self, timeout: StreamTimeoutKind) -> String {
        match timeout {
            StreamTimeoutKind::Total => format!(
                "Streaming request exceeded configured total timeout of {} seconds",
                self.total_timeout.as_secs()
            ),
            StreamTimeoutKind::Idle => format!(
                "Streaming request exceeded configured idle timeout of {} seconds",
                self.idle_timeout.as_secs()
            ),
        }
    }

    pub fn sse_error_event(&self, timeout: StreamTimeoutKind) -> Bytes {
        let payload = serde_json::json!({
            "error": {
                "type": "timeout",
                "code": "streaming_timeout",
                "message": self.message(timeout),
            }
        });
        Bytes::from(format!("event: error\ndata: {payload}\n\n"))
    }

    pub async fn until_total<F>(&self, future: F) -> Result<F::Output, StreamTimeoutKind>
    where
        F: Future,
    {
        time::timeout_at(self.total_deadline, future)
            .await
            .map_err(|_| StreamTimeoutKind::Total)
    }

    pub async fn until_activity<F>(&self, future: F) -> Result<F::Output, StreamTimeoutKind>
    where
        F: Future,
    {
        let (deadline, timeout) = self.activity_deadline()?;
        time::timeout_at(deadline, future)
            .await
            .map_err(|_| timeout)
    }

    pub async fn next<S>(&self, stream: &mut S) -> Result<Option<S::Item>, StreamTimeoutKind>
    where
        S: Stream + Unpin,
    {
        let (next_deadline, timeout) = self.activity_deadline()?;
        time::timeout_at(next_deadline, stream.next())
            .await
            .map_err(|_| timeout)
    }

    fn activity_deadline(&self) -> Result<(Instant, StreamTimeoutKind), StreamTimeoutKind> {
        let now = Instant::now();
        if now >= self.total_deadline {
            return Err(StreamTimeoutKind::Total);
        }

        let idle_deadline = now.checked_add(self.idle_timeout).unwrap_or(now);
        Ok(if idle_deadline < self.total_deadline {
            (idle_deadline, StreamTimeoutKind::Idle)
        } else {
            (self.total_deadline, StreamTimeoutKind::Total)
        })
    }

    pub async fn read_text_limited<S, E>(
        &self,
        stream: &mut S,
        max_size: usize,
    ) -> Result<String, StreamBodyReadError>
    where
        S: Stream<Item = Result<Bytes, E>> + Unpin,
        E: fmt::Display,
    {
        let mut buf = Vec::new();
        let mut total_size = 0;

        loop {
            let chunk = match self.next(stream).await {
                Ok(Some(chunk)) => chunk,
                Ok(None) => break,
                Err(timeout) => return Err(StreamBodyReadError::Timeout(timeout)),
            };

            let chunk = chunk.map_err(|err| StreamBodyReadError::Read(err.to_string()))?;
            total_size += chunk.len();
            if total_size > max_size {
                return Err(StreamBodyReadError::TooLarge { max_size });
            }
            buf.extend_from_slice(&chunk);
        }

        String::from_utf8(buf).map_err(|err| StreamBodyReadError::InvalidUtf8(err.to_string()))
    }

    pub fn body_read_error_message(&self, error: &StreamBodyReadError) -> String {
        match error {
            StreamBodyReadError::Timeout(timeout) => self.message(*timeout),
            StreamBodyReadError::TooLarge { max_size } => {
                format!("Streaming error body exceeded configured limit of {max_size} bytes")
            }
            StreamBodyReadError::Read(error) => {
                format!("Failed to read upstream error body: {error}")
            }
            StreamBodyReadError::InvalidUtf8(error) => {
                format!("Invalid UTF-8 in upstream error body: {error}")
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum StreamTimeoutKind {
    Total,
    Idle,
}

#[derive(Debug)]
pub enum StreamBodyReadError {
    Timeout(StreamTimeoutKind),
    TooLarge { max_size: usize },
    Read(String),
    InvalidUtf8(String),
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures_util::stream;

    use super::*;

    #[tokio::test]
    async fn next_reports_total_timeout() {
        let deadline = StreamDeadline::new(Duration::from_millis(10), Duration::from_secs(1));
        let mut stream = stream::pending::<()>();

        let result = deadline.next(&mut stream).await;

        assert!(matches!(result, Err(StreamTimeoutKind::Total)));
    }

    #[tokio::test]
    async fn next_reports_idle_timeout() {
        let deadline = StreamDeadline::new(Duration::from_secs(1), Duration::from_millis(10));
        let mut stream = stream::pending::<()>();

        let result = deadline.next(&mut stream).await;

        assert!(matches!(result, Err(StreamTimeoutKind::Idle)));
    }

    #[tokio::test]
    async fn until_activity_reports_idle_timeout() {
        let deadline = StreamDeadline::new(Duration::from_secs(1), Duration::from_millis(10));

        let result = deadline.until_activity(std::future::pending::<()>()).await;

        assert!(matches!(result, Err(StreamTimeoutKind::Idle)));
    }

    #[tokio::test]
    async fn read_text_limited_reports_idle_timeout() {
        let deadline = StreamDeadline::new(Duration::from_secs(1), Duration::from_millis(10));
        let mut stream = stream::pending::<Result<Bytes, std::io::Error>>();

        let result = deadline.read_text_limited(&mut stream, 1024).await;

        assert!(matches!(
            result,
            Err(StreamBodyReadError::Timeout(StreamTimeoutKind::Idle))
        ));
    }

    #[tokio::test]
    async fn read_text_limited_rejects_oversized_body() {
        let deadline = StreamDeadline::new(Duration::from_secs(1), Duration::from_secs(1));
        let mut stream = stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(b"hello"))]);

        let result = deadline.read_text_limited(&mut stream, 4).await;

        assert!(matches!(
            result,
            Err(StreamBodyReadError::TooLarge { max_size: 4 })
        ));
    }
}
