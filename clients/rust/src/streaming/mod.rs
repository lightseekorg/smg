mod sse;
mod typed_stream;

pub(crate) use sse::sse_stream;
pub use sse::SseEvent;
pub use typed_stream::TypedStream;

/// Test helper — not part of the public API.
#[doc(hidden)]
pub fn __test_sse_stream(
    byte_stream: impl futures::stream::Stream<Item = Result<bytes::Bytes, reqwest::Error>>
        + Send
        + Unpin
        + 'static,
) -> std::pin::Pin<Box<dyn futures::stream::Stream<Item = Result<SseEvent, crate::SmgError>> + Send>>
{
    sse_stream(byte_stream)
}
