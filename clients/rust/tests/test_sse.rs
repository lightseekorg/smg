use futures::StreamExt;
use smg_client::streaming::SseEvent;

/// Helper to create a byte stream from a string (simulating an HTTP response body).
fn bytes_stream(
    data: &str,
) -> impl futures::stream::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + Unpin + 'static
{
    let chunks = vec![Ok(bytes::Bytes::from(data.to_string()))];
    futures::stream::iter(chunks)
}

#[tokio::test]
async fn test_sse_parses_openai_format() {
    let raw = "data: {\"id\":\"1\",\"choices\":[]}\n\ndata: {\"id\":\"2\",\"choices\":[]}\n\ndata: [DONE]\n\n";
    let stream = smg_client::streaming::__test_sse_stream(bytes_stream(raw));
    let events: Vec<SseEvent> = stream.filter_map(|r| async { r.ok() }).collect().await;
    assert_eq!(events.len(), 2);
    assert!(events[0].data.contains("\"id\":\"1\""));
    assert!(events[1].data.contains("\"id\":\"2\""));
    assert!(events[0].event.is_none());
}

#[tokio::test]
async fn test_sse_parses_anthropic_format() {
    let raw = "event: message_start\ndata: {\"message\":{\"id\":\"msg_1\"}}\n\nevent: content_block_delta\ndata: {\"delta\":{\"text\":\"hi\"}}\n\n";
    let stream = smg_client::streaming::__test_sse_stream(bytes_stream(raw));
    let events: Vec<SseEvent> = stream.filter_map(|r| async { r.ok() }).collect().await;
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].event.as_deref(), Some("message_start"));
    assert_eq!(events[1].event.as_deref(), Some("content_block_delta"));
}
