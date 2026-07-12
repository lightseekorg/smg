use std::{future::Future, io};

use axum::{
    body::to_bytes,
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures_util::{future::join_all, stream, StreamExt};
use openai_protocol::{
    common::{StringOrArray, Usage},
    completion::{CompletionRequest, CompletionResponse, CompletionStreamResponse},
};
use tokio::sync::mpsc;
use uuid::Uuid;

use super::{common::responses::build_sse_response, utils};
use crate::routers::error;

struct BatchMetadata {
    id: String,
    created: u64,
    model: String,
}

impl BatchMetadata {
    fn new(model: &str) -> Self {
        Self {
            id: format!("cmpl_{}", Uuid::now_v7()),
            created: chrono::Utc::now().timestamp() as u64,
            model: model.to_string(),
        }
    }
}

pub(crate) async fn execute_batch<F, Fut>(
    request: &CompletionRequest,
    model: &str,
    execute: F,
) -> Option<Response>
where
    F: Fn(CompletionRequest) -> Fut,
    Fut: Future<Output = Response>,
{
    let StringOrArray::Array(prompts) = &request.prompt else {
        return None;
    };

    let requests = prompts.iter().map(|prompt| {
        let mut request = request.clone();
        request.prompt = StringOrArray::String(prompt.clone());
        request
    });
    let metadata = BatchMetadata::new(model);
    let responses = join_all(requests.map(execute)).await;
    Some(merge_batch_responses(responses, request.n.unwrap_or(1), metadata, request.stream).await)
}

async fn merge_batch_responses(
    responses: Vec<Response>,
    choices_per_prompt: u32,
    metadata: BatchMetadata,
    streaming: bool,
) -> Response {
    let mut successful = Vec::with_capacity(responses.len());
    for response in responses {
        if !response.status().is_success() {
            return response;
        }
        successful.push(response);
    }

    if successful.is_empty() {
        return error::internal_error("empty_completion_batch", "Completion batch was empty");
    }

    if streaming {
        merge_streaming_responses(successful, choices_per_prompt, metadata)
    } else {
        merge_non_streaming_responses(successful, choices_per_prompt, metadata).await
    }
}

async fn merge_non_streaming_responses(
    responses: Vec<Response>,
    choices_per_prompt: u32,
    metadata: BatchMetadata,
) -> Response {
    let mut choices = Vec::new();
    let mut usage = UsageAccumulator::default();
    let mut system_fingerprint = None;

    for (prompt_index, response) in responses.into_iter().enumerate() {
        let body = match to_bytes(response.into_body(), usize::MAX).await {
            Ok(body) => body,
            Err(err) => {
                return error::internal_error(
                    "completion_batch_body_read_failed",
                    format!("Failed to read completion response: {err}"),
                )
            }
        };
        let response = match serde_json::from_slice::<CompletionResponse>(&body) {
            Ok(response) => response,
            Err(err) => {
                return error::internal_error(
                    "completion_batch_response_invalid",
                    format!("Failed to parse completion response: {err}"),
                )
            }
        };

        if system_fingerprint.is_none() {
            system_fingerprint = response.system_fingerprint;
        }
        if let Some(response_usage) = response.usage {
            usage.add(response_usage);
        }

        let offset = u32::try_from(prompt_index)
            .unwrap_or(u32::MAX)
            .saturating_mul(choices_per_prompt);
        choices.extend(response.choices.into_iter().map(|mut choice| {
            choice.index = offset.saturating_add(choice.index);
            choice
        }));
    }

    Json(CompletionResponse {
        id: metadata.id,
        object: "text_completion".to_string(),
        created: metadata.created,
        model: metadata.model,
        choices,
        usage: usage.finish(),
        system_fingerprint,
    })
    .into_response()
}

fn merge_streaming_responses(
    responses: Vec<Response>,
    choices_per_prompt: u32,
    metadata: BatchMetadata,
) -> Response {
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

    #[expect(
        clippy::disallowed_methods,
        reason = "the merged response body owns cancellation of all source streams"
    )]
    tokio::spawn(async move {
        if let Err(err) = forward_streams(responses, choices_per_prompt, &metadata, &tx).await {
            utils::send_error_sse(&tx, err, "stream_error");
        }
        let _ = tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n")));
    });

    build_sse_response(rx)
}

async fn forward_streams(
    responses: Vec<Response>,
    choices_per_prompt: u32,
    metadata: &BatchMetadata,
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
) -> Result<(), String> {
    let streams = responses
        .into_iter()
        .enumerate()
        .map(|(prompt_index, response)| {
            response
                .into_body()
                .into_data_stream()
                .map(move |chunk| (prompt_index, chunk))
                .boxed()
        });
    let mut streams = stream::select_all(streams);
    let mut usage = UsageAccumulator::default();
    let mut system_fingerprint = None;

    while let Some((prompt_index, chunk)) = streams.next().await {
        let chunk = chunk.map_err(|err| format!("Completion stream read failed: {err}"))?;
        let event = String::from_utf8_lossy(&chunk);
        let event = event.trim();
        if event == "data: [DONE]" {
            continue;
        }

        let Some(json) = event.strip_prefix("data: ") else {
            tx.send(Ok(chunk))
                .map_err(|_| "Client disconnected".to_string())?;
            continue;
        };
        let Ok(mut response) = serde_json::from_str::<CompletionStreamResponse>(json.trim()) else {
            tx.send(Ok(chunk))
                .map_err(|_| "Client disconnected".to_string())?;
            continue;
        };

        if system_fingerprint.is_none() {
            system_fingerprint = response.system_fingerprint.clone();
        }
        if let Some(response_usage) = response.usage.take() {
            usage.add(response_usage);
        }

        let offset = u32::try_from(prompt_index)
            .unwrap_or(u32::MAX)
            .saturating_mul(choices_per_prompt);
        for choice in &mut response.choices {
            choice.index = offset.saturating_add(choice.index);
        }
        if response.choices.is_empty() {
            continue;
        }

        response.id.clone_from(&metadata.id);
        response.created = metadata.created;
        response.model.clone_from(&metadata.model);
        send_stream_event(tx, &response)?;
    }

    if let Some(usage) = usage.finish() {
        send_stream_event(
            tx,
            &CompletionStreamResponse {
                id: metadata.id.clone(),
                object: "text_completion".to_string(),
                created: metadata.created,
                choices: vec![],
                model: metadata.model.clone(),
                system_fingerprint,
                usage: Some(usage),
            },
        )?;
    }

    Ok(())
}

fn send_stream_event(
    tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    response: &CompletionStreamResponse,
) -> Result<(), String> {
    let json = serde_json::to_vec(response)
        .map_err(|err| format!("Failed to serialize completion response: {err}"))?;
    let mut event = Vec::with_capacity(json.len() + 8);
    event.extend_from_slice(b"data: ");
    event.extend_from_slice(&json);
    event.extend_from_slice(b"\n\n");
    tx.send(Ok(Bytes::from(event)))
        .map_err(|_| "Client disconnected".to_string())
}

#[derive(Default)]
struct UsageAccumulator {
    seen: bool,
    prompt: u32,
    completion: u32,
    cached: u32,
    reasoning: u32,
}

impl UsageAccumulator {
    fn add(&mut self, usage: Usage) {
        self.seen = true;
        self.prompt = self.prompt.saturating_add(usage.prompt_tokens);
        self.completion = self.completion.saturating_add(usage.completion_tokens);
        self.cached = self.cached.saturating_add(
            usage
                .prompt_tokens_details
                .map_or(0, |details| details.cached_tokens),
        );
        self.reasoning = self.reasoning.saturating_add(
            usage
                .completion_tokens_details
                .and_then(|details| details.reasoning_tokens)
                .unwrap_or_default(),
        );
    }

    fn finish(self) -> Option<Usage> {
        self.seen.then(|| {
            Usage::from_counts(self.prompt, self.completion)
                .with_cached_tokens(self.cached)
                .with_reasoning_tokens(self.reasoning)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use axum::{
        body::{to_bytes, Body},
        response::{IntoResponse, Response},
        Json,
    };
    use bytes::Bytes;
    use futures_util::stream;
    use openai_protocol::{
        common::StringOrArray,
        completion::{CompletionRequest, CompletionResponse, CompletionStreamResponse},
    };

    use super::execute_batch;

    fn request(stream: bool) -> CompletionRequest {
        let mut request = serde_json::json!({
            "model": "test-model",
            "prompt": ["first", "second"],
            "echo": true,
            "n": 2,
            "stream": stream
        });
        if stream {
            request["stream_options"] = serde_json::json!({"include_usage": true});
        }
        serde_json::from_value(request).unwrap()
    }

    fn prompt(request: CompletionRequest) -> String {
        assert!(request.echo);
        assert_eq!(request.n, Some(2));
        let StringOrArray::String(prompt) = request.prompt else {
            panic!("batch entry was not scalar")
        };
        prompt
    }

    #[tokio::test]
    async fn fans_out_and_merges_non_streaming_responses() {
        let merged = execute_batch(&request(false), "test-model", |request| async move {
            let prompt = prompt(request);
            let (prompt_tokens, completion_tokens) =
                if prompt == "first" { (3, 5) } else { (4, 6) };
            let choices = [0, 1].map(|index| {
                serde_json::json!({"text": index.to_string(), "index": index,
                                   "finish_reason": "stop"})
            });
            Json(serde_json::json!({
                "id": prompt,
                "object": "text_completion",
                "created": 1,
                "model": "test-model",
                "choices": choices,
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens}
            }))
            .into_response()
        })
        .await;
        let merged = merged.unwrap();
        let body = to_bytes(merged.into_body(), usize::MAX).await.unwrap();
        let result: CompletionResponse = serde_json::from_slice(&body).unwrap();

        assert!(result.id.starts_with("cmpl_"));
        assert_eq!(
            result
                .choices
                .iter()
                .map(|choice| choice.index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        let usage = result.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 7);
        assert_eq!(usage.completion_tokens, 11);
        assert_eq!(usage.total_tokens, 18);
    }

    fn streaming_response(request: CompletionRequest) -> Response {
        let prompt = prompt(request);
        let (prompt_tokens, completion_tokens) = if prompt == "first" { (3, 5) } else { (4, 6) };
        let events = [
            serde_json::json!({"id": prompt, "object": "text_completion", "created": 1,
                              "model": "test-model", "choices": [{"text": prompt,
                              "index": 0, "finish_reason": "stop"}]}),
            serde_json::json!({"id": prompt, "object": "text_completion", "created": 1,
                              "model": "test-model", "choices": [], "usage": {
                              "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                              "total_tokens": prompt_tokens + completion_tokens}}),
        ]
        .map(|event| Ok::<Bytes, Infallible>(Bytes::from(format!("data: {event}\n\n"))))
        .into_iter()
        .chain(std::iter::once(Ok(Bytes::from_static(b"data: [DONE]\n\n"))));
        Response::new(Body::from_stream(stream::iter(events)))
    }

    #[tokio::test]
    async fn multiplexes_streams_with_global_indices_and_one_done() {
        let merged = execute_batch(&request(true), "test-model", |request| async move {
            streaming_response(request)
        })
        .await
        .unwrap();
        let body = to_bytes(merged.into_body(), usize::MAX).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        let mut chunks = Vec::new();
        let mut done = 0;

        for event in body.split("\n\n").filter(|event| !event.is_empty()) {
            let data = event.strip_prefix("data: ").unwrap();
            if data == "[DONE]" {
                done += 1;
            } else {
                chunks.push(serde_json::from_str::<CompletionStreamResponse>(data).unwrap());
            }
        }

        assert_eq!(done, 1);
        assert!(chunks.iter().all(|chunk| chunk.id == chunks[0].id));
        let mut indices = chunks
            .iter()
            .flat_map(|chunk| chunk.choices.iter().map(|choice| choice.index))
            .collect::<Vec<_>>();
        indices.sort_unstable();
        assert_eq!(indices, vec![0, 2]);
        let usage = chunks.into_iter().find_map(|chunk| chunk.usage).unwrap();
        assert_eq!(usage.prompt_tokens, 7);
        assert_eq!(usage.completion_tokens, 11);
        assert_eq!(usage.total_tokens, 18);
    }
}
