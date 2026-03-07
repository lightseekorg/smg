//! Completion response processing stage: Wraps generate pipeline output as CompletionResponse
//! (non-streaming) or transforms the generate SSE stream to CompletionStreamResponse (streaming).

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Instant,
};

use async_trait::async_trait;
use axum::response::Response;
use bytes::Bytes;
use futures_util::StreamExt;
use openai_protocol::{
    common::{StringOrArray, Usage},
    completion::{
        CompletionChoice, CompletionRequest, CompletionResponse, CompletionStreamChoice,
        CompletionStreamResponse,
    },
    generate::GenerateFinishReason,
};
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::error;

use crate::{
    core::AttachedBody,
    routers::{
        error,
        grpc::{
            common::{responses::build_sse_response, stages::PipelineStage},
            context::{FinalResponse, RequestContext},
            regular::{processor, streaming},
        },
    },
};

/// Completion response processing stage
///
/// Delegates execution to the generate response processing pipeline, then
/// wraps the result in the OpenAI `/v1/completions` response format.
pub(crate) struct CompletionResponseProcessingStage {
    processor: processor::ResponseProcessor,
    streaming_processor: Arc<streaming::StreamingProcessor>,
}

impl CompletionResponseProcessingStage {
    pub fn new(
        processor: processor::ResponseProcessor,
        streaming_processor: Arc<streaming::StreamingProcessor>,
    ) -> Self {
        Self {
            processor,
            streaming_processor,
        }
    }
}

#[async_trait]
impl PipelineStage for CompletionResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        self.process_completion_response(ctx).await
    }

    fn name(&self) -> &'static str {
        "CompletionResponseProcessing"
    }
}

impl CompletionResponseProcessingStage {
    async fn process_completion_response(
        &self,
        ctx: &mut RequestContext,
    ) -> Result<Option<Response>, Response> {
        let start_time = Instant::now();
        let is_streaming = ctx.is_streaming();
        let completion_req = ctx
            .state
            .original_completion_request
            .clone()
            .ok_or_else(|| {
                error!(
                    function = "CompletionResponseProcessingStage::execute",
                    "original_completion_request not set — was CompletionPreparationStage skipped?"
                );
                error::internal_error(
                    "missing_completion_request",
                    "Original completion request not preserved by preparation stage",
                )
            })?;

        let execution_result = ctx.state.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "CompletionResponseProcessingStage::execute",
                "No execution result"
            );
            error::internal_error("no_execution_result", "No execution result")
        })?;

        let dispatch = ctx
            .state
            .dispatch
            .as_ref()
            .ok_or_else(|| {
                error!(
                    function = "CompletionResponseProcessingStage::execute",
                    "Dispatch metadata not set"
                );
                error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
            })?
            .clone();

        let tokenizer = ctx.tokenizer_arc().ok_or_else(|| {
            error!(
                function = "CompletionResponseProcessingStage::process_completion_response",
                "Tokenizer not cached in context"
            );
            error::internal_error(
                "tokenizer_not_cached",
                "Tokenizer not cached in context - preparation stage may have been skipped",
            )
        })?;

        if is_streaming {
            let response = self.streaming_processor.clone().process_streaming_generate(
                execution_result,
                ctx.generate_request_arc(),
                dispatch,
                tokenizer,
            );

            let response = match ctx.state.load_guards.take() {
                Some(guards) => AttachedBody::wrap_response(response, guards),
                None => response,
            };

            let created = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            let prompt_text = resolve_prompt_text(&completion_req);
            let echo_prompt = if completion_req.echo {
                Some(prompt_text)
            } else {
                None
            };

            let transformed = transform_generate_sse_to_completion_sse(
                response,
                completion_req.model.clone(),
                created,
                echo_prompt,
                completion_req.suffix.clone(),
            );

            return Ok(Some(transformed));
        }

        // Non-streaming
        let request_logprobs = ctx.generate_request().return_logprob.unwrap_or(false);
        let generate_request = ctx.generate_request_arc();

        let stop_decoder = ctx.state.response.stop_decoder.as_mut().ok_or_else(|| {
            error!(
                function = "CompletionResponseProcessingStage::execute",
                "Stop decoder not initialized"
            );
            error::internal_error(
                "stop_decoder_not_initialized",
                "Stop decoder not initialized",
            )
        })?;

        let gen_responses = self
            .processor
            .process_non_streaming_generate_response(
                execution_result,
                generate_request,
                dispatch,
                stop_decoder,
                request_logprobs,
                start_time,
            )
            .await?;

        let prompt_text = resolve_prompt_text(&completion_req);
        let completion_response = build_completion_response(
            gen_responses,
            &completion_req.model,
            &prompt_text,
            completion_req.echo,
            completion_req.suffix.as_deref(),
        );

        ctx.state.response.final_response = Some(FinalResponse::Completion(completion_response));
        Ok(None)
    }
}

fn resolve_prompt_text(req: &CompletionRequest) -> String {
    match &req.prompt {
        StringOrArray::String(s) => s.clone(),
        StringOrArray::Array(arr) => arr.join(""),
    }
}

/// Build a non-streaming `CompletionResponse` from typed generate results.
pub(crate) fn build_completion_response(
    gen_responses: Vec<openai_protocol::generate::GenerateResponse>,
    model: &str,
    prompt_text: &str,
    echo: bool,
    suffix: Option<&str>,
) -> CompletionResponse {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut total_prompt = 0u32;
    let mut total_completion = 0u32;

    let choices: Vec<CompletionChoice> = gen_responses
        .iter()
        .enumerate()
        .map(|(i, gen)| {
            total_prompt = total_prompt.max(gen.meta_info.prompt_tokens);
            total_completion += gen.meta_info.completion_tokens;

            let finish_reason = match &gen.meta_info.finish_reason {
                GenerateFinishReason::Length { .. } => "length",
                GenerateFinishReason::Stop { .. } => "stop",
                GenerateFinishReason::Other(_) => "stop",
            };

            let mut text = String::new();
            if echo {
                text.push_str(prompt_text);
            }
            text.push_str(&gen.text);
            if let Some(sfx) = suffix {
                text.push_str(sfx);
            }

            CompletionChoice {
                text,
                index: i as u32,
                logprobs: None,
                finish_reason: Some(finish_reason.to_string()),
                matched_stop: gen.meta_info.matched_stop.clone(),
            }
        })
        .collect();

    let request_id = gen_responses
        .first()
        .map(|g| g.meta_info.id.as_str())
        .unwrap_or("cmpl-unknown");

    CompletionResponse {
        id: request_id.to_string(),
        object: "text_completion".to_string(),
        created,
        model: model.to_string(),
        choices,
        usage: Some(Usage {
            prompt_tokens: total_prompt,
            completion_tokens: total_completion,
            total_tokens: total_prompt + total_completion,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }),
        system_fingerprint: None,
    }
}

/// Transform an SSE response from SGLang generate format to OpenAI CompletionStreamResponse.
///
/// SGLang emits accumulated text per chunk; this function computes deltas per index so
/// the output matches the OpenAI `/v1/completions` streaming contract.
///
fn transform_generate_sse_to_completion_sse(
    response: Response,
    model: String,
    created: u64,
    echo_prompt: Option<String>,
    suffix: Option<String>,
) -> Response {
    let (_, body) = response.into_parts();
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();

    #[expect(
        clippy::disallowed_methods,
        reason = "streaming transform is fire-and-forget; client disconnect terminates it"
    )]
    tokio::spawn(async move {
        let mut stream = body.into_data_stream();
        let mut prev_lens: HashMap<u32, usize> = HashMap::new();
        let mut request_id = String::from("cmpl-unknown");
        let mut echo_sent_for: HashSet<u32> = HashSet::new();
        let mut buf = String::new();

        while let Some(chunk_result) = stream.next().await {
            let bytes = match chunk_result {
                Ok(b) => b,
                Err(e) => {
                    error!("Completion stream read error: {e}");
                    break;
                }
            };

            buf.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(boundary) = buf.find("\n\n") {
                let event = buf[..boundary].trim().to_string();
                buf = buf[boundary + 2..].to_string();

                if event.is_empty() {
                    continue;
                }

                let Some(data) = event.strip_prefix("data: ") else {
                    continue;
                };

                if data == "[DONE]" {
                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                    continue;
                }

                let Ok(gen_chunk) = serde_json::from_str::<Value>(data) else {
                    continue;
                };

                if gen_chunk.get("error").is_some() {
                    let _ = tx.send(Ok(Bytes::from(format!("data: {data}\n\n"))));
                    continue;
                }

                let index = gen_chunk["index"].as_u64().unwrap_or(0) as u32;
                let accumulated_text = gen_chunk["text"].as_str().unwrap_or("");

                if request_id == "cmpl-unknown" {
                    if let Some(id) = gen_chunk["meta_info"]["id"].as_str() {
                        request_id = if let Some(base) = id.rsplit_once('-') {
                            format!("cmpl-{}", base.0)
                        } else {
                            format!("cmpl-{id}")
                        };
                    }
                }

                let prev_len = prev_lens.entry(index).or_insert(0);
                let mut delta = String::new();
                if echo_prompt.is_some() && echo_sent_for.insert(index) {
                    delta.push_str(echo_prompt.as_deref().unwrap_or(""));
                }
                if accumulated_text.len() > *prev_len {
                    delta.push_str(&accumulated_text[*prev_len..]);
                }
                *prev_len = accumulated_text.len();

                let finish_reason = match &gen_chunk["meta_info"]["finish_reason"] {
                    Value::Null => None,
                    Value::String(s) => match s.as_str() {
                        "length" | "stop" => Some(s.to_string()),
                        _ => Some("stop".to_string()),
                    },
                    obj => match obj["type"].as_str() {
                        Some("length") => Some("length".to_string()),
                        _ => Some("stop".to_string()),
                    },
                };

                if finish_reason.is_some() {
                    if let Some(ref sfx) = suffix {
                        delta.push_str(sfx);
                    }
                }

                let chunk = CompletionStreamResponse {
                    id: request_id.clone(),
                    object: "text_completion".to_string(),
                    created,
                    choices: vec![CompletionStreamChoice {
                        text: delta,
                        index,
                        logprobs: None,
                        finish_reason,
                    }],
                    model: model.clone(),
                    system_fingerprint: None,
                };

                if let Ok(sse_data) = serde_json::to_string(&chunk) {
                    if tx
                        .send(Ok(Bytes::from(format!("data: {sse_data}\n\n"))))
                        .is_err()
                    {
                        return;
                    }
                }
            }
        }
    });

    build_sse_response(rx)
}

#[cfg(test)]
mod tests {
    use openai_protocol::generate::{GenerateFinishReason, GenerateFinishType, GenerateMetaInfo};

    use super::*;

    fn make_gen_response(
        text: &str,
        finish: &str,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> openai_protocol::generate::GenerateResponse {
        openai_protocol::generate::GenerateResponse {
            text: text.to_string(),
            output_ids: vec![],
            meta_info: GenerateMetaInfo {
                id: "gen-test-123".to_string(),
                finish_reason: match finish {
                    "length" => GenerateFinishReason::Length {
                        finish_type: GenerateFinishType::Length,
                        length: completion_tokens,
                    },
                    _ => GenerateFinishReason::Stop {
                        finish_type: GenerateFinishType::Stop,
                    },
                },
                prompt_tokens,
                weight_version: "default".to_string(),
                input_token_logprobs: None,
                output_token_logprobs: None,
                completion_tokens,
                cached_tokens: 0,
                e2e_latency: 0.1,
                matched_stop: None,
            },
        }
    }

    #[test]
    fn test_build_completion_response_basic() {
        let gen = make_gen_response("Hello world", "stop", 5, 2);
        let resp = build_completion_response(vec![gen], "test-model", "", false, None);

        assert_eq!(resp.object, "text_completion");
        assert_eq!(resp.model, "test-model");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].text, "Hello world");
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(resp.usage.as_ref().unwrap().prompt_tokens, 5);
        assert_eq!(resp.usage.as_ref().unwrap().completion_tokens, 2);
    }

    #[test]
    fn test_build_completion_response_echo() {
        let gen = make_gen_response("world", "stop", 5, 1);
        let resp = build_completion_response(vec![gen], "m", "Hello ", true, None);
        assert_eq!(resp.choices[0].text, "Hello world");
    }

    #[test]
    fn test_build_completion_response_suffix() {
        let gen = make_gen_response("Hello", "stop", 1, 1);
        let resp = build_completion_response(vec![gen], "m", "", false, Some("[END]"));
        assert_eq!(resp.choices[0].text, "Hello[END]");
    }

    #[test]
    fn test_build_completion_response_echo_and_suffix() {
        let gen = make_gen_response("world", "stop", 5, 1);
        let resp = build_completion_response(vec![gen], "m", "Hello ", true, Some("!"));
        assert_eq!(resp.choices[0].text, "Hello world!");
    }

    #[test]
    fn test_build_completion_response_length_finish() {
        let gen = make_gen_response("partial", "length", 5, 3);
        let resp = build_completion_response(vec![gen], "m", "", false, None);
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn test_build_completion_response_multiple_choices() {
        let responses = vec![
            make_gen_response("first", "stop", 5, 1),
            make_gen_response("second", "length", 5, 2),
        ];
        let resp = build_completion_response(responses, "m", "", false, None);
        assert_eq!(resp.choices.len(), 2);
        assert_eq!(resp.choices[0].index, 0);
        assert_eq!(resp.choices[1].index, 1);
        assert_eq!(resp.choices[0].text, "first");
        assert_eq!(resp.choices[1].text, "second");
        assert_eq!(resp.usage.as_ref().unwrap().completion_tokens, 3);
    }
}
