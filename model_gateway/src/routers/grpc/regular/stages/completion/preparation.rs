//! Completion preparation stage: Convert CompletionRequest to GenerateRequest,
//! then delegate to the generate preparation stage.

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::{
    common::StringOrArray, completion::CompletionRequest, generate::GenerateRequest,
    sampling_params::SamplingParams,
};

use super::super::generate::GeneratePreparationStage;
use crate::routers::grpc::{
    common::stages::PipelineStage,
    context::{RequestContext, RequestType},
};

/// Completion preparation stage
///
/// Converts `CompletionRequest` into a `GenerateRequest`, replaces the context's
/// request type so downstream stages operate on a standard generate request,
/// then delegates to `GeneratePreparationStage` for tokenization and stop decoder setup.
pub(crate) struct CompletionPreparationStage {
    generate_stage: GeneratePreparationStage,
}

impl CompletionPreparationStage {
    pub fn new() -> Self {
        Self {
            generate_stage: GeneratePreparationStage,
        }
    }
}

#[async_trait]
impl PipelineStage for CompletionPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let completion_req = ctx.completion_request_arc();
        let gen_request = Arc::new(build_generate_from_completion(&completion_req));

        // Preserve original for response formatting, swap type so middle stages see Generate
        ctx.state.original_completion_request = Some(completion_req);
        ctx.input.request_type = RequestType::Generate(gen_request);

        self.generate_stage.execute(ctx).await
    }

    fn name(&self) -> &'static str {
        "CompletionPreparation"
    }
}

/// Convert a `CompletionRequest` into a `GenerateRequest`, mapping all applicable fields.
pub(crate) fn build_generate_from_completion(body: &CompletionRequest) -> GenerateRequest {
    let prompt_text = match &body.prompt {
        StringOrArray::String(s) => s.clone(),
        StringOrArray::Array(arr) => arr.join(""),
    };

    GenerateRequest {
        text: Some(prompt_text),
        model: Some(body.model.clone()),
        sampling_params: Some(SamplingParams {
            temperature: body.temperature,
            max_new_tokens: body.max_tokens,
            top_p: body.top_p,
            top_k: body.top_k,
            frequency_penalty: body.frequency_penalty,
            presence_penalty: body.presence_penalty,
            repetition_penalty: body.repetition_penalty,
            stop: body.stop.clone(),
            ignore_eos: Some(body.ignore_eos),
            n: body.n,
            min_p: body.min_p,
            min_new_tokens: body.min_tokens,
            regex: body.regex.clone(),
            ebnf: body.ebnf.clone(),
            json_schema: body.json_schema.clone(),
            stop_token_ids: body.stop_token_ids.clone(),
            no_stop_trim: Some(body.no_stop_trim),
            skip_special_tokens: Some(body.skip_special_tokens),
            sampling_seed: body.sampling_seed.or_else(|| body.seed.map(|s| s as u64)),
        }),
        stream: body.stream,
        return_logprob: body.logprobs.map(|_| true),
        top_logprobs_num: body.logprobs.map(|l| l as i32),
        lora_path: body.lora_path.clone(),
        session_params: body.session_params.clone(),
        return_hidden_states: body.return_hidden_states,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_completion_request() -> CompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "prompt": "Hello world",
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": false
        }))
        .unwrap()
    }

    #[test]
    fn test_basic_field_mapping() {
        let req = sample_completion_request();
        let gen = build_generate_from_completion(&req);

        assert_eq!(gen.text.as_deref(), Some("Hello world"));
        assert_eq!(gen.model.as_deref(), Some("test-model"));
        assert!(!gen.stream);

        let params = gen.sampling_params.unwrap();
        assert_eq!(params.max_new_tokens, Some(100));
        assert_eq!(params.temperature, Some(0.7));
    }

    #[test]
    fn test_prompt_array_joined() {
        let req: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "prompt": ["Hello", " ", "world"]
        }))
        .unwrap();
        let gen = build_generate_from_completion(&req);
        assert_eq!(gen.text.as_deref(), Some("Hello world"));
    }

    #[test]
    fn test_stream_flag_forwarded() {
        let req: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "prompt": "hi",
            "stream": true
        }))
        .unwrap();
        let gen = build_generate_from_completion(&req);
        assert!(gen.stream);
    }

    #[test]
    fn test_skip_special_tokens_forwarded() {
        let req: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "prompt": "hi",
            "skip_special_tokens": false
        }))
        .unwrap();
        let gen = build_generate_from_completion(&req);
        assert_eq!(
            gen.sampling_params.unwrap().skip_special_tokens,
            Some(false)
        );
    }

    #[test]
    fn test_sampling_seed_prefers_engine_specific() {
        let req: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "prompt": "hi",
            "seed": 10,
            "sampling_seed": 42
        }))
        .unwrap();
        let gen = build_generate_from_completion(&req);
        assert_eq!(gen.sampling_params.unwrap().sampling_seed, Some(42));
    }

    #[test]
    fn test_seed_fallback_when_no_sampling_seed() {
        let req: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "prompt": "hi",
            "seed": 10
        }))
        .unwrap();
        let gen = build_generate_from_completion(&req);
        assert_eq!(gen.sampling_params.unwrap().sampling_seed, Some(10));
    }

    #[test]
    fn test_logprobs_mapping() {
        let req: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "prompt": "hi",
            "logprobs": 5
        }))
        .unwrap();
        let gen = build_generate_from_completion(&req);
        assert_eq!(gen.return_logprob, Some(true));
        assert_eq!(gen.top_logprobs_num, Some(5));
    }

    #[test]
    fn test_lora_and_session_forwarded() {
        let req: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "prompt": "hi",
            "lora_path": "/path/to/lora"
        }))
        .unwrap();
        let gen = build_generate_from_completion(&req);
        assert_eq!(gen.lora_path.as_deref(), Some("/path/to/lora"));
    }
}
