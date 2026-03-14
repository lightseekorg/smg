use openai_protocol::{
    common::{StreamOptions, StringOrArray},
    completion::CompletionRequest,
    validated::Normalizable,
};
use serde_json::{Map, Value};
use validator::Validate;

fn base_request() -> CompletionRequest {
    CompletionRequest {
        model: "test-model".to_string(),
        prompt: StringOrArray::String("hello".to_string()),
        suffix: None,
        max_tokens: Some(16),
        temperature: None,
        top_p: None,
        n: None,
        stream: false,
        stream_options: None,
        logprobs: None,
        echo: false,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        best_of: None,
        logit_bias: None,
        user: None,
        seed: None,
        top_k: None,
        min_p: None,
        min_tokens: None,
        repetition_penalty: None,
        regex: None,
        ebnf: None,
        json_schema: None,
        stop_token_ids: None,
        no_stop_trim: false,
        ignore_eos: false,
        skip_special_tokens: true,
        lora_path: None,
        session_params: None,
        return_hidden_states: false,
        sampling_seed: None,
        other: Map::<String, Value>::new(),
    }
}

#[test]
fn test_completion_prompt_cannot_be_empty() {
    let mut req = base_request();
    req.prompt = StringOrArray::String(String::new());
    req.normalize();
    assert!(req.validate().is_err());
}

#[test]
fn test_completion_prompt_array_items_cannot_be_empty() {
    let mut req = base_request();
    req.prompt = StringOrArray::Array(vec!["first".to_string(), String::new()]);
    req.normalize();
    assert!(req.validate().is_err());
}

#[test]
fn test_completion_prompt_array_is_protocol_valid() {
    let mut req = base_request();
    req.prompt = StringOrArray::Array(vec!["first".to_string(), "second".to_string()]);
    req.normalize();
    assert!(req.validate().is_ok());
}

#[test]
fn test_completion_stream_options_require_stream() {
    let mut req = base_request();
    req.stream_options = Some(StreamOptions {
        include_usage: Some(true),
    });
    req.normalize();
    assert!(req.validate().is_err());
}

#[test]
fn test_completion_best_of_must_cover_n() {
    let mut req = base_request();
    req.n = Some(3);
    req.best_of = Some(2);
    req.normalize();
    assert!(req.validate().is_err());
}
