use serde_json::Value;
use thiserror::Error;

use crate::worker::Endpoint;

pub(crate) const SGLANG_FIELDS: &[&str] = &[
    "request_id",
    "priority",
    "top_k",
    "min_p",
    "min_tokens",
    "regex",
    "ebnf",
    "json_schema",
    "stop_token_ids",
    "no_stop_trim",
    "ignore_eos",
    "continue_final_message",
    "skip_special_tokens",
    "lora_path",
    "session_params",
    "separate_reasoning",
    "stream_reasoning",
    "chat_template",
    "chat_template_kwargs",
    "return_hidden_states",
    "repetition_penalty",
    "sampling_seed",
    "backend_url",
];

pub(crate) fn strip_sglang_fields(payload: &mut Value) {
    if let Some(obj) = payload.as_object_mut() {
        for field in SGLANG_FIELDS {
            obj.remove(*field);
        }
    }
}

#[derive(Error, Debug)]
pub enum ProviderError {
    #[error("Unsupported endpoint: {0:?}")]
    UnsupportedEndpoint(Endpoint),

    #[error("Transform error: {0}")]
    TransformError(String),
}
