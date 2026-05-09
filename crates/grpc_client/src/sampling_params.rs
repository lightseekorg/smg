//! Backend-neutral OpenAI → sampling-params builders shared by the SGLang
//! and TokenSpeed gRPC clients. Returns [`sglang::SamplingParams`] (the
//! most permissive shape today); other backends translate from this at
//! their wire seam.

use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ResponseFormat, StringOrArray},
    completion::CompletionRequest,
    messages::CreateMessageRequest,
    responses::ResponsesRequest,
    sampling_params::SamplingParams as GenerateSamplingParams,
};
use tracing::warn;

use crate::sglang_scheduler::proto;

/// Build gRPC `SamplingParams` from a `ChatCompletionRequest`.
pub fn build_grpc_sampling_params_from_chat(
    request: &ChatCompletionRequest,
    tool_call_constraint: Option<(String, String)>,
) -> Result<proto::SamplingParams, String> {
    let stop_sequences = extract_stop_strings(request);

    let max_new_tokens = request.max_completion_tokens;

    // Hardcode to true: gRPC backends return raw token IDs, not decoded text.
    // Detokenization happens on the SMG Rust side (StopDecoder/Sequence).
    //
    // Note: TokenSpeed's HTTP serving_chat sets this to false when tools are
    // present (serving_chat.py:178-179) — but mirroring that on the gRPC
    // path measurably HURTS BFCL accuracy. We tested it: simple_python
    // dropped from ~88.75 % to 79 %, parallel_multiple from ~84.5 % to
    // 60.5 %. With skip_special_tokens=false the engine emits the
    // ``<|tool_call_*|>`` special tokens in the raw output stream, and the
    // SMG-side detokenizer + kimik2 tool-call parser then double-counts or
    // misframes them. Keep it at true so SMG sees normal tokens and
    // applies its own parsing.
    let skip_special_tokens = true;

    Ok(proto::SamplingParams {
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(1.0),
        top_k: request.top_k.unwrap_or(-1),
        min_p: request.min_p.unwrap_or(0.0),
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
        presence_penalty: request.presence_penalty.unwrap_or(0.0),
        repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
        max_new_tokens,
        stop: stop_sequences,
        stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
        skip_special_tokens,
        spaces_between_special_tokens: true, // Default from Python SamplingParams
        ignore_eos: request.ignore_eos,
        no_stop_trim: request.no_stop_trim,
        n: request.n.unwrap_or(1),
        constraint: build_constraint_for_chat(request, tool_call_constraint)?,
        ..Default::default()
    })
}

/// Build gRPC `SamplingParams` from a `ResponsesRequest`.
///
/// Used by Harmony models only. Regular models use the Chat API path.
/// Constraints come from the Harmony preparation stage (`structural_tag`)
/// or tool handling.
pub fn build_grpc_sampling_params_from_responses(
    request: &ResponsesRequest,
    constraint: Option<(String, String)>,
) -> Result<proto::SamplingParams, String> {
    let max_new_tokens = request.max_output_tokens;

    Ok(proto::SamplingParams {
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(1.0),
        top_k: request.top_k,
        min_p: request.min_p,
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
        presence_penalty: request.presence_penalty.unwrap_or(0.0),
        repetition_penalty: request.repetition_penalty,
        max_new_tokens,
        stop: vec![],           // Does not pass through request.stop yet (follow-up fix)
        stop_token_ids: vec![], // Handled by Harmony stop tokens
        skip_special_tokens: false, // Keep special tokens for Harmony
        spaces_between_special_tokens: true,
        ignore_eos: false,
        no_stop_trim: false,
        n: 1, // Responses API doesn't support n>1
        constraint: build_constraint_for_responses(constraint)?,
        ..Default::default()
    })
}

/// Build gRPC `SamplingParams` from a `CreateMessageRequest` (Anthropic
/// Messages API).
pub fn build_grpc_sampling_params_from_messages(
    request: &CreateMessageRequest,
    tool_call_constraint: Option<(String, String)>,
) -> Result<proto::SamplingParams, String> {
    let stop_sequences = request.stop_sequences.clone().unwrap_or_default();

    // Hardcode to true: gRPC backends return raw token IDs, not decoded text.
    let skip_special_tokens = true;

    Ok(proto::SamplingParams {
        temperature: request.temperature.unwrap_or(1.0) as f32,
        top_p: request.top_p.unwrap_or(1.0) as f32,
        top_k: request.top_k.map(|v| v as i32).unwrap_or(-1),
        min_p: 0.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: 1.0,
        max_new_tokens: Some(request.max_tokens),
        stop: stop_sequences,
        stop_token_ids: vec![],
        skip_special_tokens,
        spaces_between_special_tokens: true,
        ignore_eos: false,
        no_stop_trim: false,
        n: 1,
        constraint: build_constraint_for_responses(tool_call_constraint)?,
        ..Default::default()
    })
}

/// Build gRPC `SamplingParams` from a `CompletionRequest`
/// (`/v1/completions`).
pub fn build_grpc_sampling_params_from_completion(
    request: &CompletionRequest,
) -> Result<proto::SamplingParams, String> {
    let stop_sequences = match &request.stop {
        Some(StringOrArray::String(s)) => vec![s.clone()],
        Some(StringOrArray::Array(arr)) => arr.clone(),
        None => vec![],
    };

    let constraint = build_single_constraint_from_completion(request)?;

    Ok(proto::SamplingParams {
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(1.0),
        top_k: request.top_k.unwrap_or(-1),
        min_p: request.min_p.unwrap_or(0.0),
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
        presence_penalty: request.presence_penalty.unwrap_or(0.0),
        repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
        max_new_tokens: request.max_tokens,
        min_new_tokens: request.min_tokens.unwrap_or(0),
        stop: stop_sequences,
        stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
        skip_special_tokens: request.skip_special_tokens,
        spaces_between_special_tokens: true,
        ignore_eos: request.ignore_eos,
        no_stop_trim: request.no_stop_trim,
        n: request.n.unwrap_or(1),
        constraint,
        ..Default::default()
    })
}

/// Build gRPC `SamplingParams` from the plain `GenerateSamplingParams`
/// shape used by `/generate`.
pub fn build_sampling_params_from_plain(
    params: Option<&GenerateSamplingParams>,
) -> Result<proto::SamplingParams, String> {
    let mut sampling = proto::SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        top_k: -1,
        repetition_penalty: 1.0,
        n: 1,
        skip_special_tokens: true,
        spaces_between_special_tokens: true,
        ..Default::default()
    };

    let Some(p) = params else {
        return Ok(sampling);
    };

    macro_rules! map_field {
        ($field:ident) => {
            if let Some(val) = p.$field {
                sampling.$field = val;
            }
        };
    }

    map_field!(temperature);
    map_field!(top_p);
    map_field!(top_k);
    map_field!(frequency_penalty);
    map_field!(presence_penalty);
    map_field!(repetition_penalty);
    map_field!(min_p);
    map_field!(ignore_eos);
    map_field!(skip_special_tokens);
    map_field!(no_stop_trim);

    if let Some(stop) = &p.stop {
        match stop {
            StringOrArray::String(s) => sampling.stop.push(s.clone()),
            StringOrArray::Array(arr) => sampling.stop.extend(arr.clone()),
        }
    }

    if let Some(stop_token_ids) = &p.stop_token_ids {
        sampling.stop_token_ids.clone_from(stop_token_ids);
    }

    sampling.max_new_tokens = p.max_new_tokens;

    if let Some(min_new_tokens) = p.min_new_tokens {
        sampling.min_new_tokens = min_new_tokens;
    }

    if let Some(n) = p.n {
        sampling.n = n;
    }

    sampling.constraint = build_single_constraint_from_plain(p)?;

    Ok(sampling)
}

// ---------------------------------------------------------------------------
// Constraint helpers
// ---------------------------------------------------------------------------

fn extract_stop_strings(request: &ChatCompletionRequest) -> Vec<String> {
    match &request.stop {
        Some(StringOrArray::String(s)) => vec![s.clone()],
        Some(StringOrArray::Array(arr)) => arr.clone(),
        None => vec![],
    }
}

fn build_constraint_for_chat(
    request: &ChatCompletionRequest,
    tool_call_constraint: Option<(String, String)>,
) -> Result<Option<proto::sampling_params::Constraint>, String> {
    let mut constraints = Vec::new();

    match &request.response_format {
        Some(ResponseFormat::JsonObject) => {
            let schema = serde_json::json!({"type": "object"});
            let schema_str = serde_json::to_string(&schema)
                .map_err(|e| format!("Failed to serialize JSON schema: {e}"))?;
            constraints.push(proto::sampling_params::Constraint::JsonSchema(schema_str));
        }
        Some(ResponseFormat::JsonSchema { json_schema }) => {
            let schema_str = serde_json::to_string(&json_schema.schema)
                .map_err(|e| format!("Failed to serialize JSON schema: {e}"))?;
            constraints.push(proto::sampling_params::Constraint::JsonSchema(schema_str));
        }
        Some(ResponseFormat::Text) | None => {}
    }

    if let Some(ebnf) = &request.ebnf {
        constraints.push(proto::sampling_params::Constraint::EbnfGrammar(
            ebnf.clone(),
        ));
    }

    if let Some(regex) = &request.regex {
        constraints.push(proto::sampling_params::Constraint::Regex(regex.clone()));
    }

    // If response_format already set a constraint, drop the tool constraint
    // (matches SGLang HTTP behavior where response_format takes priority).
    if let Some((constraint_type, constraint_value)) = tool_call_constraint {
        if constraints.is_empty() {
            let tool_constraint = match constraint_type.as_str() {
                "structural_tag" => {
                    proto::sampling_params::Constraint::StructuralTag(constraint_value)
                }
                "json_schema" => proto::sampling_params::Constraint::JsonSchema(constraint_value),
                "ebnf" => proto::sampling_params::Constraint::EbnfGrammar(constraint_value),
                "regex" => proto::sampling_params::Constraint::Regex(constraint_value),
                _ => return Err(format!("Unknown constraint type: {constraint_type}")),
            };
            constraints.push(tool_constraint);
        } else {
            warn!(
                "Constrained decoding is not compatible with tool calls, dropping tool constraint"
            );
        }
    }

    match constraints.len() {
        0 => Ok(None),
        1 => Ok(constraints.pop()),
        _ => Err("Multiple constraints are not allowed.".to_string()),
    }
}

fn build_constraint_for_responses(
    constraint: Option<(String, String)>,
) -> Result<Option<proto::sampling_params::Constraint>, String> {
    if let Some((constraint_type, constraint_value)) = constraint {
        let parsed_constraint = match constraint_type.as_str() {
            "structural_tag" => proto::sampling_params::Constraint::StructuralTag(constraint_value),
            "json_schema" => proto::sampling_params::Constraint::JsonSchema(constraint_value),
            "ebnf" => proto::sampling_params::Constraint::EbnfGrammar(constraint_value),
            "regex" => proto::sampling_params::Constraint::Regex(constraint_value),
            _ => return Err(format!("Unknown constraint type: {constraint_type}")),
        };
        Ok(Some(parsed_constraint))
    } else {
        Ok(None)
    }
}

fn build_single_constraint_from_completion(
    request: &CompletionRequest,
) -> Result<Option<proto::sampling_params::Constraint>, String> {
    let mut constraints = Vec::new();
    if let Some(json_schema) = &request.json_schema {
        constraints.push(proto::sampling_params::Constraint::JsonSchema(
            json_schema.clone(),
        ));
    }
    if let Some(regex) = &request.regex {
        constraints.push(proto::sampling_params::Constraint::Regex(regex.clone()));
    }
    if let Some(ebnf) = &request.ebnf {
        constraints.push(proto::sampling_params::Constraint::EbnfGrammar(
            ebnf.clone(),
        ));
    }

    match constraints.len() {
        0 => Ok(None),
        1 => Ok(constraints.pop()),
        _ => Err("Multiple structured constraints are not allowed".to_string()),
    }
}

fn build_single_constraint_from_plain(
    params: &GenerateSamplingParams,
) -> Result<Option<proto::sampling_params::Constraint>, String> {
    let mut constraints = Vec::new();
    if let Some(json_schema) = &params.json_schema {
        constraints.push(proto::sampling_params::Constraint::JsonSchema(
            json_schema.clone(),
        ));
    }
    if let Some(regex) = &params.regex {
        constraints.push(proto::sampling_params::Constraint::Regex(regex.clone()));
    }
    if let Some(ebnf) = &params.ebnf {
        constraints.push(proto::sampling_params::Constraint::EbnfGrammar(
            ebnf.clone(),
        ));
    }

    match constraints.len() {
        0 => Ok(None),
        1 => Ok(constraints.pop()),
        _ => Err("Multiple structured constraints are not allowed".to_string()),
    }
}
