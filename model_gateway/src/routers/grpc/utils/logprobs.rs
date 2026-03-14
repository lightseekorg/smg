//! Proto-to-OpenAI logprob conversion functions.

use std::sync::Arc;

use llm_tokenizer::traits::Tokenizer;
use openai_protocol::common::{ChatLogProbs, ChatLogProbsContent, TopLogProb};

use crate::routers::grpc::proto_wrapper::{ProtoInputLogProbs, ProtoOutputLogProbs};

/// Convert OutputLogProbs to OpenAI ChatLogProbs format
///
/// Generic over the token decoding strategy. The `decode_token` closure maps a
/// single token ID to its text representation.
pub(crate) fn convert_proto_logprobs(
    proto_logprobs: &ProtoOutputLogProbs,
    decode_token: impl Fn(u32) -> String,
) -> ChatLogProbs {
    let mut content_items = Vec::with_capacity(proto_logprobs.token_logprobs.len());

    for (i, &logprob) in proto_logprobs.token_logprobs.iter().enumerate() {
        let token_id = proto_logprobs.token_ids.get(i).copied().unwrap_or(0);
        let token_text = decode_token(token_id);
        let bytes = Some(token_text.as_bytes().to_vec());

        // Build top_logprobs for this position
        let top_logprobs = if let Some(top_logprobs_entry) = proto_logprobs.top_logprobs.get(i) {
            top_logprobs_entry
                .values
                .iter()
                .enumerate()
                .filter_map(|(j, &top_logprob)| {
                    top_logprobs_entry.token_ids.get(j).map(|&tid| {
                        let text = decode_token(tid);
                        let bytes = Some(text.as_bytes().to_vec());
                        TopLogProb {
                            token: text,
                            logprob: top_logprob,
                            bytes,
                        }
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        content_items.push(ChatLogProbsContent {
            token: token_text,
            logprob,
            bytes,
            top_logprobs,
        });
    }

    ChatLogProbs::Detailed {
        content: (!content_items.is_empty()).then_some(content_items),
    }
}

/// Convert OutputLogProbs to OpenAI ChatLogProbs format using a Tokenizer
pub(crate) fn convert_proto_to_openai_logprobs(
    proto_logprobs: &ProtoOutputLogProbs,
    tokenizer: &Arc<dyn Tokenizer>,
) -> ChatLogProbs {
    convert_proto_logprobs(proto_logprobs, |token_id| {
        tokenizer
            .decode(&[token_id], false)
            .unwrap_or_else(|_| format!("<token_{token_id}>"))
    })
}

pub(crate) fn convert_proto_to_completion_logprobs(
    proto_logprobs: &ProtoOutputLogProbs,
    tokenizer: &Arc<dyn Tokenizer>,
    initial_text_offset: u32,
    skip_special_tokens: bool,
    visible_text: Option<&str>,
) -> openai_protocol::common::LogProbs {
    let mut tokens = Vec::with_capacity(proto_logprobs.token_logprobs.len());
    let mut token_logprobs = Vec::with_capacity(proto_logprobs.token_logprobs.len());
    let mut top_logprobs = Vec::with_capacity(proto_logprobs.token_logprobs.len());
    let mut text_offset = Vec::with_capacity(proto_logprobs.token_logprobs.len());

    let mut current_offset = initial_text_offset;
    let visible_limit = visible_text.map(str::len);
    let mut emitted_visible_len = 0usize;

    for (i, &logprob) in proto_logprobs.token_logprobs.iter().enumerate() {
        if visible_limit.is_some_and(|limit| emitted_visible_len >= limit) {
            break;
        }

        let token_id = proto_logprobs.token_ids.get(i).copied().unwrap_or(0);
        let token_text = tokenizer
            .decode(&[token_id], skip_special_tokens)
            .unwrap_or_else(|_| format!("<token_{token_id}>"));

        if token_text.is_empty() {
            continue;
        }

        if let Some(limit) = visible_limit {
            let next_visible_len = emitted_visible_len + token_text.len();
            if next_visible_len > limit {
                break;
            }
            emitted_visible_len = next_visible_len;
        }

        tokens.push(token_text.clone());
        token_logprobs.push(Some(logprob));
        text_offset.push(current_offset);
        current_offset += token_text.len() as u32;

        let top_logprobs_map = if let Some(top_logprobs_entry) = proto_logprobs.top_logprobs.get(i)
        {
            let mut map = std::collections::HashMap::new();
            for (j, &top_logprob) in top_logprobs_entry.values.iter().enumerate() {
                if let Some(&tid) = top_logprobs_entry.token_ids.get(j) {
                    let text = tokenizer
                        .decode(&[tid], skip_special_tokens)
                        .unwrap_or_else(|_| format!("<token_{tid}>"));
                    if text.is_empty() {
                        continue;
                    }
                    map.insert(text, top_logprob);
                }
            }
            Some(map)
        } else {
            None
        };
        top_logprobs.push(top_logprobs_map);
    }

    openai_protocol::common::LogProbs {
        tokens,
        token_logprobs,
        top_logprobs,
        text_offset,
    }
}

/// Convert OutputLogProbs to Generate format Vec<Vec<Option<f64>>>
///
/// Generate format: [[logprob, token_id, ...], [logprob, token_id, ...], ...]
/// Each inner vec contains [logprob (f64), token_id (u32), ...]
pub(crate) fn convert_generate_output_logprobs(
    proto_logprobs: &ProtoOutputLogProbs,
) -> Vec<Vec<Option<f64>>> {
    proto_logprobs
        .token_logprobs
        .iter()
        .zip(proto_logprobs.token_ids.iter())
        .map(|(&logprob, &token_id)| vec![Some(logprob as f64), Some(token_id as f64)])
        .collect()
}

/// Convert InputLogProbs to Generate format Vec<Vec<Option<f64>>>
///
/// Generate format: [[logprob, token_id, ...], [logprob, token_id, ...], ...]
/// First token has null logprob: [[null, token_id], [logprob, token_id], ...]
pub(crate) fn convert_generate_input_logprobs(
    proto_logprobs: &ProtoInputLogProbs,
) -> Vec<Vec<Option<f64>>> {
    proto_logprobs
        .token_logprobs
        .iter()
        .zip(proto_logprobs.token_ids.iter())
        .map(|(&token_logprob, &token_id)| {
            // token_logprob is Option<f32> in unified type
            let logprob_value = token_logprob.map(|v| v as f64);
            vec![logprob_value, Some(token_id as f64)]
        })
        .collect()
}
