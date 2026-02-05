//! Shared response formatting logic
//!
//! This module contains common logic for formatting responses, including:
//! - Usage calculation from gRPC responses
//! - ChatCompletionResponse construction

use std::collections::HashMap;

use crate::{
    protocols::common::Usage,
    routers::grpc::proto_wrapper::{ProtoGenerateComplete, ProtoGenerateStreamChunk},
};

/// Build usage information from collected gRPC responses
///
/// Sums prompt_tokens and completion_tokens across all responses.
/// Typically used with n>1 parameter where multiple completions are generated.
pub(crate) fn build_usage(responses: &[ProtoGenerateComplete]) -> Usage {
    let total_prompt_tokens: u32 = responses.iter().map(|r| r.prompt_tokens()).sum();
    let total_completion_tokens: u32 = responses.iter().map(|r| r.completion_tokens()).sum();

    Usage::from_counts(total_prompt_tokens, total_completion_tokens)
}

/// Tracks per-index completion token counts across streaming chunks.
///
/// Handles the vLLM vs SGLang difference:
/// - vLLM sends delta token counts per chunk (must accumulate)
/// - SGLang sends cumulative counts in the Complete message
pub(crate) struct CompletionTokenTracker {
    tokens: HashMap<u32, u32>,
}

impl CompletionTokenTracker {
    pub fn new() -> Self {
        Self {
            tokens: HashMap::new(),
        }
    }

    /// Record tokens from a streaming chunk.
    /// For vLLM, accumulates the chunk's token count.
    /// For SGLang/TRT-LLM, this is a no-op (they report in Complete).
    pub fn record_chunk(&mut self, chunk: &ProtoGenerateStreamChunk) {
        if chunk.is_vllm() {
            *self.tokens.entry(chunk.index()).or_insert(0) += chunk.token_ids().len() as u32;
        }
    }

    /// Record the final count from a Complete message.
    /// For vLLM, preserves the accumulated count.
    /// For SGLang/TRT-LLM, uses the cumulative value from Complete.
    pub fn record_complete(&mut self, complete: &ProtoGenerateComplete) {
        let index = complete.index();
        if complete.is_vllm() {
            // Keep accumulated count; ensure entry exists
            self.tokens.entry(index).or_insert(0);
        } else {
            self.tokens.insert(index, complete.completion_tokens());
        }
    }

    /// Get total completion tokens across all indices
    pub fn total(&self) -> u32 {
        self.tokens.values().sum()
    }
}
