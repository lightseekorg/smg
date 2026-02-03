//! Tests for Harmony gRPC streaming token counting logic
//!
//! These tests verify the token counting patterns used in the Harmony streaming
//! processor for different backends (vLLM vs SGLang).
//!
//! Background:
//! - vLLM sends delta token counts in chunks, and its GenerateComplete message
//!   may have incorrect completion_tokens values. We accumulate from chunk token_ids.
//! - SGLang sends cumulative values, so we use the GenerateComplete value directly.
//!
//! The tests simulate the logic used in HarmonyStreamingProcessor without requiring
//! access to private wrapper types.

use std::collections::HashMap;

use smg::grpc_client::{sglang_proto as sglang, vllm_proto as vllm};

/// Represents the backend type for token counting logic
#[derive(Clone, Copy, PartialEq)]
enum Backend {
    Vllm,
    Sglang,
}

/// Simulates chunk data from streaming
struct TestChunk {
    backend: Backend,
    token_ids: Vec<u32>,
    index: u32,
}

/// Simulates complete message data
struct TestComplete {
    backend: Backend,
    completion_tokens: u32,
    index: u32,
}

impl TestChunk {
    fn vllm(token_ids: Vec<u32>, index: u32) -> Self {
        Self {
            backend: Backend::Vllm,
            token_ids,
            index,
        }
    }

    fn sglang(token_ids: Vec<u32>, index: u32) -> Self {
        Self {
            backend: Backend::Sglang,
            token_ids,
            index,
        }
    }

    fn is_vllm(&self) -> bool {
        self.backend == Backend::Vllm
    }
}

impl TestComplete {
    fn vllm(completion_tokens: u32, index: u32) -> Self {
        Self {
            backend: Backend::Vllm,
            completion_tokens,
            index,
        }
    }

    fn sglang(completion_tokens: u32, index: u32) -> Self {
        Self {
            backend: Backend::Sglang,
            completion_tokens,
            index,
        }
    }

    fn is_vllm(&self) -> bool {
        self.backend == Backend::Vllm
    }
}

// ============================================================================
// Tests for the token counting logic pattern
// ============================================================================

#[test]
fn test_vllm_chunk_token_counting() {
    // Test that vLLM chunks accumulate token counts from token_ids.len()
    let chunk1 = TestChunk::vllm(vec![100, 101, 102], 0); // 3 tokens
    let chunk2 = TestChunk::vllm(vec![103, 104], 0); // 2 tokens

    assert!(chunk1.is_vllm());

    // Simulate the accumulation logic from process_single_stream
    let mut completion_tokens: HashMap<u32, u32> = HashMap::new();

    // Process chunk1: For vLLM, accumulate completion tokens (vLLM sends deltas)
    if chunk1.is_vllm() {
        *completion_tokens.entry(chunk1.index).or_insert(0) += chunk1.token_ids.len() as u32;
    }
    assert_eq!(*completion_tokens.get(&0).unwrap(), 3);

    // Process chunk2
    if chunk2.is_vllm() {
        *completion_tokens.entry(chunk2.index).or_insert(0) += chunk2.token_ids.len() as u32;
    }
    assert_eq!(*completion_tokens.get(&0).unwrap(), 5);
}

#[test]
fn test_vllm_complete_preserves_accumulated_count() {
    // Test that vLLM Complete does NOT overwrite the accumulated count
    let mut completion_tokens: HashMap<u32, u32> = HashMap::new();

    // Simulate accumulated count from chunks
    completion_tokens.insert(0, 25);

    // vLLM Complete with incorrect completion_tokens=1 (the bug we're fixing)
    let complete = TestComplete::vllm(1, 0);
    assert!(complete.is_vllm());

    // Apply the fixed logic: for vLLM, keep accumulated count
    if complete.is_vllm() {
        completion_tokens.entry(complete.index).or_insert(0);
    } else {
        completion_tokens.insert(complete.index, complete.completion_tokens);
    }

    // Should still be 25, not 1
    assert_eq!(*completion_tokens.get(&0).unwrap(), 25);
}

#[test]
fn test_sglang_complete_uses_complete_value() {
    // Test that SGLang Complete DOES use the completion_tokens from the message
    let mut completion_tokens: HashMap<u32, u32> = HashMap::new();

    // SGLang Complete with correct cumulative value
    let complete = TestComplete::sglang(50, 0);
    assert!(!complete.is_vllm());

    // Apply the fixed logic: for SGLang, use complete value
    if complete.is_vllm() {
        completion_tokens.entry(complete.index).or_insert(0);
    } else {
        completion_tokens.insert(complete.index, complete.completion_tokens);
    }

    // Should be 50 from the Complete message
    assert_eq!(*completion_tokens.get(&0).unwrap(), 50);
}

#[test]
fn test_sglang_chunk_skips_accumulation() {
    // Test that SGLang chunks do NOT accumulate (SGLang sends cumulative values)
    let chunk = TestChunk::sglang(vec![100, 101], 0);
    assert!(!chunk.is_vllm());

    let mut completion_tokens: HashMap<u32, u32> = HashMap::new();

    // Only accumulate for vLLM
    if chunk.is_vllm() {
        *completion_tokens.entry(chunk.index).or_insert(0) += chunk.token_ids.len() as u32;
    }

    // Should be empty since we skipped SGLang
    assert!(!completion_tokens.contains_key(&0));
}

#[test]
fn test_multi_index_vllm_streaming() {
    // Test n>1 support with multiple indices
    let chunks = [
        TestChunk::vllm(vec![100, 101], 0),
        TestChunk::vllm(vec![200, 201, 202], 1),
        TestChunk::vllm(vec![102, 103, 104], 0),
        TestChunk::vllm(vec![203], 1),
    ];

    let mut completion_tokens: HashMap<u32, u32> = HashMap::new();

    // Process all chunks
    for chunk in chunks {
        if chunk.is_vllm() {
            *completion_tokens.entry(chunk.index).or_insert(0) += chunk.token_ids.len() as u32;
        }
    }

    // Index 0: 2 + 3 = 5 tokens
    assert_eq!(*completion_tokens.get(&0).unwrap(), 5);
    // Index 1: 3 + 1 = 4 tokens
    assert_eq!(*completion_tokens.get(&1).unwrap(), 4);

    // Total
    let total: u32 = completion_tokens.values().sum();
    assert_eq!(total, 9);
}

#[test]
fn test_responses_api_vllm_token_tracking() {
    // Test Responses API single-value tracking (not HashMap)
    let chunk1 = TestChunk::vllm(vec![100, 101, 102], 0);
    let chunk2 = TestChunk::vllm(vec![103, 104], 0);

    let mut completion_tokens: u32 = 0;

    // Accumulate for vLLM
    if chunk1.is_vllm() {
        completion_tokens += chunk1.token_ids.len() as u32;
    }
    if chunk2.is_vllm() {
        completion_tokens += chunk2.token_ids.len() as u32;
    }

    assert_eq!(completion_tokens, 5);

    // vLLM Complete should NOT overwrite
    let complete = TestComplete::vllm(1, 0);
    if !complete.is_vllm() {
        completion_tokens = complete.completion_tokens;
    }

    // Should still be 5
    assert_eq!(completion_tokens, 5);
}

// ============================================================================
// Tests using actual proto types to verify field access
// ============================================================================

#[test]
fn test_vllm_proto_chunk_structure() {
    // Verify the vLLM proto chunk has the expected fields
    let chunk = vllm::GenerateStreamChunk {
        token_ids: vec![100, 101, 102],
        prompt_tokens: 10,
        completion_tokens: 1, // May be incorrect in vLLM
        cached_tokens: 0,
        output_logprobs: None,
        input_logprobs: None,
        index: 0,
    };

    assert_eq!(chunk.token_ids.len(), 3);
    assert_eq!(chunk.completion_tokens, 1);
}

#[test]
fn test_sglang_proto_chunk_structure() {
    // Verify the SGLang proto chunk has the expected fields
    let chunk = sglang::GenerateStreamChunk {
        token_ids: vec![100, 101],
        prompt_tokens: 10,
        completion_tokens: 5, // SGLang sends cumulative
        cached_tokens: 0,
        output_logprobs: None,
        hidden_states: vec![],
        input_logprobs: None,
        index: 0,
    };

    assert_eq!(chunk.token_ids.len(), 2);
    assert_eq!(chunk.completion_tokens, 5);
}

#[test]
fn test_vllm_proto_complete_structure() {
    // Verify the vLLM proto complete has the expected fields
    let complete = vllm::GenerateComplete {
        output_ids: vec![1, 2, 3],
        finish_reason: "stop".to_string(),
        prompt_tokens: 10,
        completion_tokens: 1, // May be incorrect
        cached_tokens: 0,
        output_logprobs: None,
        input_logprobs: None,
        index: 0,
    };

    assert_eq!(complete.completion_tokens, 1);
}

#[test]
fn test_sglang_proto_complete_structure() {
    // Verify the SGLang proto complete has the expected fields
    let complete = sglang::GenerateComplete {
        output_ids: vec![1, 2, 3, 4, 5],
        finish_reason: "stop".to_string(),
        prompt_tokens: 10,
        completion_tokens: 50, // Correct cumulative value
        cached_tokens: 0,
        output_logprobs: None,
        all_hidden_states: vec![],
        matched_stop: None,
        input_logprobs: None,
        index: 0,
    };

    assert_eq!(complete.completion_tokens, 50);
}
