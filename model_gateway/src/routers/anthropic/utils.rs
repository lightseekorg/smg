//! Shared utilities for Anthropic router
//!
//! This module contains common helper functions used across different
//! Anthropic API handlers (messages, models, etc.)

use std::sync::Arc;

use futures::StreamExt;

use crate::core::{model_card::ProviderType, Worker};

// ============================================================================
// Header Propagation
// ============================================================================

/// Check if header should be propagated to Anthropic backend
///
/// Only propagates authentication and Anthropic-specific headers.
/// This prevents leaking sensitive headers like cookies or internal routing info.
pub fn should_propagate_header(key: &str) -> bool {
    key.eq_ignore_ascii_case("authorization")
        || key.eq_ignore_ascii_case("x-api-key")
        || key.eq_ignore_ascii_case("anthropic-version")
        || key.eq_ignore_ascii_case("anthropic-beta")
}

// ============================================================================
// Worker Filtering
// ============================================================================

/// Get healthy Anthropic workers from the registry.
///
/// SECURITY: In multi-provider setups, this filters by Anthropic provider to prevent
/// credential leakage. Anthropic credentials (X-API-Key, Authorization) are never sent
/// to non-Anthropic workers (e.g., OpenAI, xAI).
///
/// In single-provider setups (all workers have the same or no provider), returns all healthy workers.
/// In multi-provider setups, returns only healthy workers with `ProviderType::Anthropic`.
pub fn get_healthy_anthropic_workers(
    worker_registry: &crate::core::WorkerRegistry,
) -> Vec<Arc<dyn Worker>> {
    let workers = worker_registry.get_workers_filtered(
        None, // model_id
        None, // worker_type
        None, // connection_mode
        None, // runtime_type
        true, // healthy_only
    );

    filter_by_anthropic_provider(workers)
}

/// Find the best worker for a given model.
///
/// Returns the least-loaded healthy Anthropic worker that supports the given model.
/// Uses [`get_healthy_anthropic_workers`] for provider-aware filtering.
pub fn find_best_worker_for_model(
    worker_registry: &crate::core::WorkerRegistry,
    model_id: &str,
) -> Option<Arc<dyn Worker>> {
    get_healthy_anthropic_workers(worker_registry)
        .into_iter()
        .filter(|w| w.supports_model(model_id))
        .min_by_key(|w| w.load())
}

// ============================================================================
// Response Body Reading
// ============================================================================

/// Result of reading a response body with size limit
pub enum ReadBodyResult {
    /// Successfully read the full body
    Ok(String),
    /// Body exceeded max size
    TooLarge,
    /// Error reading body
    Error(String),
}

/// Read a response body incrementally with a size limit.
///
/// SECURITY: This prevents DoS by avoiding unbounded buffering when
/// content-length is unknown (e.g., chunked transfer encoding).
pub async fn read_response_body_limited(
    response: reqwest::Response,
    max_size: usize,
) -> ReadBodyResult {
    let mut stream = response.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut total_size: usize = 0;

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                total_size += chunk.len();
                if total_size > max_size {
                    return ReadBodyResult::TooLarge;
                }
                buf.extend_from_slice(&chunk);
            }
            Err(e) => {
                return ReadBodyResult::Error(e.to_string());
            }
        }
    }

    // Decode the entire buffer at once to avoid corrupting multibyte UTF-8
    // sequences that may be split across chunk boundaries.
    match String::from_utf8(buf) {
        Ok(body) => ReadBodyResult::Ok(body),
        Err(e) => ReadBodyResult::Error(format!("invalid UTF-8 in response body: {}", e)),
    }
}

// ============================================================================
// Worker Filtering (Internal)
// ============================================================================

/// Filter workers to only include Anthropic workers in multi-provider setups.
///
/// Treats `None` (no provider configured) as a distinct provider value so that
/// a mix of `Some(...)` and `None` workers is recognized as multi-provider,
/// preventing Anthropic credentials from leaking to untagged workers.
fn filter_by_anthropic_provider(workers: Vec<Arc<dyn Worker>>) -> Vec<Arc<dyn Worker>> {
    // Early-exit pattern to detect multiple providers without HashSet allocation.
    // Track Option<ProviderType> so None vs Some(...) counts as different.
    let mut first_provider: Option<Option<ProviderType>> = None;
    let has_multiple_providers = workers.iter().any(|w| {
        let provider = w.default_provider().cloned();
        match first_provider {
            None => {
                first_provider = Some(provider);
                false
            }
            Some(ref first) => *first != provider,
        }
    });

    if has_multiple_providers {
        // Multi-provider setup: only use explicitly Anthropic workers
        workers
            .into_iter()
            .filter(|w| matches!(w.default_provider(), Some(ProviderType::Anthropic)))
            .collect()
    } else {
        // Single-provider or no-provider setup: use all workers
        workers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_propagate_header_authorization() {
        assert!(should_propagate_header("authorization"));
        assert!(should_propagate_header("Authorization"));
        assert!(should_propagate_header("AUTHORIZATION"));
    }

    #[test]
    fn test_should_propagate_header_api_key_case_insensitive() {
        assert!(should_propagate_header("x-api-key"));
        assert!(should_propagate_header("X-Api-Key"));
        assert!(should_propagate_header("X-API-KEY"));
    }

    #[test]
    fn test_should_propagate_header_anthropic_specific() {
        assert!(should_propagate_header("anthropic-version"));
        assert!(should_propagate_header("Anthropic-Version"));
        assert!(should_propagate_header("anthropic-beta"));
        assert!(should_propagate_header("Anthropic-Beta"));
    }

    #[test]
    fn test_should_not_propagate_sensitive_headers() {
        assert!(!should_propagate_header("cookie"));
        assert!(!should_propagate_header("Cookie"));
        assert!(!should_propagate_header("set-cookie"));
    }

    #[test]
    fn test_should_not_propagate_routing_headers() {
        assert!(!should_propagate_header("host"));
        assert!(!should_propagate_header("x-forwarded-for"));
        assert!(!should_propagate_header("x-real-ip"));
        assert!(!should_propagate_header("user-agent"));
        assert!(!should_propagate_header("content-length"));
    }
}
