//! Shared utilities for Anthropic router
//!
//! This module contains common helper functions used across different
//! Anthropic API handlers (messages, models, etc.)

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
