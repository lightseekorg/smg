use std::collections::HashMap;

use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, HeaderValue},
};
use http::header::HeaderName;
use serde::Deserialize;
use tracing::debug;

static HEADER_TARGET_WORKER: HeaderName = HeaderName::from_static("x-smg-target-worker");
static HEADER_ROUTING_KEY: HeaderName = HeaderName::from_static("x-smg-routing-key");
static HEADER_MCP: HeaderName = HeaderName::from_static("x-smg-mcp");

/// Parsed and normalized memory-related request headers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryHeaderView {
    /// Normalized LTM policy value consumed by memory execution context.
    pub policy: Option<String>,
    pub subject_id: Option<String>,
    pub embedding_model: Option<String>,
    pub extraction_model: Option<String>,
}

impl MemoryHeaderView {
    /// Extract memory request settings from `x-conversation-memory-config`.
    ///
    /// `long_term_memory.enabled` is the top-level gate:
    /// - `false`: LTM policy/settings are treated as unset.
    /// - `true`: policy is read from `long_term_memory.policy` and defaults to
    ///   `none` when omitted to preserve privacy by default.
    pub fn from_http_headers(headers: &HeaderMap) -> Self {
        let Some(config) = extract_conversation_memory_config(Some(headers)) else {
            return Self::default();
        };
        let ltm_enabled = config.long_term_memory.enabled;
        let policy = if ltm_enabled {
            config
                .long_term_memory
                .policy
                .or_else(|| Some("none".to_string()))
        } else {
            None
        };
        Self {
            policy,
            subject_id: ltm_enabled
                .then_some(config.long_term_memory.subject_id)
                .flatten(),
            embedding_model: ltm_enabled
                .then_some(config.long_term_memory.embedding_model_id)
                .flatten(),
            extraction_model: ltm_enabled
                .then_some(config.long_term_memory.extraction_model_id)
                .flatten(),
        }
    }
}

fn extract_header_value<'a>(headers: Option<&'a HeaderMap>, name: &HeaderName) -> Option<&'a str> {
    headers
        .and_then(|h| h.get(name))
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
}

pub fn extract_target_worker(headers: Option<&HeaderMap>) -> Option<&str> {
    extract_header_value(headers, &HEADER_TARGET_WORKER)
}

pub fn extract_routing_key(headers: Option<&HeaderMap>) -> Option<&str> {
    extract_header_value(headers, &HEADER_ROUTING_KEY)
}

/// Check if SMG MCP orchestration is enabled via `X-SMG-MCP: enabled` header.
pub fn is_smg_mcp_enabled(headers: Option<&HeaderMap>) -> bool {
    headers
        .and_then(|h| h.get(&HEADER_MCP))
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.eq_ignore_ascii_case("enabled"))
}

/// Copy request headers to a Vec of name-value string pairs
/// Used for forwarding headers to backend workers
pub fn copy_request_headers(req: &Request<Body>) -> Vec<(String, String)> {
    req.headers()
        .iter()
        .filter_map(|(name, value)| {
            // Convert header value to string, skipping non-UTF8 headers
            value
                .to_str()
                .ok()
                .map(|v| (name.to_string(), v.to_string()))
        })
        .collect()
}

/// Convert headers from reqwest Response to axum HeaderMap
/// Filters out hop-by-hop headers that shouldn't be forwarded
pub fn preserve_response_headers(reqwest_headers: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();

    for (name, value) in reqwest_headers {
        // Skip hop-by-hop headers that shouldn't be forwarded
        // Use eq_ignore_ascii_case to avoid string allocation
        if should_forward_header_no_alloc(name.as_str()) {
            // The original name and value are already valid, so we can just clone them
            headers.insert(name.clone(), value.clone());
        }
    }

    headers
}

/// Determine if a header should be forwarded without allocating (case-insensitive)
fn should_forward_header_no_alloc(name: &str) -> bool {
    // List of headers that should NOT be forwarded (hop-by-hop headers)
    // Use eq_ignore_ascii_case to avoid to_lowercase() allocation
    !(name.eq_ignore_ascii_case("connection")
        || name.eq_ignore_ascii_case("keep-alive")
        || name.eq_ignore_ascii_case("proxy-authenticate")
        || name.eq_ignore_ascii_case("proxy-authorization")
        || name.eq_ignore_ascii_case("te")
        || name.eq_ignore_ascii_case("trailers")
        || name.eq_ignore_ascii_case("transfer-encoding")
        || name.eq_ignore_ascii_case("upgrade")
        || name.eq_ignore_ascii_case("content-encoding")
        || name.eq_ignore_ascii_case("host"))
}

/// API provider types for provider-specific header handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiProvider {
    Anthropic,
    Xai,
    OpenAi,
    Gemini,
    Generic,
}

impl ApiProvider {
    /// Detect provider type from URL
    pub fn from_url(url: &str) -> Self {
        if url.contains("anthropic") {
            ApiProvider::Anthropic
        } else if url.contains("x.ai") {
            ApiProvider::Xai
        } else if url.contains("openai.com") {
            ApiProvider::OpenAi
        } else if url.contains("googleapis.com") {
            ApiProvider::Gemini
        } else {
            ApiProvider::Generic
        }
    }

    /// Extract auth credential from request headers with provider-specific logic.
    ///
    /// - **Gemini**: prefers `x-goog-api-key`, then `Authorization`, then worker key.
    /// - **Anthropic**: prefers `x-api-key`, then `Authorization`, then worker key.
    /// - **All others**: prefers `Authorization`, then worker key with `Bearer` prefix.
    pub fn extract_auth_header(
        self,
        headers: Option<&HeaderMap>,
        worker_api_key: Option<&String>,
    ) -> Option<HeaderValue> {
        if let Some(h) = headers {
            match self {
                ApiProvider::Anthropic => {
                    // Prefer x-api-key
                    if let Some(v) = h.get("x-api-key").and_then(|v| {
                        v.to_str()
                            .ok()
                            .filter(|s| !s.trim().is_empty())
                            .map(|_| v.clone())
                    }) {
                        return Some(v);
                    }
                }
                ApiProvider::Gemini => {
                    // Prefer x-goog-api-key
                    if let Some(v) = h.get("x-goog-api-key").and_then(|v| {
                        v.to_str()
                            .ok()
                            .filter(|s| !s.trim().is_empty())
                            .map(|_| v.clone())
                    }) {
                        return Some(v);
                    }
                }
                _ => {}
            }
        }

        // Standard: Authorization header first, then worker key with Bearer
        extract_auth_header(headers, worker_api_key)
    }

    /// Apply provider-specific auth headers to a reqwest request builder.
    ///
    /// - **Anthropic**: strips `Bearer` prefix and sets `x-api-key` + `anthropic-version`.
    /// - **Gemini**: strips `Bearer` prefix and sets `x-goog-api-key`.
    /// - **Others**: forwards the `Authorization` header as-is.
    pub fn apply_headers(
        self,
        mut req: reqwest::RequestBuilder,
        auth_header: Option<&HeaderValue>,
    ) -> reqwest::RequestBuilder {
        match self {
            ApiProvider::Anthropic => {
                if let Some(auth) = auth_header {
                    if let Ok(auth_str) = auth.to_str() {
                        // Strip Bearer scheme case-insensitively (RFC 7235)
                        let api_key = auth_str
                            .split_once(' ')
                            .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
                            .map(|(_, token)| token)
                            .unwrap_or(auth_str)
                            .trim();
                        if !api_key.is_empty() {
                            req = req
                                .header("x-api-key", api_key)
                                .header("anthropic-version", "2023-06-01");
                        }
                    }
                }
            }
            ApiProvider::Gemini => {
                if let Some(auth) = auth_header {
                    if let Ok(auth_str) = auth.to_str() {
                        let api_key = auth_str
                            .split_once(' ')
                            .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
                            .map(|(_, token)| token)
                            .unwrap_or(auth_str)
                            .trim();
                        if !api_key.is_empty() {
                            req = req.header("x-goog-api-key", api_key);
                        }
                    }
                }
            }
            ApiProvider::Xai | ApiProvider::OpenAi | ApiProvider::Generic => {
                if let Some(auth) = auth_header {
                    req = req.header("Authorization", auth);
                }
            }
        }

        req
    }
}

/// Apply provider-specific headers to request
pub fn apply_provider_headers(
    req: reqwest::RequestBuilder,
    url: &str,
    auth_header: Option<&HeaderValue>,
) -> reqwest::RequestBuilder {
    ApiProvider::from_url(url).apply_headers(req, auth_header)
}

/// Extract auth header with passthrough semantics.
///
/// Passthrough mode: User's Authorization header takes priority.
/// Fallback: Worker's API key is used only if user didn't provide auth.
///
/// This enables use cases where:
/// 1. Users send their own API keys (multi-tenant, BYOK)
/// 2. Router has a default key for users who don't provide one
pub fn extract_auth_header(
    headers: Option<&HeaderMap>,
    worker_api_key: Option<&String>,
) -> Option<HeaderValue> {
    // Passthrough: Try user's auth header first
    let user_auth = headers.and_then(|h| {
        h.get("authorization")
            .or_else(|| h.get("Authorization"))
            .cloned()
    });

    // Return user's auth if provided, otherwise use worker's API key
    user_auth
        .or_else(|| worker_api_key.and_then(|k| HeaderValue::from_str(&format!("Bearer {k}")).ok()))
}

/// Extract the subset of request headers that SMG is allowed to preserve for
/// internal execution paths such as MCP tool calls.
pub fn extract_forwardable_request_headers(headers: Option<&HeaderMap>) -> HashMap<String, String> {
    let Some(headers) = headers else {
        return HashMap::new();
    };

    let mut forwarded = HashMap::new();

    for (name, value) in headers {
        if !should_forward_request_header(name.as_str()) {
            continue;
        }

        let Ok(value) = value.to_str() else {
            continue;
        };

        forwarded
            .entry(name.as_str().to_string())
            .and_modify(|existing: &mut String| {
                existing.push_str(", ");
                existing.push_str(value);
            })
            .or_insert_with(|| value.to_string());
    }

    forwarded
}

#[inline]
pub fn should_forward_request_header(name: &str) -> bool {
    const REQUEST_ID_PREFIX: &str = "x-request-id-";

    name.eq_ignore_ascii_case("authorization")
        || name.eq_ignore_ascii_case("x-request-id")
        || name.eq_ignore_ascii_case("x-correlation-id")
        || name.eq_ignore_ascii_case("traceparent")
        || name.eq_ignore_ascii_case("tracestate")
        || name.eq_ignore_ascii_case("x-smg-routing-key")
        || name
            .get(..REQUEST_ID_PREFIX.len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(REQUEST_ID_PREFIX))
}

// ── Conversation memory config ────────────────────────────────────────────────

static HEADER_CONVERSATION_MEMORY_CONFIG: HeaderName =
    HeaderName::from_static("x-conversation-memory-config");

/// Memory configuration parsed from the `x-conversation-memory-config` request header.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ConversationMemoryConfig {
    #[serde(default)]
    pub long_term_memory: LongTermMemoryConfig,
    #[serde(default)]
    pub short_term_memory: ShortTermMemoryConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct LongTermMemoryConfig {
    #[serde(default)]
    pub enabled: bool,
    pub policy: Option<String>,
    pub subject_id: Option<String>,
    pub embedding_model_id: Option<String>,
    pub extraction_model_id: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ShortTermMemoryConfig {
    #[serde(default)]
    pub enabled: bool,
    pub condenser_model_id: Option<String>,
}

/// Extract memory configuration from the `x-conversation-memory-config` JSON header.
///
/// Returns `None` when the header is absent or unparsable — callers should skip
/// memory injection entirely so the common (no-memory) path is zero-cost.
pub(crate) fn extract_conversation_memory_config(
    headers: Option<&HeaderMap>,
) -> Option<ConversationMemoryConfig> {
    let value = headers.and_then(|h| h.get(&HEADER_CONVERSATION_MEMORY_CONFIG))?;

    let raw = match value.to_str() {
        Ok(s) if !s.is_empty() => s,
        _ => {
            debug!("Invalid or empty x-conversation-memory-config header; ignoring");
            return None;
        }
    };

    match serde_json::from_str::<ConversationMemoryConfig>(raw) {
        Ok(mut cfg) => {
            cfg.long_term_memory.policy = normalize_optional_string(cfg.long_term_memory.policy);
            cfg.long_term_memory.subject_id =
                normalize_optional_string(cfg.long_term_memory.subject_id);
            cfg.long_term_memory.embedding_model_id =
                normalize_optional_string(cfg.long_term_memory.embedding_model_id);
            cfg.long_term_memory.extraction_model_id =
                normalize_optional_string(cfg.long_term_memory.extraction_model_id);
            cfg.short_term_memory.condenser_model_id =
                normalize_optional_string(cfg.short_term_memory.condenser_model_id);
            Some(cfg)
        }
        Err(e) => {
            debug!(error = %e, "Failed to parse x-conversation-memory-config header; ignoring");
            None
        }
    }
}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_header_value_returns_value() {
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-routing-key", "test-key".parse().unwrap());
        assert_eq!(extract_routing_key(Some(&headers)), Some("test-key"));
    }

    #[test]
    fn test_extract_header_value_returns_none_for_missing() {
        let headers = HeaderMap::new();
        assert_eq!(extract_routing_key(Some(&headers)), None);
    }

    #[test]
    fn test_extract_header_value_returns_none_for_empty() {
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-routing-key", "".parse().unwrap());
        assert_eq!(extract_routing_key(Some(&headers)), None);
    }

    #[test]
    fn test_extract_header_value_returns_none_for_none_headers() {
        assert_eq!(extract_routing_key(None), None);
    }

    #[test]
    fn test_extract_target_worker() {
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-target-worker", "2".parse().unwrap());
        assert_eq!(extract_target_worker(Some(&headers)), Some("2"));
    }

    #[test]
    fn test_extract_target_worker_missing() {
        let headers = HeaderMap::new();
        assert_eq!(extract_target_worker(Some(&headers)), None);
    }

    #[test]
    fn test_should_forward_request_header_whitelist() {
        assert!(should_forward_request_header("authorization"));
        assert!(should_forward_request_header("Authorization"));
        assert!(should_forward_request_header("AUTHORIZATION"));
        assert!(should_forward_request_header("x-request-id"));
        assert!(should_forward_request_header("X-Request-Id"));
        assert!(should_forward_request_header("x-correlation-id"));
        assert!(should_forward_request_header("X-Correlation-ID"));
        assert!(should_forward_request_header("traceparent"));
        assert!(should_forward_request_header("Traceparent"));
        assert!(should_forward_request_header("tracestate"));
        assert!(should_forward_request_header("Tracestate"));
        assert!(should_forward_request_header("x-request-id-user"));
        assert!(should_forward_request_header("X-Request-ID-Span"));
        assert!(should_forward_request_header("x-request-id-123"));
        assert!(should_forward_request_header("x-smg-routing-key"));
        assert!(should_forward_request_header("X-SMG-Routing-Key"));
    }

    #[test]
    fn test_should_forward_request_header_blocked() {
        assert!(!should_forward_request_header("content-type"));
        assert!(!should_forward_request_header("Content-Type"));
        assert!(!should_forward_request_header("content-length"));
        assert!(!should_forward_request_header("host"));
        assert!(!should_forward_request_header("Host"));
        assert!(!should_forward_request_header("connection"));
        assert!(!should_forward_request_header("transfer-encoding"));
        assert!(!should_forward_request_header("accept"));
        assert!(!should_forward_request_header("accept-encoding"));
        assert!(!should_forward_request_header("user-agent"));
        assert!(!should_forward_request_header("cookie"));
        assert!(!should_forward_request_header("x-custom-header"));
        assert!(!should_forward_request_header("x-api-key"));
    }

    #[test]
    fn test_extract_forwardable_request_headers_filters_to_allowlist() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer abc".parse().unwrap());
        headers.insert("x-request-id", "req-123".parse().unwrap());
        headers.insert("x-custom-header", "blocked".parse().unwrap());

        let forwarded = extract_forwardable_request_headers(Some(&headers));

        assert_eq!(
            forwarded.get("authorization"),
            Some(&"Bearer abc".to_string())
        );
        assert_eq!(forwarded.get("x-request-id"), Some(&"req-123".to_string()));
        assert!(!forwarded.contains_key("x-custom-header"));
    }

    #[test]
    fn test_extract_forwardable_request_headers_preserves_repeated_values() {
        let mut headers = HeaderMap::new();
        headers.append("tracestate", "vendor1=value1".parse().unwrap());
        headers.append("tracestate", "vendor2=value2".parse().unwrap());

        let forwarded = extract_forwardable_request_headers(Some(&headers));

        assert_eq!(
            forwarded.get("tracestate"),
            Some(&"vendor1=value1, vendor2=value2".to_string())
        );
    }

    #[test]
    fn test_extract_auth_header_falls_back_with_non_auth_headers_present() {
        let mut headers = HeaderMap::new();
        headers.insert("openai-project", "project-123".parse().unwrap());

        let auth = extract_auth_header(Some(&headers), Some(&"worker-secret".to_string()));

        assert_eq!(auth.unwrap(), "Bearer worker-secret");
    }

    #[test]
    fn test_provider_extract_auth_header_prefers_anthropic_key() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "anthropic-key".parse().unwrap());

        let auth = ApiProvider::Anthropic.extract_auth_header(Some(&headers), None);

        assert_eq!(auth.unwrap(), "anthropic-key");
    }

    #[test]
    fn test_memory_header_view_defaults_when_header_missing() {
        let headers = HeaderMap::new();
        let view = MemoryHeaderView::from_http_headers(&headers);

        assert_eq!(view.policy, None);
        assert_eq!(view.subject_id, None);
        assert_eq!(view.embedding_model, None);
        assert_eq!(view.extraction_model, None);
    }

    #[test]
    fn test_memory_header_view_defaults_when_ltm_disabled() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-conversation-memory-config",
            r#"{"long_term_memory":{"enabled":false,"policy":"store","subject_id":"s1","embedding_model_id":"e1","extraction_model_id":"x1"},"short_term_memory":{"enabled":true,"condenser_model_id":"cond-1"}}"#
                .parse()
                .unwrap(),
        );

        let view = MemoryHeaderView::from_http_headers(&headers);

        assert_eq!(view.policy, None);
        assert_eq!(view.subject_id, None);
        assert_eq!(view.embedding_model, None);
        assert_eq!(view.extraction_model, None);
    }

    #[test]
    fn test_memory_header_view_uses_ltm_fields_from_conversation_config() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-conversation-memory-config",
            r#"{"long_term_memory":{"enabled":true,"policy":" store_only ","subject_id":" subject_1 ","embedding_model_id":" text-embedding-3-small ","extraction_model_id":" gpt-4.1-mini "}}"#
                .parse()
                .unwrap(),
        );

        let view = MemoryHeaderView::from_http_headers(&headers);

        assert_eq!(view.policy.as_deref(), Some("store_only"));
        assert_eq!(view.subject_id.as_deref(), Some("subject_1"));
        assert_eq!(
            view.embedding_model.as_deref(),
            Some("text-embedding-3-small")
        );
        assert_eq!(view.extraction_model.as_deref(), Some("gpt-4.1-mini"));
    }

    #[test]
    fn extract_conversation_memory_config_with_valid_json_populates_all_fields() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-conversation-memory-config",
            r#"{"long_term_memory":{"enabled":true,"policy":"recall_only","subject_id":"sub-1","embedding_model_id":"emb-model","extraction_model_id":"ext-model"},"short_term_memory":{"enabled":true,"condenser_model_id":"cond-model"}}"#
                .parse()
                .unwrap(),
        );

        let cfg = extract_conversation_memory_config(Some(&headers))
            .expect("valid JSON must parse to Some");

        assert!(cfg.long_term_memory.enabled);
        assert_eq!(cfg.long_term_memory.policy.as_deref(), Some("recall_only"));
        assert_eq!(cfg.long_term_memory.subject_id.as_deref(), Some("sub-1"));
        assert_eq!(
            cfg.long_term_memory.embedding_model_id.as_deref(),
            Some("emb-model")
        );
        assert_eq!(
            cfg.long_term_memory.extraction_model_id.as_deref(),
            Some("ext-model")
        );
        assert!(cfg.short_term_memory.enabled);
        assert_eq!(
            cfg.short_term_memory.condenser_model_id.as_deref(),
            Some("cond-model")
        );
    }

    #[test]
    fn extract_conversation_memory_config_with_invalid_json_returns_none() {
        let mut headers = HeaderMap::new();
        headers.insert("x-conversation-memory-config", "not-json".parse().unwrap());

        assert!(extract_conversation_memory_config(Some(&headers)).is_none());
    }

    #[test]
    fn extract_conversation_memory_config_with_absent_header_returns_none() {
        assert!(extract_conversation_memory_config(Some(&HeaderMap::new())).is_none());
    }

    #[test]
    fn extract_conversation_memory_config_with_no_headers_returns_none() {
        assert!(extract_conversation_memory_config(None).is_none());
    }
}
