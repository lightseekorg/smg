use axum::http::HeaderMap;

use crate::{config::MemoryRuntimeConfig, routers::common::header_utils::MemoryHeaderView};

/// Normalized per-request memory execution context derived from request headers
/// and runtime configuration flags.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryExecutionContext {
    /// Caller requested LTM store via policy.
    pub store_ltm_requested: bool,
    /// Store is effectively enabled after runtime gates are applied.
    pub store_ltm_active: bool,
    /// Caller requested recall via policy.
    pub recall_requested: bool,
    /// Recall is effectively enabled after runtime gates are applied.
    pub recall_active: bool,
    /// Subject key used by downstream memory systems.
    pub subject_id: Option<String>,
    /// Optional embedding model override.
    pub embedding_model: Option<String>,
    /// Optional extraction model override.
    pub extraction_model: Option<String>,
}

impl MemoryExecutionContext {
    /// Builds memory context by reading HTTP headers and applying runtime gates.
    pub fn from_http_headers(headers: &HeaderMap, runtime: &MemoryRuntimeConfig) -> Self {
        let header_view = MemoryHeaderView::from_http_headers(headers);
        Self::from_headers(&header_view, runtime)
    }

    /// Builds memory context from a normalized header view and runtime gates.
    pub fn from_headers(headers: &MemoryHeaderView, runtime: &MemoryRuntimeConfig) -> Self {
        let policy = Policy::from_value(headers.policy.as_deref());
        // `none` is an explicit per-request opt-out of LTM behavior.
        let store_ltm_requested = if policy.disables_ltm() {
            false
        } else {
            policy.allows_ltm_store()
        };
        let recall_requested = if policy.disables_ltm() {
            false
        } else {
            policy.allows_recall()
        };

        Self {
            store_ltm_active: store_ltm_requested && runtime.enabled,
            store_ltm_requested,
            recall_active: recall_requested && runtime.enabled,
            recall_requested,
            subject_id: headers.subject_id.clone(),
            embedding_model: headers.embedding_model.clone(),
            extraction_model: headers.extraction_model.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Policy {
    /// Store memory only (no recall in this request).
    StoreOnly,
    /// Store new memory and enable recall.
    StoreAndRecall,
    /// Recall only; do not create new memory.
    RecallOnly,
    /// Explicitly disable LTM behavior for this conversation.
    None,
    /// Header missing or unknown value; treated as "no explicit request".
    #[default]
    Unspecified,
}

impl Policy {
    fn disables_ltm(self) -> bool {
        matches!(self, Self::None)
    }
}

impl Policy {
    /// Parse policy header values; unknown values are treated as unspecified.
    fn from_value(value: Option<&str>) -> Self {
        let Some(value) = value.map(normalize) else {
            return Self::Unspecified;
        };

        match value {
            v if v.eq_ignore_ascii_case("store_only") => Self::StoreOnly,
            v if v.eq_ignore_ascii_case("store_and_recall") => Self::StoreAndRecall,
            v if v.eq_ignore_ascii_case("recall_only") => Self::RecallOnly,
            v if v.eq_ignore_ascii_case("none") => Self::None,
            _ => Self::Unspecified,
        }
    }

    fn allows_ltm_store(self) -> bool {
        // Unspecified/None both resolve to no store request.
        matches!(self, Self::StoreOnly | Self::StoreAndRecall)
    }

    fn allows_recall(self) -> bool {
        // Unspecified/None both resolve to no recall request.
        matches!(self, Self::StoreAndRecall | Self::RecallOnly)
    }
}

/// Trim values before policy/boolean matching.
fn normalize(value: &str) -> &str {
    value.trim()
}

#[cfg(test)]
mod tests {
    use axum::http::{header::HeaderName, HeaderValue};

    use super::*;

    fn runtime(enabled: bool) -> MemoryRuntimeConfig {
        MemoryRuntimeConfig { enabled }
    }

    #[test]
    fn store_and_recall_requested_but_not_active_when_runtime_disabled() {
        let headers = MemoryHeaderView {
            policy: Some("store_and_recall".to_string()),
            ..MemoryHeaderView::default()
        };

        let ctx = MemoryExecutionContext::from_headers(&headers, &runtime(false));

        assert!(ctx.store_ltm_requested);
        assert!(!ctx.store_ltm_active);
        assert!(ctx.recall_requested);
        assert!(!ctx.recall_active);
    }

    #[test]
    fn from_http_headers_trims_and_parses_case_insensitively() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-smg-ltm-memory-policy"),
            HeaderValue::from_static("  StOrE_OnLy  "),
        );
        headers.insert(
            HeaderName::from_static("x-smg-ltm-memory-subject-id"),
            HeaderValue::from_static("  subject_abc  "),
        );
        headers.insert(
            HeaderName::from_static("x-smg-ltm-memory-embedding-model"),
            HeaderValue::from_static("  text-embedding-3-small  "),
        );
        headers.insert(
            HeaderName::from_static("x-smg-ltm-memory-extraction-model"),
            HeaderValue::from_static("  gpt-4.1-mini  "),
        );

        let ctx = MemoryExecutionContext::from_http_headers(&headers, &runtime(true));

        assert!(ctx.store_ltm_requested);
        assert!(ctx.store_ltm_active);
        assert!(!ctx.recall_requested);
        assert!(!ctx.recall_active);
        assert_eq!(ctx.subject_id.as_deref(), Some("subject_abc"));
        assert_eq!(
            ctx.embedding_model.as_deref(),
            Some("text-embedding-3-small")
        );
        assert_eq!(ctx.extraction_model.as_deref(), Some("gpt-4.1-mini"));
    }
}
