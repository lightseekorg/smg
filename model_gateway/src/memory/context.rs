use axum::http::HeaderMap;

pub use super::headers::MemoryHeaderView;
use crate::config::MemoryRuntimeConfig;

/// Normalized per-request memory execution context derived from HTTP headers
/// and runtime configuration flags.
///
/// Each request can carry memory-related headers (policy, subject, models).
/// This struct captures the caller's *intent* (`*_requested` flags) and the
/// system's *readiness* (`*_active` flags).  A feature is active only when
/// both the caller requests it **and** the runtime has it enabled.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryExecutionContext {
    /// The caller asked to persist conversation memories (via policy or
    /// explicit `x-smg-memory-ltm-store-enabled` header).
    pub store_ltm_requested: bool,
    /// Store is both requested **and** the runtime has `ltm_enabled` +
    /// `ltm_store_enabled` set.  Hooks should check this flag, not
    /// `store_ltm_requested`, before writing.
    pub store_ltm_active: bool,
    /// The caller asked to recall previously stored memories (via
    /// `store_and_recall` policy).
    pub recall_requested: bool,
    /// Recall is both requested **and** the runtime has `ltm_enabled` set.
    pub recall_active: bool,
    /// Opaque identifier for the memory subject (e.g. end-user or session).
    pub subject_id: Option<String>,
    /// Recall strategy hint (e.g. "semantic", "keyword").
    pub recall_method: Option<String>,
    /// Model identifier to use for embedding generation during store/recall.
    pub embedding_model: Option<String>,
    /// Model identifier to use for memory extraction / summarization.
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
        let store_ltm_requested =
            policy.allows_ltm_store() || is_enabled(headers.ltm_store_enabled.as_deref());
        let recall_requested = policy.allows_recall();

        Self {
            store_ltm_active: store_ltm_requested
                && runtime.ltm_enabled
                && runtime.ltm_store_enabled,
            store_ltm_requested,
            recall_active: recall_requested && runtime.ltm_enabled,
            recall_requested,
            subject_id: headers.subject_id.clone(),
            recall_method: headers.recall_method.clone(),
            embedding_model: headers.embedding_model.clone(),
            extraction_model: headers.extraction_model.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Policy {
    StoreOnly,
    StoreAndRecall,
    #[default]
    None,
}

impl Policy {
    fn from_value(value: Option<&str>) -> Self {
        let Some(value) = value.map(normalize) else {
            return Self::None;
        };

        match value {
            v if v.eq_ignore_ascii_case("store_only") => Self::StoreOnly,
            v if v.eq_ignore_ascii_case("store_and_recall") => Self::StoreAndRecall,
            _ => Self::None,
        }
    }

    fn allows_ltm_store(self) -> bool {
        matches!(self, Self::StoreOnly | Self::StoreAndRecall)
    }

    fn allows_recall(self) -> bool {
        matches!(self, Self::StoreAndRecall)
    }
}

/// Parse common truthy header values after whitespace normalization.
fn is_enabled(value: Option<&str>) -> bool {
    let Some(value) = value.map(normalize) else {
        return false;
    };

    value == "1"
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("yes")
        || value.eq_ignore_ascii_case("on")
        || value.eq_ignore_ascii_case("enabled")
}

/// Trim header input before policy/flag parsing.
fn normalize(value: &str) -> &str {
    value.trim()
}

#[cfg(test)]
mod tests {
    use super::{MemoryExecutionContext, MemoryHeaderView};
    use crate::config::MemoryRuntimeConfig;

    #[test]
    fn header_policy_store_and_recall_enables_store_request() {
        let headers = MemoryHeaderView {
            policy: Some("store_and_recall".to_string()),
            ..Default::default()
        };

        let ctx = MemoryExecutionContext::from_headers(&headers, &MemoryRuntimeConfig::default());

        assert!(ctx.store_ltm_requested);
        assert!(ctx.recall_requested);
    }

    #[test]
    fn missing_headers_produce_inactive_context() {
        let ctx = MemoryExecutionContext::from_headers(
            &MemoryHeaderView::default(),
            &MemoryRuntimeConfig::default(),
        );

        assert!(!ctx.store_ltm_requested);
        assert!(!ctx.store_ltm_active);
        assert!(!ctx.recall_requested);
        assert!(!ctx.recall_active);
    }

    #[test]
    fn store_headers_become_active_when_runtime_is_ready() {
        let headers = MemoryHeaderView {
            ltm_store_enabled: Some("enabled".to_string()),
            subject_id: Some("subject-1".to_string()),
            ..Default::default()
        };

        let ctx = MemoryExecutionContext::from_headers(
            &headers,
            &MemoryRuntimeConfig {
                ltm_enabled: true,
                ltm_store_enabled: true,
            },
        );

        assert!(ctx.store_ltm_active);
        assert_eq!(ctx.subject_id.as_deref(), Some("subject-1"));
    }

    #[test]
    fn recall_is_active_for_store_and_recall_when_ltm_runtime_enabled() {
        let headers = MemoryHeaderView {
            policy: Some("store_and_recall".to_string()),
            ..Default::default()
        };

        let ctx = MemoryExecutionContext::from_headers(
            &headers,
            &MemoryRuntimeConfig {
                ltm_enabled: true,
                ltm_store_enabled: false,
            },
        );

        assert!(ctx.recall_requested);
        assert!(ctx.recall_active);
    }

    #[test]
    fn store_only_policy_does_not_request_recall() {
        let headers = MemoryHeaderView {
            policy: Some("store_only".to_string()),
            ..Default::default()
        };

        let ctx = MemoryExecutionContext::from_headers(
            &headers,
            &MemoryRuntimeConfig {
                ltm_enabled: true,
                ltm_store_enabled: true,
            },
        );

        assert!(!ctx.recall_requested);
        assert!(!ctx.recall_active);
    }
}
