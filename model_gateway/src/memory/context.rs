use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};

pub use super::headers::MemoryHeaderView;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryRuntimeConfig {
    pub ltm_enabled: bool,
    pub ltm_store_enabled: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryExecutionContext {
    pub store_ltm_requested: bool,
    pub store_ltm_active: bool,
    pub subject_id: Option<String>,
    pub recall_method: Option<String>,
    pub embedding_model: Option<String>,
    pub extraction_model: Option<String>,
}

impl MemoryExecutionContext {
    pub fn from_http_headers(headers: &HeaderMap, runtime: &MemoryRuntimeConfig) -> Self {
        let header_view = MemoryHeaderView::from_http_headers(headers);
        Self::from_headers(&header_view, runtime)
    }

    pub fn from_headers(headers: &MemoryHeaderView, runtime: &MemoryRuntimeConfig) -> Self {
        let policy = Policy::from_value(headers.policy.as_deref());
        let store_ltm_requested =
            policy.allows_ltm_store() || is_enabled(headers.ltm_store_enabled.as_deref());

        Self {
            store_ltm_active: store_ltm_requested
                && runtime.ltm_enabled
                && runtime.ltm_store_enabled,
            store_ltm_requested,
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
        match value.map(normalize) {
            Some("store_only") => Self::StoreOnly,
            Some("store_and_recall") => Self::StoreAndRecall,
            Some("none") | Some("no_policy") => Self::None,
            _ => Self::None,
        }
    }

    fn allows_ltm_store(self) -> bool {
        matches!(self, Self::StoreOnly | Self::StoreAndRecall)
    }
}

fn is_enabled(value: Option<&str>) -> bool {
    matches!(value.map(normalize), Some("1" | "true" | "yes" | "enabled"))
}

fn normalize(value: &str) -> &str {
    value.trim()
}

#[cfg(test)]
mod tests {
    use super::{MemoryExecutionContext, MemoryHeaderView, MemoryRuntimeConfig};

    #[test]
    fn header_policy_store_and_recall_enables_store_request() {
        let headers = MemoryHeaderView {
            policy: Some("store_and_recall".to_string()),
            ..Default::default()
        };

        let ctx = MemoryExecutionContext::from_headers(&headers, &MemoryRuntimeConfig::default());

        assert!(ctx.store_ltm_requested);
    }

    #[test]
    fn missing_headers_produce_inactive_context() {
        let ctx = MemoryExecutionContext::from_headers(
            &MemoryHeaderView::default(),
            &MemoryRuntimeConfig::default(),
        );

        assert!(!ctx.store_ltm_active);
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
}
