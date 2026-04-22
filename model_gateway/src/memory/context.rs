use axum::http::HeaderMap;
use tracing::warn;

use crate::{config::MemoryRuntimeConfig, routers::common::header_utils::MemoryHeaderView};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MemoryExecutionState {
    #[default]
    NotRequested,
    GatedOff,
    Active,
}

impl MemoryExecutionState {
    fn from_requested_and_runtime(requested: bool, runtime_enabled: bool) -> Self {
        match (requested, runtime_enabled) {
            (false, _) => Self::NotRequested,
            (true, false) => Self::GatedOff,
            (true, true) => Self::Active,
        }
    }

    pub fn requested(self) -> bool {
        !matches!(self, Self::NotRequested)
    }

    pub fn active(self) -> bool {
        matches!(self, Self::Active)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MemoryPolicyMode {
    StoreOnly,
    StoreAndRecall,
    RecallOnly,
    ExplicitNone,
    #[default]
    Unspecified,
    Unrecognized,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryExecutionContext {
    pub store_ltm: MemoryExecutionState,
    pub recall: MemoryExecutionState,
    pub policy_mode: MemoryPolicyMode,
    pub subject_id: Option<String>,
    pub embedding_model: Option<String>,
    pub extraction_model: Option<String>,
}

impl MemoryExecutionContext {
    pub fn from_http_headers(headers: &HeaderMap, runtime: &MemoryRuntimeConfig) -> Self {
        let header_view = MemoryHeaderView::from_http_headers(headers);
        Self::from_headers(&header_view, runtime)
    }

    pub fn from_headers(headers: &MemoryHeaderView, runtime: &MemoryRuntimeConfig) -> Self {
        let (policy, policy_mode) = parse_policy(headers.policy.as_deref());
        if matches!(policy_mode, MemoryPolicyMode::Unrecognized) {
            if let Some(raw_policy) = headers.policy.as_deref() {
                warn!(
                    policy = raw_policy,
                    "Unrecognized memory policy value; falling back to no-op policy (none)"
                );
            }
        }
        let store_ltm_requested = policy.allows_ltm_store();
        let recall_requested = policy.allows_recall();

        Self {
            store_ltm: MemoryExecutionState::from_requested_and_runtime(
                store_ltm_requested,
                runtime.enabled,
            ),
            recall: MemoryExecutionState::from_requested_and_runtime(
                recall_requested,
                runtime.enabled,
            ),
            policy_mode,
            subject_id: headers.subject_id.clone(),
            embedding_model: headers.embedding_model.clone(),
            extraction_model: headers.extraction_model.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Policy {
    StoreOnly,
    StoreAndRecall,
    RecallOnly,
    None,
    #[default]
    Unspecified,
}

impl Policy {
    fn from_value(value: Option<&str>) -> Self {
        let Some(value) = value.map(str::trim).filter(|v| !v.is_empty()) else {
            return Self::Unspecified;
        };

        match value {
            "store_only" => Self::StoreOnly,
            "store_and_recall" => Self::StoreAndRecall,
            "recall_only" => Self::RecallOnly,
            "none" => Self::None,
            _ => Self::Unspecified,
        }
    }

    fn allows_ltm_store(self) -> bool {
        matches!(self, Self::StoreOnly | Self::StoreAndRecall)
    }

    fn allows_recall(self) -> bool {
        matches!(self, Self::StoreAndRecall | Self::RecallOnly)
    }
}

fn parse_policy(raw_value: Option<&str>) -> (Policy, MemoryPolicyMode) {
    match raw_value {
        None => (Policy::Unspecified, MemoryPolicyMode::Unspecified),
        Some(value) => {
            let parsed = Policy::from_value(Some(value));
            let mode = match parsed {
                Policy::StoreOnly => MemoryPolicyMode::StoreOnly,
                Policy::StoreAndRecall => MemoryPolicyMode::StoreAndRecall,
                Policy::RecallOnly => MemoryPolicyMode::RecallOnly,
                Policy::None => MemoryPolicyMode::ExplicitNone,
                Policy::Unspecified => MemoryPolicyMode::Unrecognized,
            };
            (parsed, mode)
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderValue;

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

        assert_eq!(ctx.store_ltm, MemoryExecutionState::GatedOff);
        assert_eq!(ctx.recall, MemoryExecutionState::GatedOff);
        assert_eq!(ctx.policy_mode, MemoryPolicyMode::StoreAndRecall);
    }

    #[test]
    fn from_http_headers_reads_conversation_memory_config() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-conversation-memory-config",
            HeaderValue::from_static(
                r#"{"long_term_memory":{"enabled":true,"policy":"store_and_recall","subject_id":"  subject_abc  ","embedding_model_id":"  text-embedding-3-small  ","extraction_model_id":"  gpt-4.1-mini  "}}"#,
            ),
        );

        let ctx = MemoryExecutionContext::from_http_headers(&headers, &runtime(true));

        assert_eq!(ctx.store_ltm, MemoryExecutionState::Active);
        assert_eq!(ctx.recall, MemoryExecutionState::Active);
        assert_eq!(ctx.policy_mode, MemoryPolicyMode::StoreAndRecall);
        assert_eq!(ctx.subject_id.as_deref(), Some("subject_abc"));
        assert_eq!(
            ctx.embedding_model.as_deref(),
            Some("text-embedding-3-small")
        );
        assert_eq!(ctx.extraction_model.as_deref(), Some("gpt-4.1-mini"));
    }
}
