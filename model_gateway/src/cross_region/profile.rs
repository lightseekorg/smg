use serde::{Deserialize, Serialize};

use super::{CrossRegionError, CrossRegionResult, RequestMode};
use crate::config::CrossRegionFailoverMode;

/// Input and output modality hints normalized from DP-API headers.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModalityPolicy {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

/// Failover policy normalized from routing profile headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FailoverPolicy {
    pub failover_mode: CrossRegionFailoverMode,
    pub max_retry: u32,
}

impl FailoverPolicy {
    /// Build a failover policy from mode and retry cap.
    pub fn new(failover_mode: CrossRegionFailoverMode, max_retry: u32) -> Self {
        Self {
            failover_mode,
            max_retry,
        }
    }
}

/// Runtime view of a routing profile as consumed by candidate calculation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingProfileContext {
    pub entry_region: String,
    pub allowed_regions: Vec<String>,
    pub model_id: String,
    #[serde(rename = "failoverPolicy")]
    pub failover_policy: FailoverPolicy,
    pub request_mode: RequestMode,
    pub modality: ModalityPolicy,
}

impl RoutingProfileContext {
    /// Build a Phase 1 routing profile context after enforcing minimal invariants.
    pub fn new(
        entry_region: impl Into<String>,
        allowed_regions: Vec<String>,
        model_id: impl Into<String>,
        failover_policy: FailoverPolicy,
        request_mode: RequestMode,
        modality: ModalityPolicy,
    ) -> CrossRegionResult<Self> {
        let context = Self {
            entry_region: entry_region.into(),
            allowed_regions,
            model_id: model_id.into(),
            failover_policy,
            request_mode,
            modality,
        };
        context.validate()?;
        Ok(context)
    }

    /// Validate the normalized profile context without parsing raw headers.
    pub fn validate(&self) -> CrossRegionResult<()> {
        if self.entry_region.trim().is_empty() {
            return Err(CrossRegionError::InvalidProfile {
                reason: "entry_region must not be empty".to_string(),
            });
        }
        if self.allowed_regions.is_empty()
            || self
                .allowed_regions
                .iter()
                .any(|region| region.trim().is_empty())
        {
            return Err(CrossRegionError::InvalidProfile {
                reason: "allowed_regions must contain at least one non-empty region".to_string(),
            });
        }
        if self.model_id.trim().is_empty() {
            return Err(CrossRegionError::InvalidProfile {
                reason: "model_id must not be empty".to_string(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_profile_context_validates_basic_fields() {
        let context = RoutingProfileContext::new(
            "us-ashburn-1",
            vec!["us-ashburn-1".to_string(), "us-chicago-1".to_string()],
            "cohere.command-r-plus",
            FailoverPolicy::new(CrossRegionFailoverMode::Manual, 1),
            RequestMode::Unresolved,
            ModalityPolicy::default(),
        )
        .expect("profile should be valid");

        assert_eq!(context.model_id, "cohere.command-r-plus");
    }

    #[test]
    fn routing_profile_context_rejects_empty_model() {
        let error = RoutingProfileContext::new(
            "us-ashburn-1",
            vec!["us-ashburn-1".to_string()],
            " ",
            FailoverPolicy::new(CrossRegionFailoverMode::Manual, 1),
            RequestMode::Unresolved,
            ModalityPolicy::default(),
        )
        .expect_err("empty model should fail");

        assert!(error.to_string().contains("model_id"));
    }

    #[test]
    fn failover_policy_serializes_with_contract_field_names() {
        let context = RoutingProfileContext::new(
            "us-ashburn-1",
            vec!["us-ashburn-1".to_string()],
            "cohere.command-r-plus",
            FailoverPolicy::new(CrossRegionFailoverMode::Automatic, 2),
            RequestMode::Unresolved,
            ModalityPolicy::default(),
        )
        .expect("profile should be valid");

        let value = serde_json::to_value(context).expect("serialize profile");

        assert_eq!(value["failoverPolicy"]["failoverMode"], "AUTOMATIC");
        assert_eq!(value["failoverPolicy"]["maxRetry"], 2);
        assert!(value.get("failover_mode").is_none());
        assert!(value.get("max_retry").is_none());
    }
}
