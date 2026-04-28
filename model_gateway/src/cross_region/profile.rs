use serde::{Deserialize, Serialize};

use super::{CrossRegionError, CrossRegionResult};
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
    pub allowed_regions: Vec<String>,
    #[serde(rename = "modelIds")]
    pub model_ids: Vec<String>,
    #[serde(rename = "failoverPolicy")]
    pub failover_policy: FailoverPolicy,
    pub modality: ModalityPolicy,
}

impl RoutingProfileContext {
    /// Build a Phase 1 routing profile context after enforcing minimal invariants.
    pub fn new(
        allowed_regions: Vec<String>,
        model_ids: Vec<String>,
        failover_policy: FailoverPolicy,
        modality: ModalityPolicy,
    ) -> CrossRegionResult<Self> {
        let context = Self {
            allowed_regions,
            model_ids,
            failover_policy,
            modality,
        };
        context.validate()?;
        Ok(context)
    }

    /// Validate the normalized profile context without parsing raw headers.
    pub fn validate(&self) -> CrossRegionResult<()> {
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
        if self.model_ids.len() != 1
            || self
                .model_ids
                .iter()
                .any(|model_id| model_id.trim().is_empty())
        {
            return Err(CrossRegionError::InvalidProfile {
                reason: "model_ids must contain exactly one non-empty model id for Phase 1"
                    .to_string(),
            });
        }
        Ok(())
    }

    /// Return the single Phase 1 model id after validating the profile invariant.
    pub fn single_model_id(&self) -> CrossRegionResult<&str> {
        match self.model_ids.as_slice() {
            [model_id] if !model_id.trim().is_empty() => Ok(model_id.as_str()),
            _ => Err(CrossRegionError::InvalidProfile {
                reason: "model_ids must contain exactly one non-empty model id for Phase 1"
                    .to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_profile_context_validates_basic_fields() {
        let context = RoutingProfileContext::new(
            vec!["us-ashburn-1".to_string(), "us-chicago-1".to_string()],
            vec!["cohere.command-r-plus".to_string()],
            FailoverPolicy::new(CrossRegionFailoverMode::Manual, 1),
            ModalityPolicy::default(),
        )
        .expect("profile should be valid");

        assert_eq!(context.model_ids, vec!["cohere.command-r-plus".to_string()]);
        assert_eq!(
            context.single_model_id().expect("single model id"),
            "cohere.command-r-plus"
        );
    }

    #[test]
    fn routing_profile_context_rejects_empty_model() {
        let error = RoutingProfileContext::new(
            vec!["us-ashburn-1".to_string()],
            vec![" ".to_string()],
            FailoverPolicy::new(CrossRegionFailoverMode::Manual, 1),
            ModalityPolicy::default(),
        )
        .expect_err("empty model should fail");

        assert!(error.to_string().contains("model_ids"));
    }

    #[test]
    fn routing_profile_context_rejects_multiple_models_for_phase1() {
        let error = RoutingProfileContext::new(
            vec!["us-ashburn-1".to_string()],
            vec![
                "cohere.command-r-plus".to_string(),
                "meta.llama-3".to_string(),
            ],
            FailoverPolicy::new(CrossRegionFailoverMode::Manual, 1),
            ModalityPolicy::default(),
        )
        .expect_err("multiple models should fail");

        assert!(error.to_string().contains("exactly one"));
    }

    #[test]
    fn failover_policy_serializes_with_contract_field_names() {
        let context = RoutingProfileContext::new(
            vec!["us-ashburn-1".to_string()],
            vec!["cohere.command-r-plus".to_string()],
            FailoverPolicy::new(CrossRegionFailoverMode::Automatic, 2),
            ModalityPolicy::default(),
        )
        .expect("profile should be valid");

        let value = serde_json::to_value(context).expect("serialize profile");

        assert_eq!(value["failoverPolicy"]["failoverMode"], "AUTOMATIC");
        assert_eq!(value["failoverPolicy"]["maxRetry"], 2);
        assert_eq!(
            value["modelIds"],
            serde_json::json!(["cohere.command-r-plus"])
        );
        assert!(value.get("failover_mode").is_none());
        assert!(value.get("max_retry").is_none());
    }
}
