use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MultiTenantRateLimitConfig {
    pub enabled: bool,
    pub default_tokens_per_minute: u32,
    pub default_requests_per_minute: u32,
    pub tenants: HashMap<String, TenantTokenPolicy>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TenantTokenPolicy {
    pub tokens_per_minute: u32,
    pub requests_per_minute: u32,
}

impl MultiTenantRateLimitConfig {
    #[must_use]
    pub fn policy_for(&self, tenant_key: &str) -> Option<TenantTokenPolicy> {
        if !self.enabled {
            return None;
        }

        self.tenants.get(tenant_key).cloned().or_else(|| {
            (self.default_tokens_per_minute > 0 || self.default_requests_per_minute > 0).then_some(
                TenantTokenPolicy {
                    tokens_per_minute: self.default_tokens_per_minute,
                    requests_per_minute: self.default_requests_per_minute,
                },
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{MultiTenantRateLimitConfig, TenantTokenPolicy};

    #[test]
    fn policy_for_returns_none_when_disabled() {
        let config = MultiTenantRateLimitConfig {
            enabled: false,
            default_tokens_per_minute: 100,
            default_requests_per_minute: 10,
            tenants: HashMap::from([(
                "tenant-a".to_string(),
                TenantTokenPolicy {
                    tokens_per_minute: 50,
                    requests_per_minute: 5,
                },
            )]),
        };

        assert!(config.policy_for("tenant-a").is_none());
        assert!(config.policy_for("missing").is_none());
    }

    #[test]
    fn policy_for_prefers_tenant_override() {
        let tenant_policy = TenantTokenPolicy {
            tokens_per_minute: 50,
            requests_per_minute: 5,
        };
        let config = MultiTenantRateLimitConfig {
            enabled: true,
            default_tokens_per_minute: 100,
            default_requests_per_minute: 10,
            tenants: HashMap::from([("tenant-a".to_string(), tenant_policy.clone())]),
        };

        let policy = config.policy_for("tenant-a").expect("tenant policy");

        assert_eq!(policy.tokens_per_minute, tenant_policy.tokens_per_minute);
        assert_eq!(
            policy.requests_per_minute,
            tenant_policy.requests_per_minute
        );
    }

    #[test]
    fn policy_for_uses_default_when_tenant_missing_and_defaults_enabled() {
        let config = MultiTenantRateLimitConfig {
            enabled: true,
            default_tokens_per_minute: 100,
            default_requests_per_minute: 10,
            tenants: HashMap::new(),
        };

        let policy = config.policy_for("missing").expect("default policy");

        assert_eq!(policy.tokens_per_minute, 100);
        assert_eq!(policy.requests_per_minute, 10);
    }

    #[test]
    fn policy_for_returns_none_when_no_matching_tenant_or_defaults() {
        let config = MultiTenantRateLimitConfig {
            enabled: true,
            default_tokens_per_minute: 0,
            default_requests_per_minute: 0,
            tenants: HashMap::new(),
        };

        assert!(config.policy_for("missing").is_none());
    }
}
