//! Factory for creating load balancing policies

use std::{sync::Arc, time::Duration};

use super::{
    BucketConfig, BucketPolicy, CacheAwareConfig, CacheAwarePolicy, CelPolicyEngine,
    ConsistentHashingPolicy, LoadBalancingPolicy, ManualConfig, ManualPolicy, PowerOfTwoPolicy,
    PrefixHashConfig, PrefixHashPolicy, RandomPolicy, RoundRobinPolicy, RoutingStrategy,
};
use crate::config::PolicyConfig;

/// Factory for creating policy instances
pub struct PolicyFactory;

impl PolicyFactory {
    /// Create a policy from configuration.
    ///
    /// For the `MetricsDriven` policy variant, `metrics_store` must be `Some`.
    /// If it is `None` the factory falls back to `RoundRobinPolicy` and logs a
    /// warning.
    pub fn create_from_config(config: &PolicyConfig) -> Arc<dyn LoadBalancingPolicy> {
        Self::create_from_config_with_store(config, None)
    }

    /// Create a policy from configuration, optionally supplying a `MetricsStore`
    /// for the `MetricsDriven` variant.
    pub fn create_from_config_with_store(
        config: &PolicyConfig,
        metrics_store: Option<Arc<metrics_service::MetricsStore>>,
    ) -> Arc<dyn LoadBalancingPolicy> {
        match config {
            PolicyConfig::Random => Arc::new(RandomPolicy::new()),
            PolicyConfig::RoundRobin => Arc::new(RoundRobinPolicy::new()),
            PolicyConfig::PowerOfTwo { .. } => {
                if let Some(store) = metrics_store {
                    Arc::new(PowerOfTwoPolicy::with_metrics_store(store))
                } else {
                    tracing::warn!(
                        "PowerOfTwo policy created without a MetricsStore; \
                         load metrics will not be available and the policy \
                         will fall back to request-count estimates"
                    );
                    Arc::new(PowerOfTwoPolicy::new())
                }
            }
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
                block_size,
            } => {
                let config = CacheAwareConfig {
                    cache_threshold: *cache_threshold,
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    eviction_interval_secs: *eviction_interval_secs,
                    max_tree_size: *max_tree_size,
                    block_size: *block_size,
                };
                Arc::new(CacheAwarePolicy::with_config(config))
            }
            PolicyConfig::Bucket {
                balance_abs_threshold,
                balance_rel_threshold,
                bucket_adjust_interval_secs,
            } => {
                let config = BucketConfig {
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    bucket_adjust_interval_secs: *bucket_adjust_interval_secs,
                };
                Arc::new(BucketPolicy::with_config(config))
            }
            PolicyConfig::Manual {
                eviction_interval_secs,
                max_idle_secs,
                assignment_mode,
            } => {
                let config = ManualConfig {
                    eviction_interval_secs: *eviction_interval_secs,
                    max_idle_secs: *max_idle_secs,
                    assignment_mode: *assignment_mode,
                };
                Arc::new(ManualPolicy::with_config(config))
            }
            PolicyConfig::ConsistentHashing => Arc::new(ConsistentHashingPolicy::new()),
            PolicyConfig::PrefixHash {
                prefix_token_count,
                load_factor,
            } => {
                let config = PrefixHashConfig {
                    prefix_token_count: *prefix_token_count,
                    load_factor: *load_factor,
                };
                Arc::new(PrefixHashPolicy::new(config))
            }
            PolicyConfig::MetricsDriven {
                strategy,
                fresh_threshold_secs,
                stale_threshold_secs,
            } => {
                if let Some(store) = metrics_store {
                    let routing_strategy = Self::parse_routing_strategy(strategy);
                    Arc::new(CelPolicyEngine::with_strategy(
                        store,
                        Duration::from_secs(*fresh_threshold_secs),
                        Duration::from_secs(*stale_threshold_secs),
                        routing_strategy,
                    ))
                } else {
                    tracing::warn!(
                        "MetricsDriven policy requested but no MetricsStore provided; \
                         falling back to round-robin"
                    );
                    Arc::new(RoundRobinPolicy::new())
                }
            }
        }
    }

    /// Parse a strategy string into a `RoutingStrategy`.
    ///
    /// Named strategies (`"min_kv_cache_tokens"`, `"min_in_flight"`) are
    /// recognized case-insensitively; anything else is treated as a CEL
    /// expression to compile.
    fn parse_routing_strategy(strategy: &str) -> RoutingStrategy {
        match strategy.to_lowercase().as_str() {
            "min_kv_cache_tokens" | "minkvcachetokens" => RoutingStrategy::MinKvCacheTokens,
            "min_in_flight" | "mininflight" => RoutingStrategy::MinInFlight,
            _ => {
                // Treat as CEL expression — compile once
                RoutingStrategy::custom(strategy).unwrap_or_else(|err| {
                    tracing::warn!(
                        "Failed to compile CEL strategy '{}': {}; falling back to MinKvCacheTokens",
                        strategy,
                        err
                    );
                    RoutingStrategy::MinKvCacheTokens
                })
            }
        }
    }

    /// Create a policy by name (for dynamic loading)
    pub fn create_by_name(name: &str) -> Option<Arc<dyn LoadBalancingPolicy>> {
        match name.to_lowercase().as_str() {
            "random" => Some(Arc::new(RandomPolicy::new())),
            "round_robin" | "roundrobin" => Some(Arc::new(RoundRobinPolicy::new())),
            "power_of_two" | "poweroftwo" => Some(Arc::new(PowerOfTwoPolicy::new())),
            "cache_aware" | "cacheaware" => Some(Arc::new(CacheAwarePolicy::new())),
            "bucket" => Some(Arc::new(BucketPolicy::new())),
            "manual" => Some(Arc::new(ManualPolicy::new())),
            "consistent_hashing" | "consistenthashing" => {
                Some(Arc::new(ConsistentHashingPolicy::new()))
            }
            "prefix_hash" | "prefixhash" => Some(Arc::new(PrefixHashPolicy::with_defaults())),
            // MetricsDriven requires a MetricsStore — callers that know the
            // store should use create_from_config_with_store instead.
            "metrics_driven" | "metricsdriven" => {
                tracing::warn!(
                    "create_by_name(\"metrics_driven\") called without a MetricsStore; \
                     use create_from_config_with_store instead. Falling back to round-robin."
                );
                Some(Arc::new(RoundRobinPolicy::new()))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_from_config() {
        let policy = PolicyFactory::create_from_config(&PolicyConfig::Random);
        assert_eq!(policy.name(), "random");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::RoundRobin);
        assert_eq!(policy.name(), "round_robin");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        });
        assert_eq!(policy.name(), "power_of_two");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::CacheAware {
            cache_threshold: 0.7,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 30,
            max_tree_size: 1000,
            block_size: 16,
        });
        assert_eq!(policy.name(), "cache_aware");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::Bucket {
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            bucket_adjust_interval_secs: 5,
        });
        assert_eq!(policy.name(), "bucket");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::Manual {
            eviction_interval_secs: 60,
            max_idle_secs: 4 * 3600,
            assignment_mode: Default::default(),
        });
        assert_eq!(policy.name(), "manual");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::ConsistentHashing);
        assert_eq!(policy.name(), "consistent_hashing");
    }

    #[tokio::test]
    async fn test_create_by_name() {
        assert!(PolicyFactory::create_by_name("random").is_some());
        assert!(PolicyFactory::create_by_name("RANDOM").is_some());
        assert!(PolicyFactory::create_by_name("round_robin").is_some());
        assert!(PolicyFactory::create_by_name("RoundRobin").is_some());
        assert!(PolicyFactory::create_by_name("power_of_two").is_some());
        assert!(PolicyFactory::create_by_name("PowerOfTwo").is_some());
        assert!(PolicyFactory::create_by_name("cache_aware").is_some());
        assert!(PolicyFactory::create_by_name("CacheAware").is_some());
        assert!(PolicyFactory::create_by_name("bucket").is_some());
        assert!(PolicyFactory::create_by_name("Bucket").is_some());
        assert!(PolicyFactory::create_by_name("manual").is_some());
        assert!(PolicyFactory::create_by_name("Manual").is_some());
        assert!(PolicyFactory::create_by_name("consistent_hashing").is_some());
        assert!(PolicyFactory::create_by_name("ConsistentHashing").is_some());
        assert!(PolicyFactory::create_by_name("unknown").is_none());
    }

    #[test]
    fn test_parse_routing_strategy() {
        assert!(matches!(
            PolicyFactory::parse_routing_strategy("min_kv_cache_tokens"),
            RoutingStrategy::MinKvCacheTokens
        ));
        assert!(matches!(
            PolicyFactory::parse_routing_strategy("MinKvCacheTokens"),
            RoutingStrategy::MinKvCacheTokens
        ));
        assert!(matches!(
            PolicyFactory::parse_routing_strategy("min_in_flight"),
            RoutingStrategy::MinInFlight
        ));
        // Valid CEL compiles fine
        assert!(matches!(
            PolicyFactory::parse_routing_strategy("in_flight_requests * 2"),
            RoutingStrategy::Custom { .. }
        ));
    }
}
