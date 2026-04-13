use std::sync::{Arc, OnceLock};

use dashmap::DashMap;
use parking_lot::RwLock;
use serde_json;
use smg_mesh::OptionalMeshSyncManager;
use tracing::{debug, info, warn};

/// Policy Registry for managing model-to-policy mappings
///
/// This registry manages the dynamic assignment of load balancing policies to models.
/// When the first worker of a new model is added, it determines the policy for that model.
/// All subsequent workers of the same model use the established policy.
/// When the last worker of a model is removed, the policy mapping is cleaned up.
use super::{BucketPolicy, CacheAwarePolicy, DPRankLoadPolicy, LoadBalancingPolicy, PolicyFactory};
use crate::{
    config::types::PolicyConfig,
    worker::{KvEventMonitor, Worker},
};

/// Registry for managing model-to-policy mappings
#[derive(Clone)]
pub struct PolicyRegistry {
    /// Model ID -> Policy instance mapping (lock-free reads via DashMap)
    model_policies: Arc<DashMap<String, Arc<dyn LoadBalancingPolicy>>>,

    /// Model ID -> Worker count for cleanup tracking (lock-free reads via DashMap)
    model_worker_counts: Arc<DashMap<String, usize>>,

    /// Default policy instance (cached, immutable after creation)
    default_policy: Arc<dyn LoadBalancingPolicy>,

    /// Prefill policy for PD mode (set once at startup, lock-free reads via OnceLock)
    prefill_policy: Arc<OnceLock<Arc<dyn LoadBalancingPolicy>>>,

    /// Decode policy for PD mode (set once at startup, lock-free reads via OnceLock)
    decode_policy: Arc<OnceLock<Arc<dyn LoadBalancingPolicy>>>,

    /// Optional mesh sync manager for state synchronization
    /// When None, the registry works independently without mesh synchronization
    /// Uses RwLock for thread-safe access when setting mesh_sync after initialization
    mesh_sync: Arc<RwLock<OptionalMeshSyncManager>>,

    /// Optional KV event monitor for event-driven cache-aware routing.
    /// When set, new CacheAwarePolicy instances are injected with this monitor.
    kv_event_monitor: Arc<RwLock<Option<Arc<KvEventMonitor>>>>,

    // DP-rank policy: Supports the selection of dp-rank outside the engine.
    dp_rank_policy: Arc<OnceLock<Arc<dyn DPRankLoadPolicy>>>,
}

impl PolicyRegistry {
    /// Create a new PolicyRegistry with a default policy
    pub fn new(default_policy_config: PolicyConfig) -> Self {
        let default_policy = Self::create_policy_from_config(&default_policy_config);

        Self {
            model_policies: Arc::new(DashMap::new()),
            model_worker_counts: Arc::new(DashMap::new()),
            default_policy,
            prefill_policy: Arc::new(OnceLock::new()),
            decode_policy: Arc::new(OnceLock::new()),
            mesh_sync: Arc::new(RwLock::new(None)),
            kv_event_monitor: Arc::new(RwLock::new(None)),
            dp_rank_policy: Arc::new(OnceLock::new()),
        }
    }

    /// Set mesh sync manager (thread-safe, can be called after initialization)
    pub fn set_mesh_sync(&self, mesh_sync: OptionalMeshSyncManager) {
        {
            let mut guard = self.mesh_sync.write();
            guard.clone_from(&mesh_sync);
        }

        Self::maybe_inject_mesh_sync(&self.default_policy, mesh_sync.as_ref());
        if let Some(policy) = self.prefill_policy.get() {
            if !Arc::ptr_eq(policy, &self.default_policy) {
                Self::maybe_inject_mesh_sync(policy, mesh_sync.as_ref());
            }
        }
        if let Some(policy) = self.decode_policy.get() {
            if !Arc::ptr_eq(policy, &self.default_policy) {
                Self::maybe_inject_mesh_sync(policy, mesh_sync.as_ref());
            }
        }
        for entry in self.model_policies.iter() {
            if !Arc::ptr_eq(entry.value(), &self.default_policy) {
                Self::maybe_inject_mesh_sync(entry.value(), mesh_sync.as_ref());
            }
        }
    }

    /// Set KV event monitor (thread-safe, can be called after initialization).
    /// Propagates to all existing cache-aware policies (including default, prefill, decode).
    pub fn set_kv_event_monitor(&self, monitor: Option<Arc<KvEventMonitor>>) {
        {
            let mut guard = self.kv_event_monitor.write();
            guard.clone_from(&monitor);
        }

        // Propagate to existing cache-aware policies so they don't miss the monitor.
        // This covers the default_policy (created before the monitor was available)
        // and any model/PD policies that were already set up.
        Self::maybe_inject_monitor(&self.default_policy, monitor.as_ref());
        if let Some(p) = self.prefill_policy.get() {
            Self::maybe_inject_monitor(p, monitor.as_ref());
        }
        if let Some(p) = self.decode_policy.get() {
            Self::maybe_inject_monitor(p, monitor.as_ref());
        }
        for entry in self.model_policies.iter() {
            Self::maybe_inject_monitor(entry.value(), monitor.as_ref());
        }
    }

    /// Inject KV event monitor into a policy if it's cache-aware.
    fn maybe_inject_monitor(
        policy: &Arc<dyn LoadBalancingPolicy>,
        monitor: Option<&Arc<KvEventMonitor>>,
    ) {
        if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
            cache_aware.set_kv_event_monitor(monitor.cloned());
        }
    }

    fn maybe_inject_mesh_sync(
        policy: &Arc<dyn LoadBalancingPolicy>,
        mesh_sync: Option<&Arc<smg_mesh::MeshSyncManager>>,
    ) {
        if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
            cache_aware.set_mesh_sync(mesh_sync.cloned());
        }
    }

    /// Called when a worker is added
    /// Returns the policy that should be used for this worker's model
    pub fn on_worker_added(
        &self,
        model_id: &str,
        policy_hint: Option<&str>,
    ) -> Arc<dyn LoadBalancingPolicy> {
        // Increment worker count using DashMap entry API
        let count = self
            .model_worker_counts
            .entry(model_id.to_string())
            .and_modify(|c| *c += 1)
            .or_insert(1);
        debug!("Worker added for model {}, count: {}", model_id, *count);
        drop(count); // Release the entry lock

        // Check if model already has a policy (lock-free read via DashMap)
        if let Some(existing_policy) = self.model_policies.get(model_id) {
            debug!(
                "Model {} already has policy: {}",
                model_id,
                existing_policy.name()
            );
            return Arc::clone(&existing_policy);
        }

        // New model - determine policy
        let policy = self.determine_policy_for_model(model_id, policy_hint);

        info!(
            "Assigning policy {} to new model {}",
            policy.name(),
            model_id
        );

        // Store policy for this model (DashMap handles concurrent inserts)
        self.model_policies
            .insert(model_id.to_string(), Arc::clone(&policy));

        // Sync to mesh if enabled (no-op if mesh is not enabled)
        {
            let guard = self.mesh_sync.read();
            if let Some(ref mesh_sync) = *guard {
                // Serialize policy config (simplified - just store policy name for now)
                let config = serde_json::to_vec(&policy.name()).unwrap_or_default();
                mesh_sync.sync_policy_state(
                    model_id.to_string(),
                    policy.name().to_string(),
                    config,
                );
            }
        }

        policy
    }

    /// Called when a worker is removed
    pub fn on_worker_removed(&self, model_id: &str) {
        // Decrement worker count and check if cleanup needed
        let should_cleanup = if let Some(mut count_ref) = self.model_worker_counts.get_mut(model_id)
        {
            *count_ref = count_ref.saturating_sub(1);
            debug!(
                "Worker removed for model {}, count: {}",
                model_id, *count_ref
            );
            if *count_ref == 0 {
                drop(count_ref); // Release before remove
                self.model_worker_counts.remove(model_id);
                true
            } else {
                false
            }
        } else {
            warn!(
                "Attempted to remove worker for model {} with no registered workers",
                model_id
            );
            false
        };

        // Clean up policy if this was the last worker
        if should_cleanup {
            if let Some((_, policy)) = self.model_policies.remove(model_id) {
                info!(
                    "Removed policy {} for model {} (last worker removed)",
                    policy.name(),
                    model_id
                );
            }

            // Sync removal to mesh if enabled (no-op if mesh is not enabled)
            {
                let guard = self.mesh_sync.read();
                if let Some(ref mesh_sync) = *guard {
                    mesh_sync.remove_policy_state(model_id);
                }
            }
        }
    }

    /// Get the policy for a model (lock-free via DashMap)
    pub fn get_policy(&self, model_id: &str) -> Option<Arc<dyn LoadBalancingPolicy>> {
        self.model_policies.get(model_id).map(|r| Arc::clone(&r))
    }

    /// Get the default policy
    pub fn get_default_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        Arc::clone(&self.default_policy)
    }

    /// Get policy for a model, or default if not found
    pub fn get_policy_or_default(&self, model_id: &str) -> Arc<dyn LoadBalancingPolicy> {
        self.get_policy(model_id)
            .unwrap_or_else(|| self.get_default_policy())
    }

    /// Determine policy for a new model
    fn determine_policy_for_model(
        &self,
        model_id: &str,
        policy_hint: Option<&str>,
    ) -> Arc<dyn LoadBalancingPolicy> {
        // 1. Check policy hint from worker
        if let Some(policy_type) = policy_hint {
            debug!("Using policy hint '{}' for model {}", policy_type, model_id);
            return self.create_policy_from_type(policy_type);
        }

        // 2. Use default policy
        debug!("Using default policy for model {}", model_id);
        Arc::clone(&self.default_policy)
    }

    /// Create a policy from a type string (delegates to PolicyFactory)
    fn create_policy_from_type(&self, policy_type: &str) -> Arc<dyn LoadBalancingPolicy> {
        if policy_type == "cache_aware" {
            let cache_aware = CacheAwarePolicy::new();
            {
                let guard = self.mesh_sync.read();
                cache_aware.set_mesh_sync(guard.clone());
            }
            {
                let guard = self.kv_event_monitor.read();
                if let Some(ref monitor) = *guard {
                    cache_aware.set_kv_event_monitor(Some(Arc::clone(monitor)));
                }
            }
            Arc::new(cache_aware)
        } else {
            PolicyFactory::create_by_name(policy_type).unwrap_or_else(|| {
                warn!("Unknown policy type '{}', using default", policy_type);
                Arc::clone(&self.default_policy)
            })
        }
    }

    /// Create a policy from a PolicyConfig (delegates to PolicyFactory)
    fn create_policy_from_config(config: &PolicyConfig) -> Arc<dyn LoadBalancingPolicy> {
        PolicyFactory::create_from_config(config)
    }

    /// Get current model->policy mappings (for debugging/monitoring)
    pub fn get_all_mappings(&self) -> std::collections::HashMap<String, String> {
        self.model_policies
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().name().to_string()))
            .collect()
    }

    /// Get worker counts per model
    pub fn get_worker_counts(&self) -> std::collections::HashMap<String, usize> {
        self.model_worker_counts
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect()
    }

    /// Clear all policies (useful for testing)
    pub fn clear(&self) {
        self.model_policies.clear();
        self.model_worker_counts.clear();
    }

    /// Set the prefill policy for PD mode (lock-free, set once at startup)
    pub fn set_prefill_policy(&self, policy: Arc<dyn LoadBalancingPolicy>) {
        // OnceLock::set returns Err if already set, which we ignore since
        // the policy should only be set once at startup
        let _ = self.prefill_policy.set(policy);
    }

    pub fn set_dp_rank_policy(&self, policy: Arc<dyn DPRankLoadPolicy>) {
        // OnceLock::set returns Err if already set, which we ignore since
        // the policy should only be set once at startup
        debug!("set dp rank policy");
        let _ = self.dp_rank_policy.set(policy);
    }

    pub fn get_dp_rank_policy(&self) -> Option<Arc<dyn DPRankLoadPolicy>> {
        self.dp_rank_policy.get().map(Arc::clone)
    }

    /// Set the decode policy for PD mode (lock-free, set once at startup)
    pub fn set_decode_policy(&self, policy: Arc<dyn LoadBalancingPolicy>) {
        // OnceLock::set returns Err if already set, which we ignore since
        // the policy should only be set once at startup
        let _ = self.decode_policy.set(policy);
    }

    /// Get the prefill policy for PD mode, or default if not set (lock-free)
    pub fn get_prefill_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        self.prefill_policy
            .get()
            .map(Arc::clone)
            .unwrap_or_else(|| self.get_default_policy())
    }

    /// Get the decode policy for PD mode, or default if not set (lock-free)
    pub fn get_decode_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        self.decode_policy
            .get()
            .map(Arc::clone)
            .unwrap_or_else(|| self.get_default_policy())
    }

    /// Get all PowerOfTwo policies that need load updates (lock-free)
    pub fn get_all_power_of_two_policies(&self) -> Vec<Arc<dyn LoadBalancingPolicy>> {
        let mut power_of_two_policies = Vec::new();

        if self.default_policy.name() == "power_of_two" {
            power_of_two_policies.push(Arc::clone(&self.default_policy));
        }

        // Get prefill and decode policies (lock-free via OnceLock::get)
        let prefill_policy_opt = self.prefill_policy.get();
        let decode_policy_opt = self.decode_policy.get();

        if let Some(policy) = prefill_policy_opt {
            if policy.name() == "power_of_two" && !Arc::ptr_eq(policy, &self.default_policy) {
                power_of_two_policies.push(Arc::clone(policy));
            }
        }

        if let Some(policy) = decode_policy_opt {
            if policy.name() == "power_of_two"
                && !Arc::ptr_eq(policy, &self.default_policy)
                && !prefill_policy_opt.is_some_and(|p| Arc::ptr_eq(p, policy))
            {
                power_of_two_policies.push(Arc::clone(policy));
            }
        }

        for entry in self.model_policies.iter() {
            let policy = entry.value();
            if policy.name() == "power_of_two" {
                let already_added = power_of_two_policies.iter().any(|p| Arc::ptr_eq(p, policy));
                if !already_added {
                    power_of_two_policies.push(Arc::clone(policy));
                }
            }
        }

        power_of_two_policies
    }

    /// Initialize cache-aware policy with workers if applicable
    /// This should be called after workers are registered for a model
    pub fn init_cache_aware_policy(&self, model_id: &str, workers: &[Arc<dyn Worker>]) {
        // Get the policy for this model
        if let Some(policy) = self.get_policy(model_id) {
            if policy.name() == "cache_aware" {
                if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                    debug!(
                        "Initializing cache-aware policy with {} workers for model {}",
                        workers.len(),
                        model_id
                    );
                    cache_aware.init_workers(workers);
                }
            }
        }
    }

    /// Remove a worker from cache-aware policy if applicable
    /// This should be called when a worker is being removed
    pub fn remove_worker_from_cache_aware(&self, model_id: &str, worker_url: &str) {
        // Get the policy for this model
        if let Some(policy) = self.get_policy(model_id) {
            if policy.name() == "cache_aware" {
                if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                    cache_aware.remove_worker_by_url(worker_url);
                    debug!(
                        "Removed worker {} from cache-aware policy for model {}",
                        worker_url, model_id
                    );
                }
            }
        }
    }

    /// Initialize cache-aware policies for PD mode (prefill and decode) - lock-free
    pub fn init_pd_cache_aware_policies(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
    ) {
        // Initialize prefill policy if it's cache-aware (lock-free via OnceLock::get)
        if let Some(prefill_policy) = self.prefill_policy.get() {
            if prefill_policy.name() == "cache_aware" {
                if let Some(cache_aware) =
                    prefill_policy.as_any().downcast_ref::<CacheAwarePolicy>()
                {
                    if !prefill_workers.is_empty() {
                        debug!(
                            "Initializing prefill cache-aware policy with {} workers",
                            prefill_workers.len()
                        );
                        cache_aware.init_workers(prefill_workers);
                    }
                }
            }
        }

        // Initialize decode policy if it's cache-aware (lock-free via OnceLock::get)
        if let Some(decode_policy) = self.decode_policy.get() {
            if decode_policy.name() == "cache_aware" {
                if let Some(cache_aware) = decode_policy.as_any().downcast_ref::<CacheAwarePolicy>()
                {
                    if !decode_workers.is_empty() {
                        debug!(
                            "Initializing decode cache-aware policy with {} workers",
                            decode_workers.len()
                        );
                        cache_aware.init_workers(decode_workers);
                    }
                }
            }
        }
    }

    /// Initialize bucket policies for PD mode - lock-free
    pub fn init_pd_bucket_policies(&self, prefill_workers: &[Arc<dyn Worker>]) {
        // Initialize prefill policy if it's bucket (lock-free via OnceLock::get)
        if let Some(prefill_policy) = self.prefill_policy.get() {
            if prefill_policy.name() == "bucket" {
                if let Some(bucket) = prefill_policy.as_any().downcast_ref::<BucketPolicy>() {
                    if !prefill_workers.is_empty() {
                        debug!(
                            "Initializing prefill bucket policy with {} workers",
                            prefill_workers.len()
                        );
                        bucket.init_prefill_worker_urls(prefill_workers);
                    }
                }
            }
        }
    }

    /// Apply remote tree operation to cache-aware policy for a model
    /// This is called when receiving tree state updates from mesh
    pub fn apply_remote_tree_operation(&self, model_id: &str, operation: &smg_mesh::TreeOperation) {
        let model_policy = self.get_policy(model_id);

        if let Some(ref policy) = model_policy {
            if policy.name() == "cache_aware" {
                if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                    cache_aware.apply_remote_tree_operation(model_id, operation);
                }
            }
        }

        // Skip default if same Arc as model policy (avoid double application)
        if !model_policy
            .as_ref()
            .is_some_and(|p| Arc::ptr_eq(p, &self.default_policy))
            && self.default_policy.name() == "cache_aware"
        {
            if let Some(cache_aware) = self
                .default_policy
                .as_any()
                .downcast_ref::<CacheAwarePolicy>()
            {
                cache_aware.apply_remote_tree_operation(model_id, operation);
            }
        }

        if let Some(prefill_policy) = self.prefill_policy.get() {
            if prefill_policy.name() == "cache_aware" {
                if let Some(cache_aware) =
                    prefill_policy.as_any().downcast_ref::<CacheAwarePolicy>()
                {
                    cache_aware.apply_remote_tree_operation(model_id, operation);
                }
            }
        }

        if let Some(decode_policy) = self.decode_policy.get() {
            if decode_policy.name() == "cache_aware" {
                if let Some(cache_aware) = decode_policy.as_any().downcast_ref::<CacheAwarePolicy>()
                {
                    cache_aware.apply_remote_tree_operation(model_id, operation);
                }
            }
        }
    }

    /// Apply lightweight tenant delta directly to CacheAwarePolicy trees.
    /// No TreeState deserialization — inserts/evictions go straight to the radix tree.
    pub fn apply_tenant_delta(
        &self,
        model_id: &str,
        inserts: &[smg_mesh::TenantInsert],
        evictions: &[smg_mesh::TenantEvict],
    ) {
        let apply_to = |policy: &dyn LoadBalancingPolicy| {
            if policy.name() == "cache_aware" {
                if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                    cache_aware.apply_tenant_delta(model_id, inserts, evictions);
                }
            }
        };

        let model_policy = self.get_policy(model_id);
        if let Some(ref policy) = model_policy {
            apply_to(policy.as_ref());
        }

        // Apply to default if different from model policy
        if !model_policy
            .as_ref()
            .is_some_and(|p| Arc::ptr_eq(p, &self.default_policy))
        {
            apply_to(self.default_policy.as_ref());
        }

        if let Some(prefill_policy) = self.prefill_policy.get() {
            apply_to(prefill_policy.as_ref());
        }

        if let Some(decode_policy) = self.decode_policy.get() {
            apply_to(decode_policy.as_ref());
        }
    }

    pub fn apply_remote_tree_state(&self, model_id: &str, tree_state: &smg_mesh::TreeState) {
        let model_policy = self.get_policy(model_id);

        if let Some(ref policy) = model_policy {
            if policy.name() == "cache_aware" {
                if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                    cache_aware.apply_remote_tree_state(model_id, tree_state);
                }
            }
        }

        // Skip default if same Arc as model policy (avoid double application)
        if !model_policy
            .as_ref()
            .is_some_and(|p| Arc::ptr_eq(p, &self.default_policy))
            && self.default_policy.name() == "cache_aware"
        {
            if let Some(cache_aware) = self
                .default_policy
                .as_any()
                .downcast_ref::<CacheAwarePolicy>()
            {
                cache_aware.apply_remote_tree_state(model_id, tree_state);
            }
        }

        if let Some(prefill_policy) = self.prefill_policy.get() {
            if prefill_policy.name() == "cache_aware" {
                if let Some(cache_aware) =
                    prefill_policy.as_any().downcast_ref::<CacheAwarePolicy>()
                {
                    cache_aware.apply_remote_tree_state(model_id, tree_state);
                }
            }
        }

        if let Some(decode_policy) = self.decode_policy.get() {
            if decode_policy.name() == "cache_aware" {
                if let Some(cache_aware) = decode_policy.as_any().downcast_ref::<CacheAwarePolicy>()
                {
                    cache_aware.apply_remote_tree_state(model_id, tree_state);
                }
            }
        }
    }
}

impl smg_mesh::TreeStateSubscriber for PolicyRegistry {
    fn apply_remote_tree_state(&self, model_id: &str, tree_state: &smg_mesh::TreeState) {
        PolicyRegistry::apply_remote_tree_state(self, model_id, tree_state);
    }

    fn apply_tenant_delta(
        &self,
        model_id: &str,
        inserts: &[smg_mesh::TenantInsert],
        evictions: &[smg_mesh::TenantEvict],
    ) {
        PolicyRegistry::apply_tenant_delta(self, model_id, inserts, evictions);
    }

    fn export_tree_state(&self, model_id: &str) -> Option<smg_mesh::TreeState> {
        // Try model-specific policy first, then default
        let policy = self.get_policy(model_id);
        if let Some(ref p) = policy {
            if let Some(cache_aware) = p.as_any().downcast_ref::<CacheAwarePolicy>() {
                return cache_aware.export_tree_state(model_id);
            }
        }
        if let Some(cache_aware) = self
            .default_policy
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
        {
            return cache_aware.export_tree_state(model_id);
        }
        None
    }

    fn export_tree_snapshot(&self, model_id: &str) -> Option<kv_index::snapshot::TreeSnapshot> {
        // Try model-specific policy first, then default
        let policy = self.get_policy(model_id);
        if let Some(ref p) = policy {
            if let Some(cache_aware) = p.as_any().downcast_ref::<CacheAwarePolicy>() {
                return cache_aware.export_tree_snapshot(model_id);
            }
        }
        if let Some(cache_aware) = self
            .default_policy
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
        {
            return cache_aware.export_tree_snapshot(model_id);
        }
        None
    }
}

impl std::fmt::Debug for PolicyRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyRegistry")
            .field("model_policies", &self.model_policies)
            .field("model_worker_counts", &self.model_worker_counts)
            .field("default_policy", &self.default_policy.name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::worker::HealthCheckConfig;
    use smg_mesh::{MeshSyncManager, StateStores};

    use super::*;
    use crate::{
        policies::SelectWorkerInfo,
        worker::{BasicWorkerBuilder, Worker, WorkerType, UNKNOWN_MODEL_ID},
    };

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_policy_registry_basic() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // First worker of a model sets the policy
        let policy1 = registry.on_worker_added("llama-3", Some("cache_aware"));
        assert_eq!(policy1.name(), "cache_aware");

        // Second worker of same model uses existing policy
        let policy2 = registry.on_worker_added("llama-3", Some("round_robin"));
        assert_eq!(policy2.name(), "cache_aware"); // Ignores hint, uses existing

        // Different model can have different policy
        let policy3 = registry.on_worker_added("gpt-4", Some("random"));
        assert_eq!(policy3.name(), "random");

        // Check mappings
        let mappings = registry.get_all_mappings();
        assert_eq!(mappings.get("llama-3").unwrap(), "cache_aware");
        assert_eq!(mappings.get("gpt-4").unwrap(), "random");

        // Check worker counts
        let counts = registry.get_worker_counts();
        assert_eq!(*counts.get("llama-3").unwrap(), 2);
        assert_eq!(*counts.get("gpt-4").unwrap(), 1);
    }

    #[test]
    fn test_policy_registry_cleanup() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // Add workers
        registry.on_worker_added("llama-3", Some("cache_aware"));
        registry.on_worker_added("llama-3", None);
        assert_eq!(registry.get_worker_counts().get("llama-3"), Some(&2));

        // Remove one worker - policy should remain
        registry.on_worker_removed("llama-3");
        assert!(registry.get_policy("llama-3").is_some());
        assert_eq!(registry.get_worker_counts().get("llama-3"), Some(&1));

        // Remove last worker - policy should be cleaned up
        registry.on_worker_removed("llama-3");
        assert!(registry.get_policy("llama-3").is_none());
        assert_eq!(registry.get_worker_counts().get("llama-3"), None);
    }

    #[test]
    fn test_default_policy() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // No hint, no template - uses default
        let policy = registry.on_worker_added("unknown-model", None);
        assert_eq!(policy.name(), "round_robin");

        // Get default directly
        let default = registry.get_default_policy();
        assert_eq!(default.name(), "round_robin");
    }

    #[test]
    fn test_set_mesh_sync_propagates_to_default_cache_aware_policy() {
        let registry = PolicyRegistry::new(PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 0,
            max_tree_size: 10_000,
            block_size: 16,
        });

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let mesh_sync = Arc::new(MeshSyncManager::new(stores.clone(), "node1".to_string()));
        registry.set_mesh_sync(Some(mesh_sync.clone()));

        let policy = registry.get_default_policy();
        let cache_aware = policy
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
            .expect("default policy should be cache_aware");

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .health_config(no_health_check())
                .build(),
        )];

        cache_aware.init_workers(&workers);
        let selected = cache_aware.select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some("mesh aware"),
                ..Default::default()
            },
        );

        assert_eq!(selected, Some(0));
        // sync_tree_operation buffers tenant deltas and bumps version,
        // but does not populate tree_configs (that requires checkpoint).
        // Verify the mesh hook actually ran by checking for buffered deltas.
        assert!(
            stores.tenant_delta_inserts.get(UNKNOWN_MODEL_ID).is_some(),
            "mesh hook should have buffered tenant delta inserts"
        );
    }

    #[test]
    fn test_remote_tree_state_push_updates_default_cache_aware_policy() {
        use smg_mesh::{TreeInsertOp, TreeKey, TreeOperation, TreeState};

        let registry = Arc::new(PolicyRegistry::new(PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 0,
            max_tree_size: 10_000,
            block_size: 16,
        }));

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let mesh_sync = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));
        mesh_sync.register_tree_state_subscriber(registry.clone());
        registry.set_mesh_sync(Some(mesh_sync.clone()));

        let worker1: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .health_config(no_health_check())
                .build(),
        );
        let worker2: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w2:8000")
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .health_config(no_health_check())
                .build(),
        );
        let workers = vec![worker1, worker2];

        let default_policy = registry.get_default_policy();
        let cache_aware = default_policy
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
            .expect("default policy should be cache_aware");
        cache_aware.init_workers(&workers);

        let mut tree_state = TreeState::new(UNKNOWN_MODEL_ID.to_string());
        tree_state.add_operation(TreeOperation::Insert(TreeInsertOp {
            key: TreeKey::Text("mesh push".to_string()),
            tenant: "http://w2:8000".to_string(),
        }));

        mesh_sync.apply_remote_tree_operation(UNKNOWN_MODEL_ID.to_string(), tree_state, None);

        let selected = cache_aware.select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some("mesh push"),
                ..Default::default()
            },
        );

        assert_eq!(selected, Some(1));
    }
}
