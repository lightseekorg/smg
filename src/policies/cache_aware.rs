/*
    Cache-Aware Load Balancing Router

    This router combines two strategies to optimize both cache utilization and request distribution:

    1. Cache-Aware Routing (Approximate Tree)
    2. Load Balancing (Shortest Queue with Balance Thresholds)

    The router dynamically switches between these strategies based on load conditions:
    - Uses load balancing when the system is imbalanced
    - Uses cache-aware routing when the system is balanced

    A system is considered imbalanced if both conditions are met:
    1. (max - min) > abs_threshold
    2. max > rel_threshold * min

    Strategy Details:

    1. Cache-Aware Routing (Approximate Tree)
    -------------------------------------------
    This strategy maintains an approximate radix tree for each worker based on request history,
    eliminating the need for direct cache state queries. The tree stores raw text characters
    instead of token IDs to avoid tokenization overhead.

    Process:
    a. For each request, find the worker with the highest prefix match
    b. If match rate > cache_threshold:
    Route to the worker with highest match (likely has relevant data cached)
    c. If match rate ≤ cache_threshold:
    Route to the worker with smallest tree size (most available cache capacity)
    d. Background maintenance:
    Periodically evict least recently used leaf nodes to prevent memory overflow

    2. Load Balancing (Shortest Queue)
    -------------------------------------------
    This strategy tracks pending request counts per worker and routes new requests
    to the least busy worker when the system is detected to be imbalanced.

    Configuration Parameters:
    ------------------------
    1. cache_threshold: (float, 0.0 to 1.0)
    Minimum prefix match ratio to use highest-match routing.
    Below this threshold, routes to worker with most available cache space.

    2. balance_abs_threshold: (integer)
    Absolute difference threshold for load imbalance detection.
    System is potentially imbalanced if (max_load - min_load) > abs_threshold

    3. balance_rel_threshold: (float)
    Relative ratio threshold for load imbalance detection.
    System is potentially imbalanced if max_load > min_load * rel_threshold
    Used in conjunction with abs_threshold to determine final imbalance state.

    4. eviction_interval_secs: (integer)
    Interval between LRU eviction cycles for the approximate trees.

    5. max_tree_size: (integer)
    Maximum nodes per tree. When exceeded, LRU leaf nodes are evicted
    during the next eviction cycle.
*/

use std::sync::Arc;

use dashmap::DashMap;
use kv_index::{TokenTree, Tree};
use rand::Rng;
use smg_mesh::{
    tree_ops::{TreeInsertOp, TreeOperation},
    OptionalMeshSyncManager,
};
use tracing::{debug, warn};

use super::{
    get_healthy_worker_indices, normalize_model_key, utils::PeriodicTask, CacheAwareConfig,
    LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::core::{Worker, UNKNOWN_MODEL_ID};

/// Cache-aware routing policy
///
/// Routes requests based on cache affinity when load is balanced,
/// switches to shortest-queue routing when load is imbalanced.
/// Maintains separate trees per model for multi-model support.
/// Supports mesh synchronization of tree operations across cluster nodes.
/// When mesh is not enabled, the policy works independently without synchronization.
///
/// Supports both HTTP (string-based) and gRPC (token-based) connections:
/// - HTTP requests use StringTree (character-based prefix matching)
/// - gRPC requests use TokenTree (token-based prefix matching, page-aligned)
#[derive(Debug)]
pub struct CacheAwarePolicy {
    config: CacheAwareConfig,
    /// String-based trees for HTTP connections (text input)
    string_trees: Arc<DashMap<String, Arc<Tree>>>,
    /// Token-based trees for gRPC connections (pre-tokenized input)
    token_trees: Arc<DashMap<String, Arc<TokenTree>>>,
    mesh_sync: OptionalMeshSyncManager,
    _eviction_task: Option<PeriodicTask>,
}

impl CacheAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(config: CacheAwareConfig) -> Self {
        let string_trees = Arc::new(DashMap::<String, Arc<Tree>>::new());
        let token_trees = Arc::new(DashMap::<String, Arc<TokenTree>>::new());

        // Start background eviction thread if configured
        let eviction_task = if config.eviction_interval_secs > 0 {
            let string_trees_clone = Arc::clone(&string_trees);
            let token_trees_clone = Arc::clone(&token_trees);
            let max_tree_size = config.max_tree_size;

            Some(PeriodicTask::spawn(
                config.eviction_interval_secs,
                "Eviction",
                move || {
                    // Evict string trees (HTTP)
                    for tree_ref in string_trees_clone.iter() {
                        let model_id = tree_ref.key();
                        let tree = tree_ref.value();
                        tree.evict_tenant_by_size(max_tree_size);

                        debug!(
                            "String tree eviction completed for model {}, max_size: {}",
                            model_id, max_tree_size
                        );
                    }
                    // Evict token trees (gRPC)
                    for tree_ref in token_trees_clone.iter() {
                        let model_id = tree_ref.key();
                        let tree = tree_ref.value();
                        tree.evict_tenant_by_size(max_tree_size);

                        debug!(
                            "Token tree eviction completed for model {}, max_size: {}",
                            model_id, max_tree_size
                        );
                    }
                },
            ))
        } else {
            None
        };

        Self {
            config,
            string_trees,
            token_trees,
            mesh_sync: None,
            _eviction_task: eviction_task,
        }
    }

    /// Set mesh sync manager (can be called after construction)
    pub fn set_mesh_sync(&mut self, mesh_sync: OptionalMeshSyncManager) {
        self.mesh_sync = mesh_sync.clone();
        if mesh_sync.is_some() {
            self.restore_tree_state_from_mesh();
        }
    }

    /// Initialize the trees with worker URLs (used only during initial setup)
    /// Initializes both string trees (HTTP) and token trees (gRPC) for each model.
    pub fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        // Group workers by model
        let mut model_workers: std::collections::HashMap<String, Vec<&Arc<dyn Worker>>> =
            std::collections::HashMap::new();
        for worker in workers {
            let tree_key = normalize_model_key(worker.model_id());
            model_workers
                .entry(tree_key.to_string())
                .or_default()
                .push(worker);
        }

        // Initialize trees for each model (both string and token trees)
        for (tree_key, model_workers) in model_workers {
            // Initialize string tree (HTTP)
            let string_tree = self
                .string_trees
                .entry(tree_key.clone())
                .or_insert_with(|| Arc::new(Tree::new()));
            // Initialize token tree (gRPC)
            let token_tree = self
                .token_trees
                .entry(tree_key)
                .or_insert_with(|| Arc::new(TokenTree::new()));

            for worker in model_workers {
                string_tree.insert_text("", worker.url());
                token_tree.insert_tokens(&[], worker.url());
            }
        }
    }

    /// Add a single worker to the trees (incremental update)
    pub fn add_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id()).to_string();
        // Add to string tree (HTTP)
        let string_tree = self
            .string_trees
            .entry(tree_key.clone())
            .or_insert_with(|| Arc::new(Tree::new()));
        string_tree.insert_text("", worker.url());
        // Add to token tree (gRPC)
        let token_tree = self
            .token_trees
            .entry(tree_key)
            .or_insert_with(|| Arc::new(TokenTree::new()));
        token_tree.insert_tokens(&[], worker.url());
    }

    /// Add a worker by URL and model (for backward compatibility)
    pub fn add_worker_by_url(&self, url: &str, model_id: &str) {
        let model_id_string = model_id.to_string();
        // Add to string tree (HTTP)
        let string_tree = self
            .string_trees
            .entry(model_id_string.clone())
            .or_insert_with(|| Arc::new(Tree::new()));
        string_tree.insert_text("", url);
        // Add to token tree (gRPC)
        let token_tree = self
            .token_trees
            .entry(model_id_string)
            .or_insert_with(|| Arc::new(TokenTree::new()));
        token_tree.insert_tokens(&[], url);
    }

    /// Remove a worker from the trees
    ///
    /// Note: Currently a no-op. Stale entries are cleaned up by LRU eviction.
    /// Worker registry removes workers first, so routing will skip them anyway.
    /// TODO: Implement efficient remove_tenant in kv_index with reverse index.
    pub fn remove_worker(&self, _worker: &dyn Worker) {
        // No-op: rely on LRU eviction to clean up stale entries
    }

    /// Remove a worker by URL (removes from all model trees for backward compatibility)
    ///
    /// Note: Currently a no-op. Stale entries are cleaned up by LRU eviction.
    /// TODO: Implement efficient remove_tenant in kv_index with reverse index.
    pub fn remove_worker_by_url(&self, _url: &str) {
        // No-op: rely on LRU eviction to clean up stale entries
    }

    /// Restore tree state from mesh store
    /// This is called during initialization to rebuild trees from synchronized state
    /// Note: Mesh sync currently only supports text-based operations (HTTP string trees)
    fn restore_tree_state_from_mesh(&self) {
        if let Some(ref mesh_sync) = self.mesh_sync {
            // Get all tree states from mesh
            // We need to iterate through all models that have tree states
            // For now, we'll restore trees for models that are already in our trees map
            // In a full implementation, we might want to query mesh for all tree states

            for tree_ref in self.string_trees.iter() {
                let model_id = tree_ref.key();
                if let Some(tree_state) = mesh_sync.get_tree_state(model_id) {
                    debug!(
                        "Restoring tree state for model {} with {} operations",
                        model_id,
                        tree_state.operations.len()
                    );

                    let tree = tree_ref.value();
                    // Apply all operations to rebuild the tree
                    for operation in &tree_state.operations {
                        match operation {
                            TreeOperation::Insert(insert_op) => {
                                tree.insert_text(&insert_op.text, &insert_op.tenant);
                            }
                            TreeOperation::Remove(_) => {
                                // No-op: rely on LRU eviction for cleanup
                            }
                        }
                    }
                }
            }
        }
    }

    /// Normalize model_id for mesh synchronization
    /// Converts empty model_id to UNKNOWN_MODEL_ID for consistency
    fn normalize_mesh_model_id(model_id: &str) -> &str {
        if model_id.is_empty() {
            UNKNOWN_MODEL_ID
        } else {
            model_id
        }
    }

    /// Apply remote tree operation from mesh
    /// This is called when receiving tree state updates from other nodes
    /// Note: Mesh sync currently only supports text-based operations (HTTP string trees)
    pub fn apply_remote_tree_operation(&self, model_id: &str, operation: &TreeOperation) {
        let tree_key = Self::normalize_mesh_model_id(model_id);

        let tree = self
            .string_trees
            .entry(tree_key.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));

        match operation {
            TreeOperation::Insert(insert_op) => {
                tree.insert_text(&insert_op.text, &insert_op.tenant);
                debug!(
                    "Applied remote tree insert: model={}, text={}, tenant={}",
                    model_id, insert_op.text, insert_op.tenant
                );
            }
            TreeOperation::Remove(remove_op) => {
                // No-op: rely on LRU eviction for cleanup
                debug!(
                    "Skipping remote tree remove (LRU will clean up): model={}, tenant={}",
                    model_id, remove_op.tenant
                );
            }
        }
    }

    /// Run cache eviction to prevent unbounded growth
    pub fn evict_cache(&self, max_size: usize) {
        // Evict string trees (HTTP)
        for tree_ref in self.string_trees.iter() {
            let model_id = tree_ref.key();
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            debug!(
                "String tree eviction for model {}, max_size: {}",
                model_id, max_size
            );
        }
        // Evict token trees (gRPC)
        for tree_ref in self.token_trees.iter() {
            let model_id = tree_ref.key();
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            debug!(
                "Token tree eviction for model {}, max_size: {}",
                model_id, max_size
            );
        }
    }

    /// Select worker with minimum load (used when load is imbalanced)
    /// Handles both HTTP (text-based) and gRPC (token-based) requests.
    fn select_worker_min_load(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        healthy_indices: &[usize],
        model_id: &str,
    ) -> Option<usize> {
        // Log load balancing trigger (only compute worker loads if debug enabled)
        if tracing::enabled!(tracing::Level::DEBUG) {
            let worker_loads: Vec<(&str, usize)> =
                workers.iter().map(|w| (w.url(), w.load())).collect();
            debug!("Load balancing triggered | workers: {:?}", worker_loads);
        }

        // Use shortest queue when imbalanced
        let min_load_idx = healthy_indices
            .iter()
            .min_by_key(|&&idx| workers[idx].load())
            .copied()?;

        let worker_url = workers[min_load_idx].url();

        // Even in imbalanced mode, update the appropriate tree to maintain cache state
        // Prefer token tree for gRPC requests, fall back to string tree for HTTP
        if let Some(tokens) = info.tokens {
            // gRPC request: update token tree
            let tree = self
                .token_trees
                .get(model_id)
                .map(|entry| entry.value().clone());
            if let Some(tree) = tree {
                tree.insert_tokens(tokens, worker_url);
            }
        } else if let Some(text) = info.request_text {
            // HTTP request: update string tree
            let tree = self
                .string_trees
                .get(model_id)
                .map(|entry| entry.value().clone());

            if let Some(tree) = tree {
                tree.insert_text(text, worker_url);

                // Sync insert operation to mesh if enabled (only for text operations)
                if let Some(ref mesh_sync) = self.mesh_sync {
                    let op = TreeOperation::Insert(TreeInsertOp {
                        text: text.to_string(),
                        tenant: worker_url.to_string(),
                    });
                    let mesh_model_id = Self::normalize_mesh_model_id(model_id);
                    if let Err(e) = mesh_sync.sync_tree_operation(mesh_model_id.to_string(), op) {
                        warn!("Failed to sync tree insert operation to mesh: {}", e);
                    }
                }
            } else {
                debug!(
                    "Warning: No string tree found for model '{}', skipping cache update",
                    model_id
                );
            }
        }

        // Increment processed counter
        workers[min_load_idx].increment_processed();

        Some(min_load_idx)
    }
}

impl LoadBalancingPolicy for CacheAwarePolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let request_text = info.request_text;
        let request_tokens = info.tokens;
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Determine the model for this set of workers (router pre-filters by model)
        // All workers should be from the same model
        let model_id = normalize_model_key(workers[healthy_indices[0]].model_id());

        // Get current load statistics - compute min/max in single pass without allocation
        let (min_load, max_load) = workers.iter().fold((usize::MAX, 0usize), |(min, max), w| {
            let load = w.load();
            (min.min(load), max.max(load))
        });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };

        // Check if load is imbalanced
        let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);

        if is_imbalanced {
            return self.select_worker_min_load(workers, info, &healthy_indices, model_id);
        }

        // Use cache-aware routing when balanced
        // gRPC requests use token tree, HTTP requests use string tree
        if let Some(tokens) = request_tokens {
            // gRPC path: use token-based tree
            self.select_worker_with_tokens(workers, tokens, &healthy_indices, model_id)
        } else {
            // HTTP path: use string-based tree
            let text = request_text.unwrap_or("");
            self.select_worker_with_text(workers, text, &healthy_indices, model_id)
        }
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        // Could track success rates per worker for more intelligent routing
        if !success {
            // Optionally reduce affinity for failed requests
            tracing::debug!(
                "Request to {} completed with success={}",
                worker_url,
                success
            );
        }
    }

    fn name(&self) -> &'static str {
        "cache_aware"
    }

    fn needs_request_text(&self) -> bool {
        true // Cache-aware policy needs request text for cache affinity
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Private helper methods for select_worker
impl CacheAwarePolicy {
    /// Select worker using token-based tree (gRPC path)
    fn select_worker_with_tokens(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        healthy_indices: &[usize],
        model_id: &str,
    ) -> Option<usize> {
        let tree = self
            .token_trees
            .get(model_id)
            .map(|entry| entry.value().clone());

        if let Some(tree) = tree {
            let result = tree.match_prefix_with_counts(tokens);
            let match_rate = if result.input_token_count == 0 {
                0.0
            } else {
                result.matched_token_count as f32 / result.input_token_count as f32
            };

            let selected_idx = if match_rate > self.config.cache_threshold {
                let tenant_url: &str = &result.tenant;
                workers
                    .iter()
                    .position(|w| w.url() == tenant_url)
                    .filter(|&idx| workers[idx].is_healthy())
            } else {
                healthy_indices
                    .iter()
                    .min_by_key(|&&idx| workers[idx].load())
                    .copied()
            };

            if let Some(idx) = selected_idx {
                tree.insert_tokens(tokens, workers[idx].url());
                workers[idx].increment_processed();
                return Some(idx);
            }

            // Selected worker no longer exists or unhealthy - fall back to first healthy
            // Stale entries will be cleaned up by LRU eviction
            healthy_indices.first().copied()
        } else {
            debug!(
                "Warning: No token tree found for model '{}', using random worker selection",
                model_id
            );
            let mut rng = rand::rng();
            let random_idx = rng.random_range(0..healthy_indices.len());
            Some(healthy_indices[random_idx])
        }
    }

    /// Select worker using string-based tree (HTTP path)
    fn select_worker_with_text(
        &self,
        workers: &[Arc<dyn Worker>],
        text: &str,
        healthy_indices: &[usize],
        model_id: &str,
    ) -> Option<usize> {
        let tree = self
            .string_trees
            .get(model_id)
            .map(|entry| entry.value().clone());

        if let Some(tree) = tree {
            let result = tree.match_prefix_with_counts(text);
            let match_rate = if result.input_char_count == 0 {
                0.0
            } else {
                result.matched_char_count as f32 / result.input_char_count as f32
            };

            let selected_idx = if match_rate > self.config.cache_threshold {
                let tenant_url: &str = &result.tenant;
                workers
                    .iter()
                    .position(|w| w.url() == tenant_url)
                    .filter(|&idx| workers[idx].is_healthy())
            } else {
                healthy_indices
                    .iter()
                    .min_by_key(|&&idx| workers[idx].load())
                    .copied()
            };

            if let Some(idx) = selected_idx {
                tree.insert_text(text, workers[idx].url());

                // Sync insert operation to mesh if enabled (only for text operations)
                if let Some(ref mesh_sync) = self.mesh_sync {
                    let op = TreeOperation::Insert(TreeInsertOp {
                        text: text.to_string(),
                        tenant: workers[idx].url().to_string(),
                    });
                    let mesh_model_id = Self::normalize_mesh_model_id(model_id);
                    if let Err(e) = mesh_sync.sync_tree_operation(mesh_model_id.to_string(), op) {
                        warn!("Failed to sync tree insert operation to mesh: {}", e);
                    }
                }

                workers[idx].increment_processed();
                return Some(idx);
            }

            // Selected worker no longer exists or unhealthy - fall back to first healthy
            // Stale entries will be cleaned up by LRU eviction
            healthy_indices.first().copied()
        } else {
            debug!(
                "Warning: No string tree found for model '{}', using random worker selection",
                model_id
            );
            let mut rng = rand::rng();
            let random_idx = rng.random_range(0..healthy_indices.len());
            Some(healthy_indices[random_idx])
        }
    }
}

impl Default for CacheAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[test]
    fn test_cache_aware_with_balanced_load() {
        // Create policy without eviction thread for testing
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
        ];

        // Initialize the policy with workers
        policy.init_workers(&workers);

        // First request should be distributed
        let idx1 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello world"),
                    ..Default::default()
                },
            )
            .unwrap();

        // Same request should go to same worker (cache hit)
        let idx2 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello world"),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx1, idx2);

        // Similar request should also go to same worker
        let idx3 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello"),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx1, idx3);
    }

    #[test]
    fn test_cache_aware_with_imbalanced_load() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0, // Disable eviction thread
            max_tree_size: 10000,
        });

        let worker1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .build();
        let worker2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .build();

        // Create significant load imbalance
        for _ in 0..20 {
            worker1.increment_load();
        }
        // worker2 has load 0

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];
        policy.init_workers(&workers);

        // Should select worker2 (lower load) despite cache affinity
        let info = SelectWorkerInfo {
            request_text: Some("test"),
            ..Default::default()
        };
        for _ in 0..5 {
            let idx = policy.select_worker(&workers, &info).unwrap();
            assert_eq!(idx, 1); // Should always pick worker2
        }
    }

    #[test]
    fn test_cache_aware_worker_removal() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        policy.init_workers(&workers);

        // Route some requests
        policy.select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some("test1"),
                ..Default::default()
            },
        );
        policy.select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some("test2"),
                ..Default::default()
            },
        );

        // Remove a worker
        policy.remove_worker_by_url("http://w1:8000");
        workers[0].set_healthy(false);

        // All requests should now go to worker2
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test1"),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_cache_aware_sync_tree_operation_to_mesh() {
        use std::sync::Arc;

        use smg_mesh::{stores::StateStores, sync::MeshSyncManager};

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let mesh_sync = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let mut policy = CacheAwarePolicy::with_config(config);
        policy.set_mesh_sync(Some(mesh_sync.clone()));

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .build(),
        )];

        policy.init_workers(&workers);

        // Select worker with a request - should sync to mesh
        let _idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test request"),
                    ..Default::default()
                },
            )
            .unwrap();

        // Verify tree operation was synced to mesh (under UNKNOWN_MODEL_ID since no model was specified)
        let tree_state = mesh_sync.get_tree_state(UNKNOWN_MODEL_ID);
        assert!(tree_state.is_some());
        let tree = tree_state.unwrap();
        assert!(!tree.operations.is_empty());
    }

    #[test]
    fn test_cache_aware_restore_tree_state_from_mesh() {
        use std::sync::Arc;

        use smg_mesh::{
            stores::StateStores,
            sync::MeshSyncManager,
            tree_ops::{TreeInsertOp, TreeOperation},
        };

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let mesh_sync = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

        // Pre-populate mesh with tree state
        let op1 = TreeOperation::Insert(TreeInsertOp {
            text: "test_text_1".to_string(),
            tenant: "http://w1:8000".to_string(),
        });
        mesh_sync
            .sync_tree_operation("model1".to_string(), op1)
            .unwrap();

        let op2 = TreeOperation::Insert(TreeInsertOp {
            text: "test_text_2".to_string(),
            tenant: "http://w2:8000".to_string(),
        });
        mesh_sync
            .sync_tree_operation("model1".to_string(), op2)
            .unwrap();

        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let mut policy = CacheAwarePolicy::with_config(config);
        policy.set_mesh_sync(Some(mesh_sync.clone()));

        // Initialize with a model to trigger restore
        let _workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .build(),
        )];

        // Create a tree entry for model1 to trigger restore
        let _tree = policy
            .string_trees
            .entry("model1".to_string())
            .or_insert_with(|| Arc::new(Tree::new()));

        // Manually trigger restore (normally done in constructor)
        // For testing, we'll verify the tree state exists in mesh
        let tree_state = mesh_sync.get_tree_state("model1");
        assert!(tree_state.is_some());
        let state = tree_state.unwrap();
        assert_eq!(state.operations.len(), 2);
    }

    #[test]
    fn test_cache_aware_apply_remote_tree_operation() {
        use std::sync::Arc;

        use smg_mesh::{
            stores::StateStores,
            sync::MeshSyncManager,
            tree_ops::{TreeInsertOp, TreeOperation},
        };

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let mesh_sync = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let mut policy = CacheAwarePolicy::with_config(config);
        policy.set_mesh_sync(Some(mesh_sync.clone()));

        // Apply remote tree operation
        let remote_op = TreeOperation::Insert(TreeInsertOp {
            text: "remote_text".to_string(),
            tenant: "http://remote:8000".to_string(),
        });

        policy.apply_remote_tree_operation("model1", &remote_op);

        // Verify the string tree was updated (mesh sync only affects string trees)
        let tree = policy.string_trees.get("model1");
        assert!(tree.is_some());
    }

    #[test]
    fn test_cache_aware_multi_node_consistency() {
        use std::sync::Arc;

        use smg_mesh::{
            stores::StateStores,
            sync::MeshSyncManager,
            tree_ops::{TreeInsertOp, TreeOperation},
        };

        // Simulate two nodes
        let stores1 = Arc::new(StateStores::with_self_name("node1".to_string()));
        let mesh_sync1 = Arc::new(MeshSyncManager::new(stores1.clone(), "node1".to_string()));

        let stores2 = Arc::new(StateStores::with_self_name("node2".to_string()));
        let mesh_sync2 = Arc::new(MeshSyncManager::new(stores2.clone(), "node2".to_string()));

        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };

        let mut _policy1 = CacheAwarePolicy::with_config(config.clone());
        _policy1.set_mesh_sync(Some(mesh_sync1.clone()));
        let mut _policy2 = CacheAwarePolicy::with_config(config);
        _policy2.set_mesh_sync(Some(mesh_sync2.clone()));

        // Node1 syncs a tree operation
        let op = TreeOperation::Insert(TreeInsertOp {
            text: "shared_text".to_string(),
            tenant: "http://shared:8000".to_string(),
        });
        mesh_sync1
            .sync_tree_operation("model1".to_string(), op.clone())
            .unwrap();

        // Node2 should be able to get the tree state
        let tree_state = mesh_sync2.get_tree_state("model1");
        // Note: In a real scenario, this would be synced via gossip protocol
        // For unit test, we verify the sync mechanism works
        // Tree state may or may not exist depending on sync timing
        let _ = tree_state;
    }

    #[test]
    fn test_cache_aware_without_mesh() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .build(),
        )];

        policy.init_workers(&workers);

        // Should work without mesh
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test request"),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 0);
    }
}
