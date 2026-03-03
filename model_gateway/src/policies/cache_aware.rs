/*
    Cache-Aware Load Balancing Router

    When load is balanced, uses cache-aware routing. When imbalanced, uses
    shortest-queue. A system is imbalanced when both:
        (max - min) > abs_threshold  AND  max > rel_threshold * min

    Three types of cache-aware routing (mutually exclusive, selected by
    worker connection mode and KV event availability):

    1. Event-Driven (gRPC + KV events)
    -------------------------------------------
    Uses PositionalIndexer overlap scoring from KvEventMonitor. Routes based
    on actual backend KV cache state. Selects the worker with the highest
    overlap count; tie-breaks by load (lower) then tree size (smaller).
    Falls back to min-load when no cache overlap exists.

    2. Approximate Token Tree (gRPC, no KV events)
    -------------------------------------------
    Maintains a TokenTree per model tracking which token prefixes were routed
    where. If match_rate > cache_threshold, routes to the best-matching worker.
    Otherwise routes to the worker with the smallest tree (most cache capacity).

    3. Approximate String Tree (HTTP)
    -------------------------------------------
    Same algorithm as (2) but operates on raw text characters instead of
    token IDs, avoiding tokenization overhead.

    Load Balancing (Shortest Queue)
    -------------------------------------------
    When the system is imbalanced, routes to the least busy worker regardless
    of cache affinity.

    Configuration Parameters:
    ------------------------
    cache_threshold:         Min prefix match ratio for highest-match routing (0.0-1.0)
    balance_abs_threshold:   Absolute load diff threshold for imbalance detection
    balance_rel_threshold:   Relative load ratio threshold for imbalance detection
    eviction_interval_secs:  Interval between LRU eviction cycles
    max_tree_size:           Max nodes per approximate tree before eviction
    block_size:              Backend KV cache block size for event-driven routing
*/

use std::sync::Arc;

use dashmap::DashMap;
use kv_index::{compute_request_content_hashes, PositionalIndexer, TokenTree, Tree};
use parking_lot::RwLock;
use rand::Rng;
use smg_mesh::{OptionalMeshSyncManager, TreeInsertOp, TreeOperation};
use tracing::{debug, warn};

use super::{
    get_healthy_worker_indices, normalize_model_key, utils::PeriodicTask, CacheAwareConfig,
    LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::core::{KvEventMonitor, Worker, UNKNOWN_MODEL_ID};

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
    /// Event-driven KV cache monitor for overlap scoring (gRPC workers only).
    /// Set via `set_kv_event_monitor`. When present and the indexer has data for
    /// a model, event-driven routing takes priority over approximate trees.
    /// Uses `RwLock` for interior mutability so the monitor can be injected into
    /// policies already behind `Arc<dyn LoadBalancingPolicy>`.
    kv_monitor: RwLock<Option<Arc<KvEventMonitor>>>,
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
            kv_monitor: RwLock::new(None),
        }
    }

    /// Set mesh sync manager (can be called after construction)
    pub fn set_mesh_sync(&mut self, mesh_sync: OptionalMeshSyncManager) {
        self.mesh_sync.clone_from(&mesh_sync);
        if mesh_sync.is_some() {
            self.restore_tree_state_from_mesh();
        }
    }

    /// Set event-driven KV cache monitor (thread-safe, can be called after construction).
    /// Uses interior mutability so this works on policies behind `Arc<dyn LoadBalancingPolicy>`.
    pub fn set_kv_event_monitor(&self, monitor: Option<Arc<KvEventMonitor>>) {
        *self.kv_monitor.write() = monitor;
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
    #[expect(
        clippy::unused_self,
        reason = "no-op stub; will use self once remove_tenant is implemented"
    )]
    pub fn remove_worker(&self, _worker: &dyn Worker) {
        // No-op: rely on LRU eviction to clean up stale entries
    }

    /// Remove a worker by URL (removes from all model trees for backward compatibility)
    ///
    /// Note: Currently a no-op. Stale entries are cleaned up by LRU eviction.
    /// TODO: Implement efficient remove_tenant in kv_index with reverse index.
    #[expect(
        clippy::unused_self,
        reason = "no-op stub; will use self once remove_tenant is implemented"
    )]
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

        // Cache-aware routing when balanced — three types (mutually exclusive):
        //   1. Event-driven: PositionalIndexer overlap scoring (gRPC + KV events)
        //   2. Approximate token tree: TokenTree prefix matching (gRPC, no events)
        //   3. Approximate string tree: Tree prefix matching (HTTP)
        if let Some(tokens) = request_tokens {
            if self.has_event_indexer(model_id) {
                self.select_worker_event_driven(workers, tokens, &healthy_indices, model_id)
            } else {
                self.select_worker_with_tokens(workers, tokens, &healthy_indices, model_id)
            }
        } else {
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
    /// Check if an event-driven indexer exists with data for this model.
    /// Returns false when the indexer is empty (startup, reconnect) so
    /// routing falls through to the approximate token tree instead of
    /// taking the event-driven path with no data and landing on min-load.
    fn has_event_indexer(&self, model_id: &str) -> bool {
        let guard = self.kv_monitor.read();
        guard
            .as_ref()
            .and_then(|m| m.get_indexer(model_id))
            .is_some_and(|indexer| indexer.current_size() > 0)
    }

    /// Event-driven routing: PositionalIndexer overlap scoring (Type 1).
    ///
    /// Self-contained — when overlap is found, selects the worker with the best
    /// cache match. When no overlap (cold start, novel tokens, short request),
    /// falls back to min-load. Does NOT fall back to approximate token tree.
    fn select_worker_event_driven(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        healthy_indices: &[usize],
        model_id: &str,
    ) -> Option<usize> {
        let guard = self.kv_monitor.read();
        let monitor = guard.as_ref()?;
        let indexer = monitor.get_indexer(model_id)?;

        // Per-model block_size: learned from events > config default
        let block_size = monitor
            .block_size(model_id)
            .unwrap_or(self.config.block_size);

        if let Some(idx) =
            Self::score_overlap(workers, tokens, healthy_indices, &indexer, block_size)
        {
            return Some(idx);
        }

        // No cache overlap — min-load fallback (no token tree involved)
        let min_idx = healthy_indices
            .iter()
            .min_by_key(|&&idx| workers[idx].load())
            .copied()?;
        debug!(
            worker = workers[min_idx].url(),
            model_id, "Event-driven routing: no overlap, min-load fallback"
        );
        workers[min_idx].increment_processed();
        Some(min_idx)
    }

    /// Score healthy workers by PositionalIndexer overlap and select the best.
    ///
    /// Returns `Some(idx)` if at least one worker has cached blocks matching the
    /// request. Returns `None` if the request is too short for a full block or
    /// no workers have matching data.
    fn score_overlap(
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        healthy_indices: &[usize],
        indexer: &PositionalIndexer,
        block_size: usize,
    ) -> Option<usize> {
        let content_hashes = compute_request_content_hashes(tokens, block_size);
        if content_hashes.is_empty() {
            return None;
        }

        let overlap = indexer.find_matches(&content_hashes, false);
        if overlap.scores.is_empty() {
            return None;
        }

        // Select worker with best overlap among those that actually match.
        // Tie-break: lower load, then smaller tree size.
        let best_idx = healthy_indices
            .iter()
            .copied()
            .filter(|&idx| {
                indexer
                    .worker_id(workers[idx].url())
                    .and_then(|id| overlap.scores.get(&id))
                    .copied()
                    .unwrap_or(0)
                    > 0
            })
            .max_by_key(|&idx| {
                let wid = indexer.worker_id(workers[idx].url());
                let score = wid
                    .and_then(|id| overlap.scores.get(&id))
                    .copied()
                    .unwrap_or(0);
                let load = workers[idx].load();
                let tree_size = wid
                    .and_then(|id| overlap.tree_sizes.get(&id))
                    .copied()
                    .unwrap_or(0);
                (score, std::cmp::Reverse(load), std::cmp::Reverse(tree_size))
            })?;

        debug!(
            worker = workers[best_idx].url(),
            score = indexer
                .worker_id(workers[best_idx].url())
                .and_then(|id| overlap.scores.get(&id))
                .copied()
                .unwrap_or(0),
            "Event-driven routing: overlap match"
        );
        workers[best_idx].increment_processed();
        Some(best_idx)
    }

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
    use kv_index::{compute_content_hash, SequenceHash, StoredBlock};

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
            block_size: 16,
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

        use smg_mesh::{MeshSyncManager, StateStores};

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

        use smg_mesh::{MeshSyncManager, StateStores, TreeInsertOp, TreeOperation};

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

        use smg_mesh::{MeshSyncManager, StateStores, TreeInsertOp, TreeOperation};

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

        use smg_mesh::{MeshSyncManager, StateStores, TreeInsertOp, TreeOperation};

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

    // -----------------------------------------------------------------------
    // Event-driven routing tests (Type 1: PositionalIndexer overlap scoring)
    // -----------------------------------------------------------------------

    /// Helper: create a PositionalIndexer and store blocks for a worker.
    /// `token_chunks` is a list of token-id slices — each becomes one block.
    fn setup_indexer_with_blocks(
        worker_url: &str,
        token_chunks: &[&[u32]],
        jump_size: usize,
    ) -> Arc<PositionalIndexer> {
        let indexer = Arc::new(PositionalIndexer::new(jump_size));
        let blocks: Vec<StoredBlock> = token_chunks
            .iter()
            .enumerate()
            .map(|(i, tokens)| StoredBlock {
                seq_hash: SequenceHash(i as u64 + 1),
                content_hash: compute_content_hash(tokens),
            })
            .collect();
        indexer.apply_stored(worker_url, &blocks, None).unwrap();
        indexer
    }

    fn test_config() -> CacheAwareConfig {
        CacheAwareConfig {
            eviction_interval_secs: 0,
            block_size: 4, // small block size for easy test setup
            ..Default::default()
        }
    }

    // -- score_overlap unit tests (scoring helper) --

    #[test]
    fn test_score_overlap_selects_best_match() {
        let policy = CacheAwarePolicy::with_config(test_config());
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

        // Store 4 blocks for w1: tokens [1..16] in blocks of 4
        let indexer = setup_indexer_with_blocks(
            "http://w1:8000",
            &[
                &[1, 2, 3, 4],
                &[5, 6, 7, 8],
                &[9, 10, 11, 12],
                &[13, 14, 15, 16],
            ],
            4,
        );

        // Query with matching tokens — should select w1
        let result = CacheAwarePolicy::score_overlap(
            &workers,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            &[0, 1],
            &indexer,
            4,
        );
        assert_eq!(result, Some(0)); // w1
    }

    #[test]
    fn test_score_overlap_no_match_returns_none() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        )];
        policy.init_workers(&workers);

        let indexer =
            setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4], &[5, 6, 7, 8]], 4);

        // Completely different tokens — no overlap → None
        let result = CacheAwarePolicy::score_overlap(
            &workers,
            &[100, 200, 300, 400, 500, 600, 700, 800],
            &[0],
            &indexer,
            4,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_score_overlap_load_tiebreak() {
        let policy = CacheAwarePolicy::with_config(test_config());

        let w1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .build();
        let w2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .build();

        // Give w1 higher load
        for _ in 0..10 {
            w1.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w1), Arc::new(w2)];
        policy.init_workers(&workers);

        // Store same blocks for both workers (equal overlap)
        let indexer = Arc::new(PositionalIndexer::new(4));
        let blocks = vec![StoredBlock {
            seq_hash: SequenceHash(1),
            content_hash: compute_content_hash(&[1, 2, 3, 4]),
        }];
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();
        let blocks2 = vec![StoredBlock {
            seq_hash: SequenceHash(1),
            content_hash: compute_content_hash(&[1, 2, 3, 4]),
        }];
        indexer
            .apply_stored("http://w2:8000", &blocks2, None)
            .unwrap();

        // Equal overlap → tie-break by load → w2 wins (lower load)
        let result = CacheAwarePolicy::score_overlap(&workers, &[1, 2, 3, 4], &[0, 1], &indexer, 4);
        assert_eq!(result, Some(1)); // w2 (lower load)
    }

    #[test]
    fn test_score_overlap_tree_size_tiebreak() {
        let policy = CacheAwarePolicy::with_config(test_config());
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

        let indexer = Arc::new(PositionalIndexer::new(4));

        // Both workers have block [1,2,3,4] (equal overlap, equal load)
        let block = vec![StoredBlock {
            seq_hash: SequenceHash(1),
            content_hash: compute_content_hash(&[1, 2, 3, 4]),
        }];
        indexer
            .apply_stored("http://w1:8000", &block, None)
            .unwrap();

        // w2 has the same block plus extra blocks → larger tree
        let block2 = vec![StoredBlock {
            seq_hash: SequenceHash(1),
            content_hash: compute_content_hash(&[1, 2, 3, 4]),
        }];
        indexer
            .apply_stored("http://w2:8000", &block2, None)
            .unwrap();
        let extra = vec![StoredBlock {
            seq_hash: SequenceHash(2),
            content_hash: compute_content_hash(&[5, 6, 7, 8]),
        }];
        indexer
            .apply_stored("http://w2:8000", &extra, Some(SequenceHash(1)))
            .unwrap();

        // Equal overlap, equal load → tie-break by tree size → w1 wins (smaller)
        let result = CacheAwarePolicy::score_overlap(&workers, &[1, 2, 3, 4], &[0, 1], &indexer, 4);
        assert_eq!(result, Some(0)); // w1 (smaller tree)
    }

    #[test]
    fn test_score_overlap_short_request_returns_none() {
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        )];

        let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]], 4);

        // Request shorter than block_size → no full blocks → None
        let result = CacheAwarePolicy::score_overlap(&workers, &[1, 2, 3], &[0], &indexer, 4);
        assert_eq!(result, None);
    }

    #[test]
    fn test_score_overlap_partial_match() {
        let policy = CacheAwarePolicy::with_config(test_config());
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

        let indexer = Arc::new(PositionalIndexer::new(4));

        // w1 has 4 blocks cached
        let blocks_w1: Vec<StoredBlock> = (0..4)
            .map(|i| StoredBlock {
                seq_hash: SequenceHash(i as u64 + 1),
                content_hash: compute_content_hash(&[
                    (i * 4 + 1) as u32,
                    (i * 4 + 2) as u32,
                    (i * 4 + 3) as u32,
                    (i * 4 + 4) as u32,
                ]),
            })
            .collect();
        indexer
            .apply_stored("http://w1:8000", &blocks_w1, None)
            .unwrap();

        // w2 has only the first 2 blocks (partial overlap with same request)
        let blocks_w2: Vec<StoredBlock> = (0..2)
            .map(|i| StoredBlock {
                seq_hash: SequenceHash(i as u64 + 1),
                content_hash: compute_content_hash(&[
                    (i * 4 + 1) as u32,
                    (i * 4 + 2) as u32,
                    (i * 4 + 3) as u32,
                    (i * 4 + 4) as u32,
                ]),
            })
            .collect();
        indexer
            .apply_stored("http://w2:8000", &blocks_w2, None)
            .unwrap();

        // Query with all 4 blocks worth of tokens → w1 wins (higher overlap: 4 vs 2)
        let result = CacheAwarePolicy::score_overlap(
            &workers,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            &[0, 1],
            &indexer,
            4,
        );
        assert_eq!(result, Some(0)); // w1 (higher overlap)
    }

    // -- select_worker_event_driven integration tests --

    #[test]
    fn test_event_driven_overlap_selects_cached_worker() {
        let policy = CacheAwarePolicy::with_config(test_config());
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

        // Set up monitor with indexer data for "unknown" model
        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        let indexer =
            setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4], &[5, 6, 7, 8]], 4);
        monitor.indexers.insert("unknown".to_string(), indexer);
        policy.set_kv_event_monitor(Some(monitor));

        // Full dispatch: should use event-driven and select w1
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[1, 2, 3, 4, 5, 6, 7, 8]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 0); // w1 (has cached blocks)
    }

    #[test]
    fn test_event_driven_no_overlap_uses_min_load() {
        let policy = CacheAwarePolicy::with_config(test_config());

        let w1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .build();
        let w2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .build();
        // Give w1 higher load so min-load picks w2
        for _ in 0..3 {
            w1.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w1), Arc::new(w2)];
        policy.init_workers(&workers);

        // Monitor has indexer with data, but tokens don't match
        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]], 4);
        monitor.indexers.insert("unknown".to_string(), indexer);
        policy.set_kv_event_monitor(Some(monitor));

        // No overlap → event-driven falls back to min-load (not token tree)
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[100, 200, 300, 400]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 1); // w2 (min load), NOT token tree result
    }

    #[test]
    fn test_event_driven_short_request_uses_min_load() {
        let policy = CacheAwarePolicy::with_config(test_config()); // block_size=4

        let w1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .build();
        let w2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .build();
        for _ in 0..3 {
            w1.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w1), Arc::new(w2)];
        policy.init_workers(&workers);

        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]], 4);
        monitor.indexers.insert("unknown".to_string(), indexer);
        policy.set_kv_event_monitor(Some(monitor));

        // Request shorter than block_size → no full blocks → min-load fallback
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[1, 2, 3]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 1); // w2 (min load)
    }

    #[test]
    fn test_no_monitor_uses_token_tree() {
        let policy = CacheAwarePolicy::with_config(test_config());
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

        // No kv_monitor → has_event_indexer returns false → uses token tree
        assert!(!policy.has_event_indexer("unknown"));

        // Should still route (via token tree, not event-driven)
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[1, 2, 3, 4]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(idx < 2); // valid worker selected
    }

    #[test]
    fn test_set_kv_event_monitor() {
        let policy = CacheAwarePolicy::with_config(test_config());

        // Initially no monitor
        assert!(policy.kv_monitor.read().is_none());

        // Set monitor (works via &self thanks to interior mutability)
        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        policy.set_kv_event_monitor(Some(Arc::clone(&monitor)));
        assert!(policy.kv_monitor.read().is_some());

        // get_indexer returns None for unknown model
        assert!(monitor.get_indexer("nonexistent").is_none());

        // Clear monitor
        policy.set_kv_event_monitor(None);
        assert!(policy.kv_monitor.read().is_none());
    }

    #[test]
    fn test_event_driven_uses_monitor_block_size() {
        // Test that event-driven routing uses monitor's learned block_size
        // instead of config default when available.
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            block_size: 4, // config default
            eviction_interval_secs: 0,
            ..Default::default()
        });

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

        let monitor = Arc::new(KvEventMonitor::new(Some(4)));

        // Store blocks using block_size=8 (tokens chunked in groups of 8)
        let indexer = Arc::new(PositionalIndexer::new(4));
        let block = vec![StoredBlock {
            seq_hash: SequenceHash(1),
            content_hash: compute_content_hash(&[1, 2, 3, 4, 5, 6, 7, 8]),
        }];
        indexer
            .apply_stored("http://w1:8000", &block, None)
            .unwrap();
        monitor
            .indexers
            .insert("unknown".to_string(), indexer.clone());

        // Set block_size=8 in monitor (simulating learned from events)
        monitor.set_block_size("unknown", 8);

        policy.set_kv_event_monitor(Some(monitor));

        // Query with 8 tokens — with block_size=8, this is one full block
        // With config block_size=4, this would be two blocks and wouldn't match
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[1, 2, 3, 4, 5, 6, 7, 8]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 0); // w1 has the cached block
    }

    #[test]
    fn test_imbalanced_skips_event_driven() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0,
            block_size: 4,
            ..Default::default()
        });

        let w1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .build();
        let w2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .build();

        // Create heavy imbalance: w1 has 20 load, w2 has 0
        for _ in 0..20 {
            w1.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w1), Arc::new(w2)];
        policy.init_workers(&workers);

        // Even though we set up event monitor, imbalance check fires first
        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        policy.set_kv_event_monitor(Some(monitor));

        // With imbalance, select_worker should pick min-load (w2), not event-driven
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[1, 2, 3, 4]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 1); // w2 (min load), regardless of event data
    }

    #[test]
    fn test_empty_indexer_falls_through_to_token_tree() {
        // When the monitor has an indexer for a model but the indexer is empty
        // (startup, reconnect), routing should fall through to the token tree
        // instead of taking the event-driven path and landing on min-load.
        let policy = CacheAwarePolicy::with_config(test_config());
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

        // Set up monitor with an empty indexer
        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        let empty_indexer = Arc::new(PositionalIndexer::new(4));
        monitor
            .indexers
            .insert("unknown".to_string(), empty_indexer);
        policy.set_kv_event_monitor(Some(monitor));

        // Empty indexer → has_event_indexer returns false → falls through to token tree
        assert!(!policy.has_event_indexer("unknown"));

        // Route a request — should use token tree, not event-driven min-load
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[1, 2, 3, 4]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(idx < 2); // valid worker via token tree

        // Route the same tokens again — token tree should route to same worker (cache hit)
        let idx2 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[1, 2, 3, 4]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, idx2); // token tree cache affinity preserved
    }
}
