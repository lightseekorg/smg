//! Sharded positional indexer for cache-aware routing.
//!
//! Wraps N [`PositionalIndexer`] instances, each running on a dedicated OS thread.
//! Workers are sticky-assigned to shards (round-robin). Events for a worker are sent
//! to its assigned shard's channel; queries are broadcast to all shards (scatter-gather)
//! and the partial [`OverlapScores`] are merged.
//!
//! This eliminates DashMap contention between concurrent readers and writers — each
//! shard's PositionalIndexer is accessed by a single thread only.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU32, AtomicUsize, Ordering},
        Arc,
    },
    thread,
};

use dashmap::DashMap;
use rustc_hash::FxBuildHasher;

use crate::event_tree::{
    ApplyError, ContentHash, OverlapScores, PositionalIndexer, SequenceHash, StoredBlock,
    WorkerBlockMap,
};

/// Maximum number of workers supported (matches PositionalIndexer).
const MAX_WORKERS: usize = 2048;

/// Default number of shards when not specified.
const DEFAULT_NUM_SHARDS: usize = 4;

// ---------------------------------------------------------------------------
// Channel message types
// ---------------------------------------------------------------------------

/// Task sent to a shard thread.
enum ShardTask {
    ApplyStored {
        worker_id: u32,
        blocks: Vec<StoredBlock>,
        parent_seq_hash: Option<SequenceHash>,
        response_tx: flume::Sender<Result<(), ApplyError>>,
    },
    ApplyRemoved {
        worker_id: u32,
        seq_hashes: Vec<SequenceHash>,
    },
    ApplyCleared {
        worker_id: u32,
    },
    RemoveWorker {
        worker_id: u32,
    },
    FindMatches {
        content_hashes: Arc<[ContentHash]>,
        early_exit: bool,
        response_tx: flume::Sender<OverlapScores>,
    },
    Shutdown,
}

/// Handle to a single shard thread.
struct ShardHandle {
    tx: flume::Sender<ShardTask>,
    join_handle: Option<thread::JoinHandle<()>>,
}

// ---------------------------------------------------------------------------
// ShardedIndexer
// ---------------------------------------------------------------------------

/// Sharded positional indexer: N OS threads, each owning a private [`PositionalIndexer`].
///
/// Workers are sticky-assigned to shards (round-robin at intern time).
/// Write operations are fire-and-forget (except `apply_stored` which returns errors).
/// Read operations (find_matches) scatter to all shards and merge results.
pub struct ShardedIndexer {
    shards: Vec<ShardHandle>,
    /// worker_id → shard index.
    worker_to_shard: DashMap<u32, usize, FxBuildHasher>,
    /// worker URL → global worker_id.
    worker_to_id: DashMap<Arc<str>, u32, FxBuildHasher>,
    /// Monotonic counter for assigning new worker IDs.
    next_worker_id: AtomicU32,
    /// Per-worker block counts, updated by shard threads via shared reference.
    tree_sizes: Arc<[AtomicUsize]>,
    /// Number of shards.
    num_shards: usize,
    /// Jump size passed to each shard's PositionalIndexer.
    jump_size: usize,
    /// Round-robin counter for shard assignment.
    next_shard: AtomicUsize,
}

impl ShardedIndexer {
    /// Create a new ShardedIndexer with `num_shards` OS threads.
    ///
    /// `jump_size` controls the jump search stride (passed to each shard's PositionalIndexer).
    /// `num_shards` defaults to [`DEFAULT_NUM_SHARDS`] if 0.
    pub fn new(jump_size: usize, num_shards: usize) -> Self {
        assert!(jump_size > 0, "jump_size must be greater than 0");
        let num_shards = if num_shards == 0 {
            DEFAULT_NUM_SHARDS
        } else {
            num_shards
        };

        let tree_sizes: Arc<[AtomicUsize]> = (0..MAX_WORKERS)
            .map(|_| AtomicUsize::new(0))
            .collect::<Vec<_>>()
            .into();

        let mut shards = Vec::with_capacity(num_shards);
        for shard_id in 0..num_shards {
            let (tx, rx) = flume::unbounded();
            let tree_sizes_ref = Arc::clone(&tree_sizes);
            #[expect(clippy::expect_used, reason = "thread spawn failure is unrecoverable")]
            let handle = thread::Builder::new()
                .name(format!("kv-shard-{shard_id}"))
                .spawn(move || {
                    shard_loop(rx, jump_size, &tree_sizes_ref);
                })
                .expect("failed to spawn shard thread");

            shards.push(ShardHandle {
                tx,
                join_handle: Some(handle),
            });
        }

        Self {
            shards,
            worker_to_shard: DashMap::with_hasher(FxBuildHasher),
            worker_to_id: DashMap::with_hasher(FxBuildHasher),
            next_worker_id: AtomicU32::new(0),
            tree_sizes,
            num_shards,
            jump_size,
            next_shard: AtomicUsize::new(0),
        }
    }

    /// Intern a worker URL to an internal u32 ID and assign it to a shard.
    ///
    /// Thread-safe: concurrent calls for the same worker URL return the same ID.
    pub fn intern_worker(&self, worker: &str) -> u32 {
        // Fast path: already interned.
        if let Some(entry) = self.worker_to_id.get(worker) {
            return *entry.value();
        }

        // Slow path: DashMap entry API handles the race.
        let id = *self
            .worker_to_id
            .entry(Arc::from(worker))
            .or_insert_with(|| {
                let id = self.next_worker_id.fetch_add(1, Ordering::Relaxed);
                assert!(
                    (id as usize) < MAX_WORKERS,
                    "worker count {id} exceeds MAX_WORKERS ({MAX_WORKERS})"
                );
                // Assign to shard via round-robin.
                let shard = self.next_shard.fetch_add(1, Ordering::Relaxed) % self.num_shards;
                self.worker_to_shard.insert(id, shard);
                id
            })
            .value();
        id
    }

    /// Get the internal u32 ID for a worker URL, if it has been interned.
    pub fn worker_id(&self, worker: &str) -> Option<u32> {
        self.worker_to_id.get(worker).map(|entry| *entry.value())
    }

    /// Apply a "blocks stored" event for a worker.
    ///
    /// Sends the event to the worker's assigned shard and blocks until the shard
    /// processes it. Returns the result (needed for parent fallback in KvEventMonitor).
    pub fn apply_stored(
        &self,
        worker_id: u32,
        blocks: &[StoredBlock],
        parent_seq_hash: Option<SequenceHash>,
    ) -> Result<(), ApplyError> {
        if blocks.is_empty() {
            return Ok(());
        }
        let shard_idx = self.shard_for_worker(worker_id);
        let (response_tx, response_rx) = flume::bounded(1);
        let _ = self.shards[shard_idx].tx.send(ShardTask::ApplyStored {
            worker_id,
            blocks: blocks.to_vec(),
            parent_seq_hash,
            response_tx,
        });
        response_rx
            .recv()
            .unwrap_or(Err(ApplyError::WorkerNotTracked))
    }

    /// Apply a "blocks removed" event for a worker (fire-and-forget).
    pub fn apply_removed(&self, worker_id: u32, seq_hashes: &[SequenceHash]) {
        let shard_idx = self.shard_for_worker(worker_id);
        let _ = self.shards[shard_idx].tx.send(ShardTask::ApplyRemoved {
            worker_id,
            seq_hashes: seq_hashes.to_vec(),
        });
    }

    /// Apply a "cache cleared" event for a worker (fire-and-forget).
    pub fn apply_cleared(&self, worker_id: u32) {
        let shard_idx = self.shard_for_worker(worker_id);
        let _ = self.shards[shard_idx]
            .tx
            .send(ShardTask::ApplyCleared { worker_id });
    }

    /// Remove a worker entirely (fire-and-forget).
    ///
    /// The shard thread owns the WorkerBlockMap and handles cleanup.
    pub fn remove_worker(&self, worker_id: u32) {
        let shard_idx = self.shard_for_worker(worker_id);
        let _ = self.shards[shard_idx]
            .tx
            .send(ShardTask::RemoveWorker { worker_id });
    }

    /// Find overlap scores for a request's content hash sequence.
    ///
    /// Broadcasts to all shards (scatter), collects partial results (gather), merges.
    /// Since each worker exists in exactly one shard, the merge is a simple extend.
    pub fn find_matches(&self, content_hashes: &[ContentHash], early_exit: bool) -> OverlapScores {
        if content_hashes.is_empty() {
            return OverlapScores::default();
        }

        let content_hashes: Arc<[ContentHash]> = content_hashes.into();
        let mut receivers = Vec::with_capacity(self.num_shards);

        // Scatter: send FindMatches to all shards.
        for shard in &self.shards {
            let (response_tx, response_rx) = flume::bounded(1);
            let _ = shard.tx.send(ShardTask::FindMatches {
                content_hashes: Arc::clone(&content_hashes),
                early_exit,
                response_tx,
            });
            receivers.push(response_rx);
        }

        // Gather: collect partial results and merge.
        // Each worker exists in exactly one shard — no overlap in worker IDs.
        let mut merged = OverlapScores::default();
        for rx in receivers {
            if let Ok(partial) = rx.recv() {
                merged.scores.extend(partial.scores);
                merged.tree_sizes.extend(partial.tree_sizes);
            }
        }

        merged
    }

    /// Get total number of blocks across all workers.
    pub fn current_size(&self) -> usize {
        let n = self.next_worker_id.load(Ordering::Relaxed) as usize;
        self.tree_sizes[..n]
            .iter()
            .map(|size| size.load(Ordering::Relaxed))
            .sum()
    }

    /// Get the number of shards.
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }

    /// Look up the shard index for a worker_id.
    #[inline]
    fn shard_for_worker(&self, worker_id: u32) -> usize {
        self.worker_to_shard
            .get(&worker_id)
            .map(|entry| *entry.value())
            .unwrap_or(0)
    }

    /// Shut down all shard threads.
    fn shutdown(&mut self) {
        for shard in &self.shards {
            let _ = shard.tx.send(ShardTask::Shutdown);
        }
        for shard in &mut self.shards {
            if let Some(handle) = shard.join_handle.take() {
                let _ = handle.join();
            }
        }
    }
}

impl Drop for ShardedIndexer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl std::fmt::Debug for ShardedIndexer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardedIndexer")
            .field("num_shards", &self.num_shards)
            .field("jump_size", &self.jump_size)
            .field("workers", &self.next_worker_id.load(Ordering::Relaxed))
            .field("current_size", &self.current_size())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Shard thread loop
// ---------------------------------------------------------------------------

/// Per-shard state: maintains the local↔global worker ID mapping.
struct ShardState {
    indexer: PositionalIndexer,
    /// Per-worker reverse lookup maps, keyed by global worker ID.
    worker_blocks: HashMap<u32, WorkerBlockMap>,
    /// Global worker ID → local worker ID within this shard's PositionalIndexer.
    global_to_local: HashMap<u32, u32>,
    /// Local worker ID → global worker ID (for remapping find_matches results).
    local_to_global: Vec<u32>,
}

impl ShardState {
    fn new(jump_size: usize) -> Self {
        Self {
            indexer: PositionalIndexer::new(jump_size),
            worker_blocks: HashMap::new(),
            global_to_local: HashMap::new(),
            local_to_global: Vec::new(),
        }
    }

    /// Ensure a global worker ID is interned in this shard's PositionalIndexer.
    /// Returns the local worker ID.
    fn ensure_interned(&mut self, global_id: u32) -> u32 {
        if let Some(&local_id) = self.global_to_local.get(&global_id) {
            return local_id;
        }
        // Intern with a synthetic URL. The local ID is assigned sequentially.
        let local_id = self.indexer.intern_worker(&format!("w{global_id}"));
        self.global_to_local.insert(global_id, local_id);
        // local_to_global is indexed by local_id — it should grow sequentially.
        assert_eq!(
            local_id as usize,
            self.local_to_global.len(),
            "local IDs must be sequential"
        );
        self.local_to_global.push(global_id);
        local_id
    }

    /// Remap OverlapScores from local worker IDs to global worker IDs.
    fn remap_scores(&self, local_scores: OverlapScores) -> OverlapScores {
        let mut remapped = OverlapScores::default();
        for (local_id, score) in local_scores.scores {
            if let Some(&global_id) = self.local_to_global.get(local_id as usize) {
                remapped.scores.insert(global_id, score);
            }
        }
        for (local_id, size) in local_scores.tree_sizes {
            if let Some(&global_id) = self.local_to_global.get(local_id as usize) {
                remapped.tree_sizes.insert(global_id, size);
            }
        }
        remapped
    }
}

/// Main loop for a shard thread. Owns a private PositionalIndexer and per-worker
/// WorkerBlockMaps. Processes tasks from the channel until Shutdown.
fn shard_loop(rx: flume::Receiver<ShardTask>, jump_size: usize, tree_sizes: &[AtomicUsize]) {
    let mut state = ShardState::new(jump_size);

    while let Ok(task) = rx.recv() {
        match task {
            ShardTask::ApplyStored {
                worker_id,
                blocks,
                parent_seq_hash,
                response_tx,
            } => {
                let local_id = state.ensure_interned(worker_id);
                let wb = state.worker_blocks.entry(worker_id).or_default();
                let result = state
                    .indexer
                    .apply_stored(local_id, &blocks, parent_seq_hash, wb);

                // Update global tree_sizes.
                if result.is_ok() {
                    tree_sizes[worker_id as usize]
                        .store(state.indexer.worker_tree_size(local_id), Ordering::Relaxed);
                }

                let _ = response_tx.send(result);
            }
            ShardTask::ApplyRemoved {
                worker_id,
                seq_hashes,
            } => {
                let local_id = state.ensure_interned(worker_id);
                let wb = state.worker_blocks.entry(worker_id).or_default();
                state.indexer.apply_removed(local_id, &seq_hashes, wb);

                tree_sizes[worker_id as usize]
                    .store(state.indexer.worker_tree_size(local_id), Ordering::Relaxed);
            }
            ShardTask::ApplyCleared { worker_id } => {
                let local_id = state.ensure_interned(worker_id);
                let wb = state.worker_blocks.entry(worker_id).or_default();
                state.indexer.apply_cleared(local_id, wb);

                tree_sizes[worker_id as usize].store(0, Ordering::Relaxed);
            }
            ShardTask::RemoveWorker { worker_id } => {
                let local_id = state.ensure_interned(worker_id);
                if let Some(wb) = state.worker_blocks.remove(&worker_id) {
                    state.indexer.remove_worker(local_id, wb);
                }
                tree_sizes[worker_id as usize].store(0, Ordering::Relaxed);
            }
            ShardTask::FindMatches {
                content_hashes,
                early_exit,
                response_tx,
            } => {
                let local_scores = state.indexer.find_matches(&content_hashes, early_exit);
                let global_scores = state.remap_scores(local_scores);
                let _ = response_tx.send(global_scores);
            }
            ShardTask::Shutdown => break,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::event_tree::{compute_content_hash, compute_request_content_hashes};

    fn make_blocks(token_seqs: &[&[u32]]) -> Vec<StoredBlock> {
        token_seqs
            .iter()
            .enumerate()
            .map(|(i, tokens)| StoredBlock {
                seq_hash: SequenceHash(i as u64 + 1),
                content_hash: compute_content_hash(tokens),
            })
            .collect()
    }

    #[test]
    fn test_basic_store_and_match() {
        let indexer = ShardedIndexer::new(64, 2);
        let w1 = indexer.intern_worker("http://w1:8000");

        let blocks = make_blocks(&[&[1, 2, 3, 4], &[5, 6, 7, 8]]);
        indexer.apply_stored(w1, &blocks, None).unwrap();
        assert_eq!(indexer.current_size(), 2);

        let content_hashes = compute_request_content_hashes(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
        let scores = indexer.find_matches(&content_hashes, false);

        // w1 should have score > 0, keyed by global ID.
        assert!(scores.scores.contains_key(&w1), "w1 should have a score");
        assert!(*scores.scores.get(&w1).unwrap() > 0);
    }

    #[test]
    fn test_multiple_workers_different_shards() {
        let indexer = ShardedIndexer::new(64, 2);
        let w1 = indexer.intern_worker("http://w1:8000");
        let w2 = indexer.intern_worker("http://w2:8000");

        // Workers should be assigned to different shards (round-robin).
        let s1 = indexer.shard_for_worker(w1);
        let s2 = indexer.shard_for_worker(w2);
        assert_ne!(
            s1, s2,
            "two workers should be on different shards with 2 shards"
        );

        let blocks1 = make_blocks(&[&[1, 2, 3, 4]]);
        let blocks2 = make_blocks(&[&[1, 2, 3, 4], &[5, 6, 7, 8]]);

        indexer.apply_stored(w1, &blocks1, None).unwrap();
        indexer.apply_stored(w2, &blocks2, None).unwrap();
        assert_eq!(indexer.current_size(), 3);

        let content_hashes = compute_request_content_hashes(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
        let scores = indexer.find_matches(&content_hashes, false);

        // Both workers should appear in scores with global IDs.
        assert!(scores.scores.contains_key(&w1), "w1 should have a score");
        assert!(scores.scores.contains_key(&w2), "w2 should have a score");
        // w2 has 2 blocks, w1 has 1 — w2 should have higher score.
        assert!(scores.scores[&w2] >= scores.scores[&w1]);
    }

    #[test]
    fn test_apply_removed() {
        let indexer = ShardedIndexer::new(64, 1);
        let w1 = indexer.intern_worker("http://w1:8000");

        let blocks = make_blocks(&[&[1, 2, 3, 4], &[5, 6, 7, 8]]);
        indexer.apply_stored(w1, &blocks, None).unwrap();
        assert_eq!(indexer.current_size(), 2);

        // Remove the second block.
        indexer.apply_removed(w1, &[SequenceHash(2)]);
        // Fire-and-forget: wait for processing.
        thread::sleep(Duration::from_millis(10));
        assert_eq!(indexer.current_size(), 1);
    }

    #[test]
    fn test_apply_cleared() {
        let indexer = ShardedIndexer::new(64, 1);
        let w1 = indexer.intern_worker("http://w1:8000");

        let blocks = make_blocks(&[&[1, 2, 3, 4], &[5, 6, 7, 8]]);
        indexer.apply_stored(w1, &blocks, None).unwrap();
        assert_eq!(indexer.current_size(), 2);

        indexer.apply_cleared(w1);
        thread::sleep(Duration::from_millis(10));
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_remove_worker() {
        let indexer = ShardedIndexer::new(64, 1);
        let w1 = indexer.intern_worker("http://w1:8000");

        let blocks = make_blocks(&[&[1, 2, 3, 4]]);
        indexer.apply_stored(w1, &blocks, None).unwrap();
        assert_eq!(indexer.current_size(), 1);

        indexer.remove_worker(w1);
        thread::sleep(Duration::from_millis(10));
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_find_matches_empty() {
        let indexer = ShardedIndexer::new(64, 2);
        let scores = indexer.find_matches(&[], false);
        assert!(scores.scores.is_empty());
    }

    #[test]
    fn test_apply_stored_empty_blocks() {
        let indexer = ShardedIndexer::new(64, 1);
        let w1 = indexer.intern_worker("http://w1:8000");
        let result = indexer.apply_stored(w1, &[], None);
        assert!(result.is_ok());
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_worker_id_lookup() {
        let indexer = ShardedIndexer::new(64, 2);
        assert!(indexer.worker_id("http://w1:8000").is_none());

        let w1 = indexer.intern_worker("http://w1:8000");
        assert_eq!(indexer.worker_id("http://w1:8000"), Some(w1));
    }

    #[test]
    fn test_intern_worker_idempotent() {
        let indexer = ShardedIndexer::new(64, 2);
        let w1a = indexer.intern_worker("http://w1:8000");
        let w1b = indexer.intern_worker("http://w1:8000");
        assert_eq!(w1a, w1b);
    }

    #[test]
    fn test_tree_sizes_in_scores() {
        let indexer = ShardedIndexer::new(64, 2);
        let w1 = indexer.intern_worker("http://w1:8000");
        let w2 = indexer.intern_worker("http://w2:8000");

        let blocks1 = make_blocks(&[&[1, 2, 3, 4], &[5, 6, 7, 8], &[9, 10, 11, 12]]);
        let blocks2 = make_blocks(&[&[1, 2, 3, 4]]);

        indexer.apply_stored(w1, &blocks1, None).unwrap();
        indexer.apply_stored(w2, &blocks2, None).unwrap();

        let content_hashes = compute_request_content_hashes(&[1, 2, 3, 4], 4);
        let scores = indexer.find_matches(&content_hashes, false);

        // tree_sizes should reflect the total blocks per worker.
        assert_eq!(scores.tree_sizes.get(&w1).copied().unwrap_or(0), 3);
        assert_eq!(scores.tree_sizes.get(&w2).copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_apply_stored_with_parent() {
        let indexer = ShardedIndexer::new(64, 1);
        let w1 = indexer.intern_worker("http://w1:8000");

        // Store first block (no parent).
        let blocks1 = make_blocks(&[&[1, 2, 3, 4]]);
        indexer.apply_stored(w1, &blocks1, None).unwrap();

        // Store second block with parent.
        let blocks2 = vec![StoredBlock {
            seq_hash: SequenceHash(10),
            content_hash: compute_content_hash(&[5, 6, 7, 8]),
        }];
        indexer
            .apply_stored(w1, &blocks2, Some(SequenceHash(1)))
            .unwrap();
        assert_eq!(indexer.current_size(), 2);
    }

    #[test]
    fn test_apply_stored_parent_not_found() {
        let indexer = ShardedIndexer::new(64, 1);
        let w1 = indexer.intern_worker("http://w1:8000");

        // Store a block first so the worker is tracked.
        let blocks1 = make_blocks(&[&[1, 2, 3, 4]]);
        indexer.apply_stored(w1, &blocks1, None).unwrap();

        // Try to store with a non-existent parent.
        let blocks2 = make_blocks(&[&[5, 6, 7, 8]]);
        let result = indexer.apply_stored(w1, &blocks2, Some(SequenceHash(999)));
        assert!(result.is_err());
    }

    #[test]
    fn test_many_workers_many_shards() {
        let num_shards = 4;
        let indexer = ShardedIndexer::new(64, num_shards);
        let mut worker_ids = Vec::new();

        for i in 0..8 {
            let wid = indexer.intern_worker(&format!("http://w{i}:8000"));
            worker_ids.push(wid);

            let blocks = make_blocks(&[&[1, 2, 3, 4]]);
            indexer.apply_stored(wid, &blocks, None).unwrap();
        }

        assert_eq!(indexer.current_size(), 8);

        // All 8 workers should be found in find_matches.
        let content_hashes = compute_request_content_hashes(&[1, 2, 3, 4], 4);
        let scores = indexer.find_matches(&content_hashes, false);
        assert_eq!(scores.scores.len(), 8);

        for &wid in &worker_ids {
            assert!(
                scores.scores.contains_key(&wid),
                "worker {wid} should have a score"
            );
        }
    }
}
