//! Positional indexer for cache-aware routing.
//!
//! Uses `DashMap<(position, ContentHash), SeqEntry>` for O(1) random access to any
//! depth position, replacing tree pointer-chasing with direct positional lookup.
//! Jump search skips positions in strides, yielding amortized O(D/J + W) complexity.
//!
//! **Dual-hash scheme**: backends send a position-aware `block_hash` (SequenceHash)
//! and raw `token_ids` per block. The router computes a position-independent
//! ContentHash (XXH3) from token_ids, then a rolling prefix hash (also XXH3) from
//! the ContentHash sequence. SeqEntry is keyed by the router's prefix hash for
//! precise disambiguation at query time. The backend's SequenceHash is stored in
//! worker_blocks only, used for `apply_removed` reverse lookup.
//!
//! Thread safety: all methods are `&self` and internally synchronized via DashMap +
//! parking_lot::RwLock. Reads (find_matches) and writes (apply_stored/removed/cleared)
//! can proceed concurrently.

use std::{fmt, sync::Arc};

use dashmap::{mapref::entry::Entry, DashMap};
use parking_lot::RwLock;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

/// Seed for XXH3 hashing.
pub const XXH3_SEED: u64 = 1337;

/// Position-independent content hash of tokens within a single block.
/// Computed via XXH3-64 from token IDs. Same tokens always produce the same hash
/// regardless of their position in the sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ContentHash(pub u64);

/// Position-aware block hash from backend (sequence hash).
/// Matches the `block_hash` field in KvBlock proto (i64, bitwise reinterpreted as u64).
/// Different from ContentHash because it encodes the full prefix history.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct SequenceHash(pub u64);

impl From<i64> for SequenceHash {
    fn from(value: i64) -> Self {
        Self(value as u64)
    }
}

impl From<u64> for SequenceHash {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

/// Worker identifier (URL string in SMG).
/// Uses `Arc<str>` for cheap cloning on hot paths (same pattern as `TenantId`).
pub type WorkerId = Arc<str>;

/// A block from a store event, carrying both hash representations.
#[derive(Debug, Clone, Copy)]
pub struct StoredBlock {
    /// Position-aware hash from the backend proto (`block_hash` field).
    pub seq_hash: SequenceHash,
    /// Position-independent hash computed from token IDs via XXH3.
    pub content_hash: ContentHash,
}

/// Error returned by [`PositionalIndexer::apply_stored`] when the event cannot be applied.
#[derive(Debug)]
pub enum ApplyError {
    /// Worker has no entries in the index — cannot resolve parent block.
    WorkerNotTracked,
    /// The specified `parent_seq_hash` was not found in this worker's reverse lookup.
    ParentBlockNotFound,
}

impl fmt::Display for ApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkerNotTracked => write!(f, "worker not tracked in index"),
            Self::ParentBlockNotFound => write!(f, "parent block hash not found for worker"),
        }
    }
}

impl std::error::Error for ApplyError {}

/// Overlap scores: how many consecutive blocks each worker has cached.
#[derive(Debug, Default)]
pub struct OverlapScores {
    /// worker_url → number of matching prefix blocks (depth in indexer)
    pub scores: FxHashMap<WorkerId, u32>,
    /// worker_url → total blocks cached by this worker
    pub tree_sizes: FxHashMap<WorkerId, usize>,
}

/// Compute content hash from token IDs (position-independent).
/// Uses XXH3-64 streaming hasher with standard seed — avoids intermediate allocation.
pub fn compute_content_hash(token_ids: &[u32]) -> ContentHash {
    use std::hash::Hasher;
    let mut hasher = xxhash_rust::xxh3::Xxh3::with_seed(XXH3_SEED);
    for &t in token_ids {
        hasher.write(&t.to_le_bytes());
    }
    ContentHash(hasher.finish())
}

/// Chunk request tokens by block size and compute a [`ContentHash`] per full block.
///
/// This is the entry point for the **query path**: given a request's token IDs and
/// the backend's block size, produce the content-hash sequence that
/// [`PositionalIndexer::find_matches`] expects.
///
/// Partial trailing chunks (fewer tokens than `block_size`) are discarded because
/// backends only cache full blocks.
///
/// Returns an empty `Vec` if `block_size` is 0.
pub fn compute_request_content_hashes(tokens: &[u32], block_size: usize) -> Vec<ContentHash> {
    if block_size == 0 {
        tracing::warn!("compute_request_content_hashes called with block_size=0, returning empty");
        return Vec::new();
    }
    tokens
        .chunks(block_size)
        .filter(|chunk| chunk.len() == block_size)
        .map(compute_content_hash)
        .collect()
}

// ---------------------------------------------------------------------------
// SeqEntry: optimizes for the common case (one seq_hash per position+content)
// ---------------------------------------------------------------------------

/// Entry for the innermost level of the index.
///
/// Optimizes for the common case where there's only one sequence hash
/// at a given (position, content_hash) pair, avoiding HashMap allocation.
#[derive(Debug, Clone)]
enum SeqEntry {
    /// Single seq_hash → workers mapping (common case, no HashMap allocation).
    Single(SequenceHash, FxHashSet<WorkerId>),
    /// Multiple seq_hash → workers mappings (rare: different prefixes with same content).
    Multi(FxHashMap<SequenceHash, FxHashSet<WorkerId>>),
}

impl SeqEntry {
    fn new(seq_hash: SequenceHash, worker: &str) -> Self {
        let mut workers = FxHashSet::default();
        workers.insert(Arc::from(worker));
        Self::Single(seq_hash, workers)
    }

    /// Insert a worker for a given seq_hash, upgrading to Multi if needed.
    fn insert(&mut self, seq_hash: SequenceHash, worker: &str) {
        let worker_id: WorkerId = Arc::from(worker);
        match self {
            Self::Single(existing_hash, workers) if *existing_hash == seq_hash => {
                workers.insert(worker_id);
            }
            Self::Single(existing_hash, existing_workers) => {
                let mut map = FxHashMap::with_capacity_and_hasher(2, FxBuildHasher);
                map.insert(*existing_hash, std::mem::take(existing_workers));
                map.entry(seq_hash).or_default().insert(worker_id);
                *self = Self::Multi(map);
            }
            Self::Multi(map) => {
                map.entry(seq_hash).or_default().insert(worker_id);
            }
        }
    }

    /// Remove a worker from a given seq_hash.
    /// Returns true if the entry is now completely empty and should be removed.
    fn remove(&mut self, seq_hash: SequenceHash, worker: &str) -> bool {
        match self {
            Self::Single(existing_hash, workers) if *existing_hash == seq_hash => {
                workers.remove(worker);
                workers.is_empty()
            }
            Self::Single(_, _) => false,
            Self::Multi(map) => {
                if let Some(workers) = map.get_mut(&seq_hash) {
                    workers.remove(worker);
                    if workers.is_empty() {
                        map.remove(&seq_hash);
                    }
                }
                map.is_empty()
            }
        }
    }

    /// Get workers for a specific prefix hash (used in query path and event processing).
    fn get(&self, seq_hash: SequenceHash) -> Option<&FxHashSet<WorkerId>> {
        match self {
            Self::Single(existing_hash, workers) if *existing_hash == seq_hash => Some(workers),
            Self::Single(_, _) => None,
            Self::Multi(map) => map.get(&seq_hash),
        }
    }
}

// ---------------------------------------------------------------------------
// PositionalIndexer
// ---------------------------------------------------------------------------

/// Per-worker reverse lookup: backend_seq_hash → (position, content_hash, prefix_hash).
/// The `prefix_hash` is the router-computed rolling hash used as the SeqEntry key.
type LevelIndex = RwLock<FxHashMap<SequenceHash, (usize, ContentHash, SequenceHash)>>;

/// Positional indexer for cache-aware routing.
///
/// Uses `DashMap<(position, ContentHash), SeqEntry>`
/// for O(1) position access and jump search for O(D/J + W) matching complexity.
///
/// All methods take `&self` — concurrency is handled internally via DashMap sharding
/// and parking_lot::RwLock.
pub struct PositionalIndexer {
    /// Primary index: (position, content_hash) → SeqEntry.
    index: DashMap<(usize, ContentHash), SeqEntry, FxBuildHasher>,
    /// Per-worker reverse lookup: worker → { seq_hash → (position, content_hash) }.
    /// Single RwLock because structural mutations (add/remove workers) are rare;
    /// the hot path is read-only.
    worker_blocks: RwLock<FxHashMap<WorkerId, LevelIndex>>,
    /// Jump size for search optimization (default 64).
    jump_size: usize,
}

impl PositionalIndexer {
    /// Create a new PositionalIndexer with the given jump size.
    ///
    /// `jump_size` controls how many positions the search algorithm skips at a time.
    /// Larger values reduce lookups on long matching prefixes but increase scan range
    /// when workers drain. Default: 64.
    pub fn new(jump_size: usize) -> Self {
        assert!(jump_size > 0, "jump_size must be greater than 0");
        Self {
            index: DashMap::with_hasher(FxBuildHasher),
            worker_blocks: RwLock::new(FxHashMap::default()),
            jump_size,
        }
    }

    /// Apply a "blocks stored" event for a worker.
    ///
    /// `blocks`: ordered sequence of stored blocks (each with seq_hash + content_hash).
    /// `parent_seq_hash`: if Some, the sequence extends from the parent's position + 1.
    ///   If None, the sequence starts from position 0.
    pub fn apply_stored(
        &self,
        worker: &str,
        blocks: &[StoredBlock],
        parent_seq_hash: Option<SequenceHash>,
    ) -> Result<(), ApplyError> {
        if blocks.is_empty() {
            return Ok(());
        }

        let worker_id: WorkerId = Arc::from(worker);

        // Determine starting position and parent's router prefix hash.
        let (start_pos, parent_prefix) = match parent_seq_hash {
            Some(parent_hash) => {
                let wb = self.worker_blocks.read();
                let Some(level_index) = wb.get(&worker_id) else {
                    return Err(ApplyError::WorkerNotTracked);
                };
                let worker_map = level_index.read();
                let Some(&(parent_pos, _, parent_pfx)) = worker_map.get(&parent_hash) else {
                    return Err(ApplyError::ParentBlockNotFound);
                };
                (parent_pos + 1, Some(parent_pfx))
            }
            None => (0, None),
        };

        if !self.worker_blocks.read().contains_key(&worker_id) {
            self.worker_blocks
                .write()
                .entry(worker_id.clone())
                .or_insert_with(|| RwLock::new(FxHashMap::default()));
        }

        let wb = self.worker_blocks.read();
        let Some(level_index) = wb.get(&worker_id) else {
            return Ok(());
        };
        let mut worker_map = level_index.write();

        let mut prev_prefix = parent_prefix;
        for (i, block) in blocks.iter().enumerate() {
            let position = start_pos + i;
            let content_hash = block.content_hash;

            // Compute router prefix hash (rolling XXH3 of content hashes).
            // This is the SeqEntry key — consistent between store and query paths.
            let prefix_hash = match prev_prefix {
                Some(prev) => SequenceHash(Self::compute_next_seq_hash(prev.0, content_hash.0)),
                // Position 0: prefix_hash == content_hash (no parent to chain from).
                None => SequenceHash(content_hash.0),
            };

            self.index
                .entry((position, content_hash))
                .and_modify(|entry| entry.insert(prefix_hash, worker))
                .or_insert_with(|| SeqEntry::new(prefix_hash, worker));

            worker_map.insert(block.seq_hash, (position, content_hash, prefix_hash));
            prev_prefix = Some(prefix_hash);
        }

        Ok(())
    }

    /// Apply a "blocks removed" event for a worker.
    ///
    /// `seq_hashes`: position-aware block hashes to remove (from proto).
    ///
    /// **Note on orphaned entries**: Removing a block at position N does not cascade to
    /// blocks at positions > N. Those entries become orphaned — they remain in the index
    /// but won't match queries because the rolling prefix hash chain is broken at the gap.
    /// This is harmless: orphaned entries waste a small amount of memory and are cleaned up
    /// when the worker is cleared or removed. Backends typically evict from the tail (LRU),
    /// so mid-sequence gaps are rare in practice.
    pub fn apply_removed(&self, worker: &str, seq_hashes: &[SequenceHash]) {
        let worker_id: WorkerId = Arc::from(worker);

        let wb = self.worker_blocks.read();
        let Some(level_index) = wb.get(&worker_id) else {
            tracing::debug!(
                worker = %worker,
                num_hashes = seq_hashes.len(),
                "apply_removed: worker not tracked, ignoring"
            );
            return;
        };

        let mut worker_map = level_index.write();

        for &seq_hash in seq_hashes {
            let Some((position, content_hash, prefix_hash)) = worker_map.remove(&seq_hash) else {
                continue;
            };

            if let Entry::Occupied(mut occupied) = self.index.entry((position, content_hash)) {
                if occupied.get_mut().remove(prefix_hash, worker) {
                    occupied.remove();
                }
            }
        }
    }

    /// Apply a "cache cleared" event — remove all blocks for this worker but keep tracked.
    pub fn apply_cleared(&self, worker: &str) {
        self.remove_or_clear_worker(worker, true);
    }

    /// Remove a worker entirely (called when worker is removed from registry).
    pub fn remove_worker(&self, worker: &str) {
        self.remove_or_clear_worker(worker, false);
    }

    /// Get total number of blocks across all workers.
    pub fn current_size(&self) -> usize {
        self.worker_blocks
            .read()
            .values()
            .map(|level_index| level_index.read().len())
            .sum()
    }

    /// Find overlap scores for a request's content hash sequence.
    ///
    /// Uses jump search: strides by `jump_size` positions, only scanning
    /// intermediate positions when workers drain (stop matching).
    /// Complexity: amortized O(D/J + W) where D=depth, J=jump_size, W=workers.
    ///
    /// **Assumption**: Block sequences are prefix-closed — if a worker has a block at
    /// position N, it has blocks at all positions 0..N. This holds when backends evict
    /// from the tail (LRU). If `apply_removed` creates a mid-sequence gap, the rolling
    /// prefix hash detects it (the chain breaks at the gap), but the jump heuristic may
    /// over-count if it lands past the gap. In practice, backends only evict tail blocks.
    pub fn find_matches(&self, content_hashes: &[ContentHash]) -> OverlapScores {
        self.jump_search_matches(content_hashes)
    }

    // -----------------------------------------------------------------------
    // Internal: remove/clear worker
    // -----------------------------------------------------------------------

    fn remove_or_clear_worker(&self, worker: &str, keep_worker: bool) {
        let worker_id: WorkerId = Arc::from(worker);

        let mut wb = self.worker_blocks.write();
        if let Some(level_index) = wb.remove(&worker_id) {
            let worker_map = level_index.read();
            for (_, &(position, content_hash, prefix_hash)) in worker_map.iter() {
                if let Entry::Occupied(mut occupied) = self.index.entry((position, content_hash)) {
                    if occupied.get_mut().remove(prefix_hash, worker) {
                        occupied.remove();
                    }
                }
            }
        }

        if keep_worker {
            wb.insert(worker_id, RwLock::new(FxHashMap::default()));
        }
    }

    // -----------------------------------------------------------------------
    // Internal: router prefix hash + jump search
    //
    // The router computes its own rolling hash from ContentHashes (XXH3).
    // This hash is stored in SeqEntry during apply_stored and recomputed
    // at query time for precise filtering — matching Dynamo's semantics.
    // The backend's SequenceHash (from proto block_hash) stays in
    // worker_blocks only, used for apply_removed reverse lookup.
    // -----------------------------------------------------------------------

    /// Compute rolling prefix hash: XXH3(prev || current).
    #[inline]
    fn compute_next_seq_hash(prev_seq_hash: u64, current_content_hash: u64) -> u64 {
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&prev_seq_hash.to_le_bytes());
        bytes[8..].copy_from_slice(&current_content_hash.to_le_bytes());
        xxhash_rust::xxh3::xxh3_64_with_seed(&bytes, XXH3_SEED)
    }

    /// Lazily compute prefix hashes up to `target_pos`.
    #[inline]
    fn ensure_seq_hash_computed(
        seq_hashes: &mut Vec<SequenceHash>,
        target_pos: usize,
        sequence: &[ContentHash],
    ) {
        while seq_hashes.len() <= target_pos {
            let pos = seq_hashes.len();
            if pos == 0 {
                seq_hashes.push(SequenceHash(sequence[0].0));
            } else {
                let prev = seq_hashes[pos - 1].0;
                let current = sequence[pos].0;
                seq_hashes.push(SequenceHash(Self::compute_next_seq_hash(prev, current)));
            }
        }
    }

    /// Get workers at a position matching both content_hash and prefix_hash.
    fn get_workers_lazy(
        &self,
        position: usize,
        content_hash: ContentHash,
        seq_hashes: &mut Vec<SequenceHash>,
        sequence: &[ContentHash],
    ) -> Option<FxHashSet<WorkerId>> {
        let entry = self.index.get(&(position, content_hash))?;
        Self::ensure_seq_hash_computed(seq_hashes, position, sequence);
        entry.get(seq_hashes[position]).cloned()
    }

    /// Count workers at a position matching the prefix_hash.
    fn count_workers_at(
        &self,
        position: usize,
        content_hash: ContentHash,
        seq_hashes: &mut Vec<SequenceHash>,
        sequence: &[ContentHash],
    ) -> usize {
        let Some(entry) = self.index.get(&(position, content_hash)) else {
            return 0;
        };
        Self::ensure_seq_hash_computed(seq_hashes, position, sequence);
        entry
            .get(seq_hashes[position])
            .map(|workers| workers.len())
            .unwrap_or(0)
    }

    /// Scan positions sequentially, filtering by prefix_hash.
    fn linear_scan_drain(
        &self,
        sequence: &[ContentHash],
        seq_hashes: &mut Vec<SequenceHash>,
        active: &mut FxHashSet<WorkerId>,
        scores: &mut OverlapScores,
        lo: usize,
        hi: usize,
    ) {
        for (offset, &content_hash) in sequence[lo..hi].iter().enumerate() {
            if active.is_empty() {
                break;
            }
            let pos = lo + offset;

            let workers = self.index.get(&(pos, content_hash)).and_then(|entry| {
                Self::ensure_seq_hash_computed(seq_hashes, pos, sequence);
                entry.get(seq_hashes[pos]).cloned()
            });

            let Some(workers) = workers else {
                for worker in active.drain() {
                    scores.scores.insert(worker, pos as u32);
                }
                break;
            };

            active.retain(|w| {
                if workers.contains(w) {
                    true
                } else {
                    scores.scores.insert(w.clone(), pos as u32);
                    false
                }
            });
        }
    }

    fn jump_search_matches(&self, content_hashes: &[ContentHash]) -> OverlapScores {
        let mut scores = OverlapScores::default();

        if content_hashes.is_empty() {
            return scores;
        }

        let mut seq_hashes = Vec::with_capacity(content_hashes.len());

        let Some(initial_workers) =
            self.get_workers_lazy(0, content_hashes[0], &mut seq_hashes, content_hashes)
        else {
            return scores;
        };

        let mut active = initial_workers;
        if active.is_empty() {
            return scores;
        }

        let len = content_hashes.len();
        let mut current_pos = 0;

        while current_pos < len - 1 && !active.is_empty() {
            let next_pos = (current_pos + self.jump_size).min(len - 1);

            let count = self.count_workers_at(
                next_pos,
                content_hashes[next_pos],
                &mut seq_hashes,
                content_hashes,
            );

            // If the worker count at the jump destination matches the active set size,
            // all active workers are still present — safe to skip intermediate positions.
            if count == active.len() {
                current_pos = next_pos;
            } else {
                self.linear_scan_drain(
                    content_hashes,
                    &mut seq_hashes,
                    &mut active,
                    &mut scores,
                    current_pos + 1,
                    next_pos + 1,
                );
                current_pos = next_pos;
            }
        }

        let final_score = len as u32;
        for worker in active {
            scores.scores.insert(worker, final_score);
        }

        let wb = self.worker_blocks.read();
        for worker in scores.scores.keys() {
            if let Some(level_index) = wb.get(worker) {
                scores
                    .tree_sizes
                    .insert(Arc::clone(worker), level_index.read().len());
            }
        }

        scores
    }
}

impl Default for PositionalIndexer {
    fn default() -> Self {
        Self::new(64)
    }
}

impl fmt::Debug for PositionalIndexer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PositionalIndexer")
            .field("index_size", &self.index.len())
            .field("jump_size", &self.jump_size)
            .field(
                "workers",
                &self.worker_blocks.read().keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a sequence of StoredBlocks with distinct seq_hashes and content_hashes.
    fn make_blocks(content_hashes: &[u64]) -> Vec<StoredBlock> {
        // Generate seq_hashes as rolling hashes of content
        let mut blocks = Vec::new();
        let mut prev_seq: u64 = 0;
        for (i, &ch) in content_hashes.iter().enumerate() {
            let seq = if i == 0 {
                ch
            } else {
                PositionalIndexer::compute_next_seq_hash(prev_seq, ch)
            };
            prev_seq = seq;
            blocks.push(StoredBlock {
                seq_hash: SequenceHash(seq),
                content_hash: ContentHash(ch),
            });
        }
        blocks
    }

    /// Helper: create ContentHash sequence for find_matches.
    fn hashes(values: &[u64]) -> Vec<ContentHash> {
        values.iter().map(|&v| ContentHash(v)).collect()
    }

    #[test]
    fn test_new_indexer_is_empty() {
        let indexer = PositionalIndexer::default();
        let scores = indexer.find_matches(&hashes(&[1, 2, 3]));
        assert!(scores.scores.is_empty());
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_store_and_find_single_worker() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[10, 20, 30]);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&[10, 20, 30]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&3));
        assert_eq!(scores.tree_sizes.get("http://w1:8000"), Some(&3));
    }

    #[test]
    fn test_store_partial_prefix_match() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[10, 20, 30]);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        // Request has longer sequence — only first 3 match
        let scores = indexer.find_matches(&hashes(&[10, 20, 30, 40, 50]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&3));
    }

    #[test]
    fn test_store_no_match() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[10, 20, 30]);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&[99, 88, 77]));
        assert!(scores.scores.is_empty());
    }

    #[test]
    fn test_two_workers_different_depths() {
        let indexer = PositionalIndexer::new(64);
        let blocks_w1 = make_blocks(&[10, 20, 30]);
        let blocks_w2 = make_blocks(&[10, 20]);
        indexer
            .apply_stored("http://w1:8000", &blocks_w1, None)
            .unwrap();
        indexer
            .apply_stored("http://w2:8000", &blocks_w2, None)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&[10, 20, 30, 40]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&3));
        assert_eq!(scores.scores.get("http://w2:8000"), Some(&2));
    }

    #[test]
    fn test_remove_blocks() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[10, 20, 30]);
        let seq_hash_of_30 = blocks[2].seq_hash;
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();
        indexer.apply_removed("http://w1:8000", &[seq_hash_of_30]);

        // After removing block at position 2, w1 should only match 2 blocks
        let scores = indexer.find_matches(&hashes(&[10, 20, 30]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&2));
        assert_eq!(scores.tree_sizes.get("http://w1:8000"), Some(&2));
    }

    #[test]
    fn test_clear_worker() {
        let indexer = PositionalIndexer::new(64);
        let blocks_w1 = make_blocks(&[10, 20, 30]);
        let blocks_w2 = make_blocks(&[10, 20]);
        indexer
            .apply_stored("http://w1:8000", &blocks_w1, None)
            .unwrap();
        indexer
            .apply_stored("http://w2:8000", &blocks_w2, None)
            .unwrap();

        indexer.apply_cleared("http://w1:8000");

        let scores = indexer.find_matches(&hashes(&[10, 20, 30]));
        assert!(!scores.scores.contains_key("http://w1:8000"));
        assert_eq!(scores.scores.get("http://w2:8000"), Some(&2));
    }

    #[test]
    fn test_tree_sizes() {
        let indexer = PositionalIndexer::new(64);
        let blocks_w1 = make_blocks(&[10, 20, 30]);
        let blocks_w2 = make_blocks(&[10, 20]);
        indexer
            .apply_stored("http://w1:8000", &blocks_w1, None)
            .unwrap();
        indexer
            .apply_stored("http://w2:8000", &blocks_w2, None)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&[10]));
        assert_eq!(scores.tree_sizes.get("http://w1:8000"), Some(&3));
        assert_eq!(scores.tree_sizes.get("http://w2:8000"), Some(&2));
    }

    #[test]
    fn test_store_with_parent_hash() {
        let indexer = PositionalIndexer::new(64);
        // First store: blocks at positions 0, 1
        let blocks1 = make_blocks(&[10, 20]);
        let parent_seq_hash = blocks1[1].seq_hash;
        indexer
            .apply_stored("http://w1:8000", &blocks1, None)
            .unwrap();

        // Second store: blocks at positions 2, 3 (extending from parent at position 1)
        // Need seq_hashes that chain from blocks1's last seq_hash
        let ch_30 = 30u64;
        let ch_40 = 40u64;
        let seq_30 = PositionalIndexer::compute_next_seq_hash(parent_seq_hash.0, ch_30);
        let seq_40 = PositionalIndexer::compute_next_seq_hash(seq_30, ch_40);
        let blocks2 = vec![
            StoredBlock {
                seq_hash: SequenceHash(seq_30),
                content_hash: ContentHash(ch_30),
            },
            StoredBlock {
                seq_hash: SequenceHash(seq_40),
                content_hash: ContentHash(ch_40),
            },
        ];
        indexer
            .apply_stored("http://w1:8000", &blocks2, Some(parent_seq_hash))
            .unwrap();

        let scores = indexer.find_matches(&hashes(&[10, 20, 30, 40]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&4));
    }

    #[test]
    fn test_remove_nonexistent_worker() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[10, 20]);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        // Removing blocks for a worker that doesn't own them is a no-op
        indexer.apply_removed("http://w2:8000", &[SequenceHash(999)]);

        let scores = indexer.find_matches(&hashes(&[10, 20]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&2));
    }

    #[test]
    fn test_jump_search_skips_positions() {
        // Use a small jump_size to test the jump behavior
        let indexer = PositionalIndexer::new(4);

        // Worker 1: 10 blocks
        let content: Vec<u64> = (100..110).collect();
        let blocks_w1 = make_blocks(&content);
        indexer
            .apply_stored("http://w1:8000", &blocks_w1, None)
            .unwrap();

        // Worker 2: only first 6 blocks
        let blocks_w2 = make_blocks(&content[..6]);
        indexer
            .apply_stored("http://w2:8000", &blocks_w2, None)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&content));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&10));
        assert_eq!(scores.scores.get("http://w2:8000"), Some(&6));
    }

    #[test]
    fn test_seq_entry_single_to_multi_upgrade() {
        let indexer = PositionalIndexer::new(64);

        // Two workers with DIFFERENT content at position 0 but SAME content at position 1.
        // This creates different router prefix hashes at position 1, triggering Multi upgrade.
        // W1: content [10, 20] → prefix [10, hash(10||20)]
        // W2: content [99, 20] → prefix [99, hash(99||20)]
        // At position 1, both have ContentHash(20) but different prefix hashes → Multi.
        let blocks_w1 = vec![
            StoredBlock {
                seq_hash: SequenceHash(1000),
                content_hash: ContentHash(10),
            },
            StoredBlock {
                seq_hash: SequenceHash(2000),
                content_hash: ContentHash(20),
            },
        ];
        let blocks_w2 = vec![
            StoredBlock {
                seq_hash: SequenceHash(3000),
                content_hash: ContentHash(99),
            },
            StoredBlock {
                seq_hash: SequenceHash(4000),
                content_hash: ContentHash(20),
            },
        ];

        indexer
            .apply_stored("http://w1:8000", &blocks_w1, None)
            .unwrap();
        indexer
            .apply_stored("http://w2:8000", &blocks_w2, None)
            .unwrap();

        // Position 1 has same content but different prefix histories → Multi
        {
            let entry = indexer.index.get(&(1, ContentHash(20))).unwrap();
            assert!(matches!(entry.value(), SeqEntry::Multi(_)));
        }

        // Same content at same position → Single (both share prefix hash)
        // Workers with identical content get identical router prefix hashes.
        let indexer2 = PositionalIndexer::new(64);
        let blocks_w3 = make_blocks(&[10, 20]);
        let blocks_w4 = vec![
            StoredBlock {
                seq_hash: SequenceHash(5000),
                content_hash: ContentHash(10),
            },
            StoredBlock {
                seq_hash: SequenceHash(6000),
                content_hash: ContentHash(20),
            },
        ];
        indexer2
            .apply_stored("http://w3:8000", &blocks_w3, None)
            .unwrap();
        indexer2
            .apply_stored("http://w4:8000", &blocks_w4, None)
            .unwrap();
        {
            let entry = indexer2.index.get(&(0, ContentHash(10))).unwrap();
            assert!(matches!(entry.value(), SeqEntry::Single(_, _)));
        }

        // Query: [10, 20] matches only w1 (w2 has content 99 at position 0)
        let scores = indexer.find_matches(&hashes(&[10, 20]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&2));
        assert!(!scores.scores.contains_key("http://w2:8000"));

        // Remove w1 via backend seq_hashes — reverse lookup maps to correct prefix hash
        indexer.apply_removed("http://w1:8000", &[SequenceHash(1000), SequenceHash(2000)]);
        let scores2 = indexer.find_matches(&hashes(&[10, 20]));
        assert!(!scores2.scores.contains_key("http://w1:8000"));
    }

    #[test]
    fn test_concurrent_find_matches() {
        let indexer = Arc::new(PositionalIndexer::new(64));
        let blocks = make_blocks(&[10, 20, 30]);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        let mut handles = Vec::new();
        for _ in 0..8 {
            let indexer = Arc::clone(&indexer);
            handles.push(std::thread::spawn(move || {
                let scores = indexer.find_matches(&hashes(&[10, 20, 30]));
                assert_eq!(scores.scores.get("http://w1:8000"), Some(&3));
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_compute_content_hash_different_tokens() {
        let hash1 = compute_content_hash(&[1, 2, 3, 4]);
        let hash2 = compute_content_hash(&[5, 6, 7, 8]);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_remove_worker_entirely() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[10, 20, 30]);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        indexer.remove_worker("http://w1:8000");

        let scores = indexer.find_matches(&hashes(&[10, 20, 30]));
        assert!(scores.scores.is_empty());
        assert_eq!(indexer.current_size(), 0);
    }

    // -----------------------------------------------------------------------
    // Additional comprehensive tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_store_is_noop() {
        let indexer = PositionalIndexer::default();
        indexer.apply_stored("http://w1:8000", &[], None).unwrap();
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_find_matches_empty_query() {
        let indexer = PositionalIndexer::default();
        let blocks = make_blocks(&[10, 20]);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        let scores = indexer.find_matches(&[]);
        assert!(scores.scores.is_empty());
    }

    #[test]
    fn test_store_with_parent_not_found() {
        let indexer = PositionalIndexer::default();
        let blocks = make_blocks(&[10, 20]);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        // Try to extend from a non-existent parent — should return ParentBlockNotFound
        let orphan_blocks = make_blocks(&[30, 40]);
        let err = indexer
            .apply_stored("http://w1:8000", &orphan_blocks, Some(SequenceHash(999999)))
            .unwrap_err();
        assert!(matches!(err, ApplyError::ParentBlockNotFound));

        // Should still only have the original 2 blocks
        assert_eq!(indexer.current_size(), 2);
    }

    #[test]
    fn test_store_with_parent_untracked_worker() {
        let indexer = PositionalIndexer::default();
        let orphan_blocks = make_blocks(&[10, 20]);

        // Worker not tracked — should return WorkerNotTracked
        let err = indexer
            .apply_stored("http://new:8000", &orphan_blocks, Some(SequenceHash(123)))
            .unwrap_err();
        assert!(matches!(err, ApplyError::WorkerNotTracked));
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_double_remove_same_block() {
        let indexer = PositionalIndexer::default();
        let blocks = make_blocks(&[10, 20, 30]);
        let seq_hash_30 = blocks[2].seq_hash;
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        indexer.apply_removed("http://w1:8000", &[seq_hash_30]);
        // Second remove should be a no-op (already gone)
        indexer.apply_removed("http://w1:8000", &[seq_hash_30]);

        assert_eq!(indexer.current_size(), 2);
        let scores = indexer.find_matches(&hashes(&[10, 20, 30]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&2));
    }

    #[test]
    fn test_store_after_clear() {
        let indexer = PositionalIndexer::default();
        let blocks = make_blocks(&[10, 20, 30]);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        indexer.apply_cleared("http://w1:8000");
        assert_eq!(indexer.current_size(), 0);

        // Re-store after clear
        let new_blocks = make_blocks(&[40, 50]);
        indexer
            .apply_stored("http://w1:8000", &new_blocks, None)
            .unwrap();

        assert_eq!(indexer.current_size(), 2);
        let scores = indexer.find_matches(&hashes(&[40, 50]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&2));
    }

    #[test]
    fn test_store_after_remove_worker() {
        let indexer = PositionalIndexer::default();
        let blocks = make_blocks(&[10, 20]);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        indexer.remove_worker("http://w1:8000");
        assert_eq!(indexer.current_size(), 0);

        // Re-store after full removal
        let new_blocks = make_blocks(&[30, 40]);
        indexer
            .apply_stored("http://w1:8000", &new_blocks, None)
            .unwrap();

        assert_eq!(indexer.current_size(), 2);
        let scores = indexer.find_matches(&hashes(&[30, 40]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&2));
    }

    #[test]
    fn test_overlapping_stores_same_worker() {
        // Worker stores [10, 20, 30], then stores again [10, 20] — positions overlap
        let indexer = PositionalIndexer::default();
        let blocks1 = make_blocks(&[10, 20, 30]);
        indexer
            .apply_stored("http://w1:8000", &blocks1, None)
            .unwrap();

        // Re-store shorter prefix — adds duplicate entries at same positions
        let blocks2 = make_blocks(&[10, 20]);
        indexer
            .apply_stored("http://w1:8000", &blocks2, None)
            .unwrap();

        // Should still match the full depth (3 blocks from first store)
        let scores = indexer.find_matches(&hashes(&[10, 20, 30]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&3));
    }

    #[test]
    fn test_many_workers_same_prefix() {
        let indexer = PositionalIndexer::default();
        let blocks = make_blocks(&[10, 20, 30]);

        for i in 0..10 {
            let worker = format!("http://w{i}:8000");
            indexer.apply_stored(&worker, &blocks, None).unwrap();
        }

        let scores = indexer.find_matches(&hashes(&[10, 20, 30]));
        assert_eq!(scores.scores.len(), 10);
        for i in 0..10 {
            let worker = format!("http://w{i}:8000");
            assert_eq!(scores.scores.get(worker.as_str()), Some(&3));
        }
    }

    #[test]
    fn test_jump_search_with_jump_size_1() {
        // jump_size=1 means linear scan every step (degenerate case)
        let indexer = PositionalIndexer::new(1);
        let content: Vec<u64> = (100..120).collect();
        let blocks_w1 = make_blocks(&content);
        let blocks_w2 = make_blocks(&content[..10]);
        indexer
            .apply_stored("http://w1:8000", &blocks_w1, None)
            .unwrap();
        indexer
            .apply_stored("http://w2:8000", &blocks_w2, None)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&content));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&20));
        assert_eq!(scores.scores.get("http://w2:8000"), Some(&10));
    }

    #[test]
    fn test_jump_search_workers_drain_at_different_positions() {
        // 3 workers with different depths, small jump_size to exercise linear_scan_drain
        let indexer = PositionalIndexer::new(3);
        let content: Vec<u64> = (1..=15).collect();

        let blocks_w1 = make_blocks(&content); // 15 blocks
        let blocks_w2 = make_blocks(&content[..7]); // 7 blocks
        let blocks_w3 = make_blocks(&content[..4]); // 4 blocks

        indexer
            .apply_stored("http://w1:8000", &blocks_w1, None)
            .unwrap();
        indexer
            .apply_stored("http://w2:8000", &blocks_w2, None)
            .unwrap();
        indexer
            .apply_stored("http://w3:8000", &blocks_w3, None)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&content));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&15));
        assert_eq!(scores.scores.get("http://w2:8000"), Some(&7));
        assert_eq!(scores.scores.get("http://w3:8000"), Some(&4));
    }

    #[test]
    fn test_jump_search_large_sequence() {
        // Sequence larger than default jump_size to verify multiple jumps
        let indexer = PositionalIndexer::new(64);
        let content: Vec<u64> = (1..=200).collect();

        let blocks_full = make_blocks(&content);
        let blocks_half = make_blocks(&content[..100]);

        indexer
            .apply_stored("http://w1:8000", &blocks_full, None)
            .unwrap();
        indexer
            .apply_stored("http://w2:8000", &blocks_half, None)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&content));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&200));
        assert_eq!(scores.scores.get("http://w2:8000"), Some(&100));
    }

    #[test]
    fn test_single_block_store_and_match() {
        let indexer = PositionalIndexer::default();
        let blocks = make_blocks(&[42]);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&[42]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&1));
        assert_eq!(scores.tree_sizes.get("http://w1:8000"), Some(&1));
    }

    #[test]
    fn test_remove_all_blocks_one_by_one() {
        let indexer = PositionalIndexer::default();
        let blocks = make_blocks(&[10, 20, 30]);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        // Remove in reverse order
        for block in blocks.iter().rev() {
            indexer.apply_removed("http://w1:8000", &[block.seq_hash]);
        }

        assert_eq!(indexer.current_size(), 0);
        let scores = indexer.find_matches(&hashes(&[10, 20, 30]));
        assert!(scores.scores.is_empty());
    }

    #[test]
    fn test_clear_nonexistent_worker_is_noop() {
        let indexer = PositionalIndexer::default();
        let blocks = make_blocks(&[10, 20]);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        indexer.apply_cleared("http://w2:8000");

        // w1 should be unaffected
        let scores = indexer.find_matches(&hashes(&[10, 20]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&2));
    }

    #[test]
    fn test_remove_worker_nonexistent_is_noop() {
        let indexer = PositionalIndexer::default();
        indexer.remove_worker("http://ghost:8000"); // no-op, no panic
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_concurrent_read_write() {
        let indexer = Arc::new(PositionalIndexer::new(4));
        let content: Vec<u64> = (1..=20).collect();
        let blocks = make_blocks(&content);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        let mut handles = Vec::new();

        // Spawn readers
        for _ in 0..4 {
            let idx = Arc::clone(&indexer);
            let ch = hashes(&content);
            handles.push(std::thread::spawn(move || {
                for _ in 0..100 {
                    let scores = idx.find_matches(&ch);
                    assert!(scores.scores.contains_key("http://w1:8000"));
                }
            }));
        }

        // Spawn writers (add new workers concurrently)
        for i in 0..4 {
            let idx = Arc::clone(&indexer);
            let worker_content: Vec<u64> = (1..=5).collect();
            handles.push(std::thread::spawn(move || {
                let worker = format!("http://writer{i}:8000");
                let blks = make_blocks(&worker_content);
                for _ in 0..50 {
                    idx.apply_stored(&worker, &blks, None).unwrap();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // w1 should still be matchable
        let scores = indexer.find_matches(&hashes(&content));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&20));
    }

    #[test]
    fn test_dashmap_cleanup_no_memory_leak() {
        // Verify DashMap entries are cleaned up when last worker is removed
        let indexer = PositionalIndexer::default();
        let blocks = make_blocks(&[10, 20, 30]);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();
        indexer
            .apply_stored("http://w2:8000", &blocks, None)
            .unwrap();

        assert!(!indexer.index.is_empty());

        indexer.remove_worker("http://w1:8000");
        // w2 still has entries, so DashMap should still have entries
        assert!(!indexer.index.is_empty());

        indexer.remove_worker("http://w2:8000");
        // Both workers removed — DashMap should be empty
        assert_eq!(indexer.index.len(), 0);
    }

    #[test]
    fn test_compute_content_hash_empty_tokens() {
        let hash = compute_content_hash(&[]);
        // Should produce a valid hash, not panic
        let hash2 = compute_content_hash(&[]);
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_compute_content_hash_single_token() {
        let hash = compute_content_hash(&[42]);
        assert_ne!(hash, compute_content_hash(&[43]));
    }

    #[test]
    fn test_seq_hash_rolling_correctness() {
        // Verify that seq_hashes computed by ensure_seq_hash_computed match make_blocks
        let content = vec![10u64, 20, 30, 40, 50];
        let blocks = make_blocks(&content);
        let content_hashes = hashes(&content);

        let mut seq_hashes: Vec<SequenceHash> = Vec::new();
        PositionalIndexer::ensure_seq_hash_computed(&mut seq_hashes, 4, &content_hashes);

        for (i, block) in blocks.iter().enumerate() {
            assert_eq!(
                seq_hashes[i], block.seq_hash,
                "seq_hash mismatch at position {i}"
            );
        }
    }

    #[test]
    fn test_query_prefix_of_stored() {
        // Query is shorter than stored sequence
        let indexer = PositionalIndexer::default();
        let blocks = make_blocks(&[10, 20, 30, 40, 50]);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&[10, 20]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&2));
        // tree_size should still be 5 (full stored depth)
        assert_eq!(scores.tree_sizes.get("http://w1:8000"), Some(&5));
    }

    #[test]
    fn test_disjoint_workers_no_shared_prefix() {
        let indexer = PositionalIndexer::default();
        let blocks_w1 = make_blocks(&[10, 20, 30]);
        let blocks_w2 = make_blocks(&[99, 88, 77]);
        indexer
            .apply_stored("http://w1:8000", &blocks_w1, None)
            .unwrap();
        indexer
            .apply_stored("http://w2:8000", &blocks_w2, None)
            .unwrap();

        // Query matching w1 only
        let scores = indexer.find_matches(&hashes(&[10, 20, 30]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&3));
        assert!(!scores.scores.contains_key("http://w2:8000"));

        // Query matching w2 only
        let scores = indexer.find_matches(&hashes(&[99, 88, 77]));
        assert!(!scores.scores.contains_key("http://w1:8000"));
        assert_eq!(scores.scores.get("http://w2:8000"), Some(&3));
    }

    #[test]
    #[should_panic(expected = "jump_size must be greater than 0")]
    fn test_zero_jump_size_panics() {
        let _ = PositionalIndexer::new(0);
    }

    #[test]
    fn test_current_size_across_operations() {
        let indexer = PositionalIndexer::default();
        assert_eq!(indexer.current_size(), 0);

        let blocks = make_blocks(&[10, 20, 30]);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();
        assert_eq!(indexer.current_size(), 3);

        // Same blocks on different worker — size should be 6
        indexer
            .apply_stored("http://w2:8000", &blocks, None)
            .unwrap();
        assert_eq!(indexer.current_size(), 6);

        indexer.apply_removed("http://w1:8000", &[blocks[2].seq_hash]);
        assert_eq!(indexer.current_size(), 5);

        indexer.apply_cleared("http://w2:8000");
        assert_eq!(indexer.current_size(), 2);

        indexer.remove_worker("http://w1:8000");
        assert_eq!(indexer.current_size(), 0);
    }

    // -----------------------------------------------------------------------
    // compute_request_content_hashes tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_request_hashes_basic() {
        // 8 tokens with block_size=4 → 2 full blocks
        let tokens: Vec<u32> = (1..=8).collect();
        let hashes = compute_request_content_hashes(&tokens, 4);
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes[0], compute_content_hash(&[1, 2, 3, 4]));
        assert_eq!(hashes[1], compute_content_hash(&[5, 6, 7, 8]));
    }

    #[test]
    fn test_request_hashes_partial_trailing_chunk_discarded() {
        // 10 tokens with block_size=4 → 2 full blocks, trailing [9,10] discarded
        let tokens: Vec<u32> = (1..=10).collect();
        let hashes = compute_request_content_hashes(&tokens, 4);
        assert_eq!(hashes.len(), 2);
    }

    #[test]
    fn test_request_hashes_fewer_than_block_size() {
        // Fewer tokens than block_size → empty (no full blocks)
        let hashes = compute_request_content_hashes(&[1, 2, 3], 4);
        assert!(hashes.is_empty());
    }

    #[test]
    fn test_request_hashes_empty_tokens() {
        let hashes = compute_request_content_hashes(&[], 16);
        assert!(hashes.is_empty());
    }

    #[test]
    fn test_request_hashes_exact_multiple() {
        // Exactly 3 blocks of size 2
        let tokens: Vec<u32> = (1..=6).collect();
        let hashes = compute_request_content_hashes(&tokens, 2);
        assert_eq!(hashes.len(), 3);
    }

    #[test]
    fn test_request_hashes_zero_block_size_returns_empty() {
        let hashes = compute_request_content_hashes(&[1, 2, 3], 0);
        assert!(hashes.is_empty());
    }

    #[test]
    fn test_request_hashes_block_size_1() {
        // Each token is its own block
        let tokens = vec![10u32, 20, 30];
        let hashes = compute_request_content_hashes(&tokens, 1);
        assert_eq!(hashes.len(), 3);
        assert_eq!(hashes[0], compute_content_hash(&[10]));
        assert_eq!(hashes[1], compute_content_hash(&[20]));
        assert_eq!(hashes[2], compute_content_hash(&[30]));
    }

    // -----------------------------------------------------------------------
    // End-to-end: store events → query with compute_request_content_hashes
    // -----------------------------------------------------------------------

    #[test]
    fn test_end_to_end_store_and_query() {
        // Simulate: backend stores blocks computed from tokens [1..=16] with block_size=4.
        // Router queries with the same tokens and block_size.
        let indexer = PositionalIndexer::default();
        let block_size = 4;
        let tokens: Vec<u32> = (1..=16).collect();

        // Simulate store events: compute content hashes the same way the router will
        let content_hashes: Vec<ContentHash> = tokens
            .chunks(block_size)
            .map(compute_content_hash)
            .collect();

        // Backend sends blocks with arbitrary seq_hashes (backend-specific algorithm)
        let blocks: Vec<StoredBlock> = content_hashes
            .iter()
            .enumerate()
            .map(|(i, &ch)| StoredBlock {
                seq_hash: SequenceHash(0xBEEF_0000 + i as u64), // opaque backend hash
                content_hash: ch,
            })
            .collect();

        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        // Router queries: compute content hashes from request tokens
        let query_hashes = compute_request_content_hashes(&tokens, block_size);
        let scores = indexer.find_matches(&query_hashes);
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&4));
    }

    #[test]
    fn test_end_to_end_partial_overlap() {
        let indexer = PositionalIndexer::default();
        let block_size = 4;

        // Worker cached tokens [1..=8] (2 blocks)
        let cached_tokens: Vec<u32> = (1..=8).collect();
        let blocks: Vec<StoredBlock> = cached_tokens
            .chunks(block_size)
            .enumerate()
            .map(|(i, chunk)| StoredBlock {
                seq_hash: SequenceHash(i as u64 + 1),
                content_hash: compute_content_hash(chunk),
            })
            .collect();
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        // Request has tokens [1..=16] — first 2 blocks match, last 2 don't
        let query_tokens: Vec<u32> = (1..=16).collect();
        let query_hashes = compute_request_content_hashes(&query_tokens, block_size);
        let scores = indexer.find_matches(&query_hashes);
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&2));
        assert_eq!(scores.tree_sizes.get("http://w1:8000"), Some(&2));
    }

    #[test]
    fn test_end_to_end_different_backends_same_content() {
        // Two workers with different seq_hashes but same content (different backends)
        let indexer = PositionalIndexer::new(4);
        let block_size = 4;
        let tokens: Vec<u32> = (1..=8).collect();
        let content_hashes: Vec<ContentHash> = tokens
            .chunks(block_size)
            .map(compute_content_hash)
            .collect();

        // Worker 1: SGLang-style seq_hashes
        let blocks_w1: Vec<StoredBlock> = content_hashes
            .iter()
            .enumerate()
            .map(|(i, &ch)| StoredBlock {
                seq_hash: SequenceHash(0xAAAA_0000 + i as u64),
                content_hash: ch,
            })
            .collect();

        // Worker 2: vLLM-style seq_hashes (different values, same content)
        let blocks_w2: Vec<StoredBlock> = content_hashes
            .iter()
            .enumerate()
            .map(|(i, &ch)| StoredBlock {
                seq_hash: SequenceHash(0xBBBB_0000 + i as u64),
                content_hash: ch,
            })
            .collect();

        indexer
            .apply_stored("http://sglang:8000", &blocks_w1, None)
            .unwrap();
        indexer
            .apply_stored("http://vllm:8000", &blocks_w2, None)
            .unwrap();

        // Router query: both workers match on content
        let query_hashes = compute_request_content_hashes(&tokens, block_size);
        let scores = indexer.find_matches(&query_hashes);
        assert_eq!(scores.scores.get("http://sglang:8000"), Some(&2));
        assert_eq!(scores.scores.get("http://vllm:8000"), Some(&2));
    }

    // -----------------------------------------------------------------------
    // Jump boundary tests: divergence at/near jump_size boundaries
    // -----------------------------------------------------------------------

    /// Helper: store a sequence for a worker via chained continuations of `chunk_size` blocks.
    fn store_via_continuations(
        indexer: &PositionalIndexer,
        worker: &str,
        content: &[u64],
        chunk_size: usize,
    ) {
        let all_blocks = make_blocks(content);
        let mut offset = 0;
        let mut parent: Option<SequenceHash> = None;
        while offset < all_blocks.len() {
            let end = (offset + chunk_size).min(all_blocks.len());
            let chunk = &all_blocks[offset..end];
            indexer.apply_stored(worker, chunk, parent).unwrap();
            parent = Some(chunk.last().unwrap().seq_hash);
            offset = end;
        }
    }

    #[test]
    fn test_divergence_at_jump_boundaries() {
        // 128-block sequence, workers diverge at specific boundary positions.
        // With jump_size=32, boundaries are at 32, 64, 96.
        let indexer = PositionalIndexer::new(32);
        let full: Vec<u64> = (1..=128).collect();
        let full_blocks = make_blocks(&full);
        indexer
            .apply_stored("http://full:8000", &full_blocks, None)
            .unwrap();

        // Workers diverge at positions 31, 32, 33 (around first jump boundary)
        for &depth in &[31, 32, 33] {
            let partial_blocks = make_blocks(&full[..depth]);
            let worker = format!("http://depth{depth}:8000");
            indexer
                .apply_stored(&worker, &partial_blocks, None)
                .unwrap();
        }

        // Workers diverge at positions 63, 64, 65 (around second jump boundary)
        for &depth in &[63, 64, 65] {
            let partial_blocks = make_blocks(&full[..depth]);
            let worker = format!("http://depth{depth}:8000");
            indexer
                .apply_stored(&worker, &partial_blocks, None)
                .unwrap();
        }

        let scores = indexer.find_matches(&hashes(&full));
        assert_eq!(scores.scores.get("http://full:8000"), Some(&128));
        assert_eq!(scores.scores.get("http://depth31:8000"), Some(&31));
        assert_eq!(scores.scores.get("http://depth32:8000"), Some(&32));
        assert_eq!(scores.scores.get("http://depth33:8000"), Some(&33));
        assert_eq!(scores.scores.get("http://depth63:8000"), Some(&63));
        assert_eq!(scores.scores.get("http://depth64:8000"), Some(&64));
        assert_eq!(scores.scores.get("http://depth65:8000"), Some(&65));
    }

    #[test]
    fn test_exact_jump_size_sequences() {
        // Sequences that are exact multiples of jump_size (32, 64, 96).
        let indexer = PositionalIndexer::new(32);

        for &len in &[32, 64, 96] {
            let content: Vec<u64> = (1..=len as u64).collect();
            let blocks = make_blocks(&content);
            let worker = format!("http://len{len}:8000");
            indexer.apply_stored(&worker, &blocks, None).unwrap();

            let scores = indexer.find_matches(&hashes(&content));
            assert_eq!(
                scores.scores.get(worker.as_str()),
                Some(&(len as u32)),
                "exact match failed for sequence length {len}"
            );
        }
    }

    #[test]
    fn test_off_by_one_jump_boundaries() {
        // Sequences at jump_size +/- 1 to catch off-by-one errors.
        let indexer = PositionalIndexer::new(32);
        let full: Vec<u64> = (1..=128).collect();

        for &len in &[31, 33, 63, 65, 95, 97] {
            let content = &full[..len];
            let blocks = make_blocks(content);
            let worker = format!("http://len{len}:8000");
            indexer.apply_stored(&worker, &blocks, None).unwrap();

            let scores = indexer.find_matches(&hashes(content));
            assert_eq!(
                scores.scores.get(worker.as_str()),
                Some(&(len as u32)),
                "exact match failed for sequence length {len}"
            );
        }
    }

    #[test]
    fn test_staggered_workers_across_jump_boundaries() {
        // 5 workers at depths 10, 20, 35, 64, 100 with jump_size=32.
        // Tests drain tracking across multiple jump boundaries.
        let indexer = PositionalIndexer::new(32);
        let full: Vec<u64> = (1..=100).collect();

        let depths = [10, 20, 35, 64, 100];
        for &depth in &depths {
            let blocks = make_blocks(&full[..depth]);
            let worker = format!("http://w{depth}:8000");
            indexer.apply_stored(&worker, &blocks, None).unwrap();
        }

        let scores = indexer.find_matches(&hashes(&full));
        for &depth in &depths {
            let worker = format!("http://w{depth}:8000");
            assert_eq!(
                scores.scores.get(worker.as_str()),
                Some(&(depth as u32)),
                "worker at depth {depth} has wrong score"
            );
        }
    }

    #[test]
    fn test_shared_prefix_diverge_at_jump_boundary() {
        // 3 workers share 40-block prefix, then diverge with different suffixes.
        let indexer = PositionalIndexer::new(32);
        let shared: Vec<u64> = (1..=40).collect();

        // Worker 1: shared + [1001..1060] = 100 blocks total
        let mut content_w1 = shared.clone();
        content_w1.extend(1001..=1060);
        let blocks_w1 = make_blocks(&content_w1);
        indexer
            .apply_stored("http://w1:8000", &blocks_w1, None)
            .unwrap();

        // Worker 2: shared + [2001..2020] = 60 blocks total
        let mut content_w2 = shared.clone();
        content_w2.extend(2001..=2020);
        let blocks_w2 = make_blocks(&content_w2);
        indexer
            .apply_stored("http://w2:8000", &blocks_w2, None)
            .unwrap();

        // Worker 3: shared only = 40 blocks
        let blocks_w3 = make_blocks(&shared);
        indexer
            .apply_stored("http://w3:8000", &blocks_w3, None)
            .unwrap();

        // Query with w1's content: w1 gets 100, w2 and w3 drain at position 40
        let scores = indexer.find_matches(&hashes(&content_w1));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&100));
        assert_eq!(scores.scores.get("http://w2:8000"), Some(&40));
        assert_eq!(scores.scores.get("http://w3:8000"), Some(&40));
    }

    #[test]
    fn test_very_long_sequence() {
        // 1000-block sequence: full match, prefix match, mid-divergence.
        let indexer = PositionalIndexer::new(64);
        let content: Vec<u64> = (1..=1000).collect();
        let blocks = make_blocks(&content);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        // Full match
        let scores = indexer.find_matches(&hashes(&content));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&1000));

        // Prefix match (query first 500)
        let scores = indexer.find_matches(&hashes(&content[..500]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&500));

        // Divergence: first 499 match, then different content
        let mut divergent = content[..499].to_vec();
        divergent.push(999999);
        let scores = indexer.find_matches(&hashes(&divergent));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&499));
    }

    // -----------------------------------------------------------------------
    // Deep continuation chain tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_deep_continuation_chain() {
        // Build 200-block sequence via 20 continuations of 10 blocks each.
        let indexer = PositionalIndexer::new(64);
        let content: Vec<u64> = (1..=200).collect();
        store_via_continuations(&indexer, "http://w1:8000", &content, 10);

        assert_eq!(indexer.current_size(), 200);

        // Full match
        let scores = indexer.find_matches(&hashes(&content));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&200));

        // Partial match at depth 150
        let scores = indexer.find_matches(&hashes(&content[..150]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&150));
    }

    #[test]
    fn test_continuation_chain_with_multiple_workers() {
        // Two workers: w1 builds 100 blocks via 10 continuations,
        // w2 builds 50 blocks via 5 continuations (same content prefix).
        let indexer = PositionalIndexer::new(32);
        let content: Vec<u64> = (1..=100).collect();

        store_via_continuations(&indexer, "http://w1:8000", &content, 10);
        store_via_continuations(&indexer, "http://w2:8000", &content[..50], 10);

        let scores = indexer.find_matches(&hashes(&content));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&100));
        assert_eq!(scores.scores.get("http://w2:8000"), Some(&50));
    }

    #[test]
    fn test_multiple_disjoint_sequences_per_worker() {
        // Same worker stores two completely disjoint sequences (no shared prefix).
        let indexer = PositionalIndexer::new(64);

        // Sequence 1: content [10, 20, 30] at positions 0-2
        let blocks1 = make_blocks(&[10, 20, 30]);
        indexer
            .apply_stored("http://w1:8000", &blocks1, None)
            .unwrap();

        // Sequence 2: content [100, 200, 300, 400] at positions 0-3
        // This overwrites positions 0-2 for w1 (same positions, different content)
        let blocks2 = make_blocks(&[100, 200, 300, 400]);
        indexer
            .apply_stored("http://w1:8000", &blocks2, None)
            .unwrap();

        // Query for sequence 2: w1 matches all 4
        let scores = indexer.find_matches(&hashes(&[100, 200, 300, 400]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&4));

        // Query for sequence 1: w1 also matches all 3 (both sequences indexed independently)
        let scores = indexer.find_matches(&hashes(&[10, 20, 30]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&3));
    }

    // -----------------------------------------------------------------------
    // Long sequence partial removal and stale entry tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_long_sequence_partial_removal() {
        // Store 100 blocks, remove the last 20 (blocks 80-99).
        // Verify remaining 80 blocks still match.
        let indexer = PositionalIndexer::new(32);
        let content: Vec<u64> = (1..=100).collect();
        let blocks = make_blocks(&content);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        // Remove blocks at positions 80-99 (seq_hashes from those blocks)
        let to_remove: Vec<SequenceHash> = blocks[80..].iter().map(|b| b.seq_hash).collect();
        indexer.apply_removed("http://w1:8000", &to_remove);

        assert_eq!(indexer.current_size(), 80);

        // Query full 100: only first 80 match
        let scores = indexer.find_matches(&hashes(&content));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&80));

        // Query first 80: all match
        let scores = indexer.find_matches(&hashes(&content[..80]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&80));
    }

    #[test]
    fn test_remove_parent_does_not_cascade() {
        // Removing a block at position 1 does NOT remove children at positions 2-4.
        // The positional indexer stores blocks independently — no parent-child pointers.
        //
        // With jump_size > sequence length, the jump skips directly from position 0
        // to the last position, bypassing the removed middle block entirely.
        // Use jump_size=1 to force linear scan and detect the gap.
        let indexer = PositionalIndexer::new(1);
        let blocks = make_blocks(&[10, 20, 30, 40, 50]);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        // Remove block at position 1 (content_hash=20)
        indexer.apply_removed("http://w1:8000", &[blocks[1].seq_hash]);

        // Children at positions 2-4 are NOT removed (no cascade)
        assert_eq!(indexer.current_size(), 4);

        // With linear scan (jump_size=1): position 0 matches, position 1 misses → score 1.
        // Positions 2-4 are still indexed but unreachable via prefix matching
        // because the scan terminates at the gap.
        let scores = indexer.find_matches(&hashes(&[10, 20, 30, 40, 50]));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&1));
    }

    #[test]
    fn test_long_sequence_clear_and_rebuild() {
        // Clear a 100-block sequence, rebuild with 100 different blocks.
        // Verify old data is completely gone.
        let indexer = PositionalIndexer::new(32);

        // Store original
        let original: Vec<u64> = (1..=100).collect();
        let blocks = make_blocks(&original);
        indexer
            .apply_stored("http://w1:8000", &blocks, None)
            .unwrap();

        // Clear
        indexer.apply_cleared("http://w1:8000");
        assert_eq!(indexer.current_size(), 0);

        // Rebuild with different content
        let replacement: Vec<u64> = (1001..=1100).collect();
        let new_blocks = make_blocks(&replacement);
        indexer
            .apply_stored("http://w1:8000", &new_blocks, None)
            .unwrap();

        // Old content: no match
        let scores = indexer.find_matches(&hashes(&original));
        assert!(!scores.scores.contains_key("http://w1:8000"));

        // New content: full match
        let scores = indexer.find_matches(&hashes(&replacement));
        assert_eq!(scores.scores.get("http://w1:8000"), Some(&100));
    }

    #[test]
    fn test_interleaved_long_sequences() {
        // 4 workers with shared prefix, staggered at 25/50/75/100 blocks.
        let indexer = PositionalIndexer::new(32);
        let content: Vec<u64> = (1..=100).collect();

        let depths = [25, 50, 75, 100];
        for &depth in &depths {
            let blocks = make_blocks(&content[..depth]);
            let worker = format!("http://w{depth}:8000");
            indexer.apply_stored(&worker, &blocks, None).unwrap();
        }

        let scores = indexer.find_matches(&hashes(&content));
        for &depth in &depths {
            let worker = format!("http://w{depth}:8000");
            assert_eq!(
                scores.scores.get(worker.as_str()),
                Some(&(depth as u32)),
                "worker at depth {depth} has wrong score"
            );
        }

        // Verify tree_sizes
        for &depth in &depths {
            let worker = format!("http://w{depth}:8000");
            assert_eq!(
                scores.tree_sizes.get(worker.as_str()),
                Some(&depth),
                "worker at depth {depth} has wrong tree_size"
            );
        }
    }
}
