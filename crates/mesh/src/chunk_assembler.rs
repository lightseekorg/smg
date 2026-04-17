//! Receiver-side chunk reassembly for oversized stream entries.
//!
//! Stream values that exceed the gRPC message size budget are split into
//! bounded chunks on the sender and reassembled here on the receiver.
//! Tree repair pages do NOT use this path — they are emitted already
//! bounded (≤ max_tree_repair_page_bytes) and ride the single-message
//! fast path.
//!
//! Semantics are at-most-once: if a chunk is lost, the partial assembly
//! is GC'd after `timeout`, and the application regenerates the value
//! on its next retry cycle. No retries or watermarks.

// Items are wired into the gossip receive path in a follow-up commit in
// this Step 3 PR. Allow (not expect) because tests already exercise the
// items, which makes `#[expect(dead_code)]` unfulfilled under test cfg.
#![allow(dead_code)]

use std::{
    cmp::Ordering,
    mem::size_of,
    time::{Duration, Instant},
};

use dashmap::DashMap;

/// Max concurrent in-flight assemblies. Prevents a peer from flooding
/// the map with partial assemblies for unique keys that never complete.
pub const DEFAULT_MAX_CONCURRENT_ASSEMBLIES: usize = 20;

/// Max total bytes held across all in-flight assemblies. Caps the
/// receiver-side memory independently of any sender-side payload choice.
pub const DEFAULT_MAX_ASSEMBLER_BYTES: usize = 512 * 1024 * 1024;

/// Hard cap on the chunk count advertised by a single chunk header.
/// AssemblyState allocates `received`/`chunks` vectors sized to `total`,
/// so an unvalidated peer-supplied `total = u32::MAX` would trigger a
/// multi-GB allocation before the byte-cap enforcer can react. 1024
/// chunks × 10 MB/chunk is 10 GB of assembled payload — well past the
/// byte cap, so bounds still catch any realistic traffic, while the
/// per-assembly vector overhead stays under ~25 KB.
pub const MAX_TOTAL_CHUNKS: u32 = 1024;

pub struct ChunkAssembler {
    assemblies: DashMap<String, AssemblyState>,
    max_concurrent: usize,
    max_bytes: usize,
}

struct AssemblyState {
    generation: u64,
    /// Count of distinct chunk indices recorded so far. Incremented only
    /// when a previously-empty slot is filled, so repeated receives of
    /// the same (generation, index) don't over-count. Completion is
    /// `received_count == chunks.len()`.
    received_count: u32,
    chunks: Vec<Option<Vec<u8>>>,
    created_at: Instant,
}

impl AssemblyState {
    fn new(generation: u64, total: u32) -> Self {
        let n = total as usize;
        Self {
            generation,
            received_count: 0,
            chunks: vec![None; n],
            created_at: Instant::now(),
        }
    }

    fn is_complete(&self) -> bool {
        self.received_count as usize == self.chunks.len()
    }

    /// Total receiver-side memory footprint of this assembly — chunk
    /// payloads plus the per-slot overhead of the `chunks` vector.
    /// Included in the byte cap so the cap bounds real memory, not
    /// just payload bytes.
    fn bytes_held(&self) -> usize {
        let payload: usize = self.chunks.iter().flatten().map(|c| c.len()).sum();
        let overhead = self.chunks.len() * size_of::<Option<Vec<u8>>>();
        payload + overhead
    }

    fn assemble(self) -> Vec<u8> {
        let total_size: usize = self.chunks.iter().flatten().map(|c| c.len()).sum();
        let mut out = Vec::with_capacity(total_size);
        for chunk in self.chunks.into_iter().flatten() {
            out.extend_from_slice(&chunk);
        }
        out
    }
}

impl ChunkAssembler {
    pub fn new() -> Self {
        Self::with_limits(
            DEFAULT_MAX_CONCURRENT_ASSEMBLIES,
            DEFAULT_MAX_ASSEMBLER_BYTES,
        )
    }

    pub fn with_limits(max_concurrent: usize, max_bytes: usize) -> Self {
        Self {
            assemblies: DashMap::new(),
            max_concurrent,
            max_bytes,
        }
    }

    /// Record an incoming chunk. Returns `Some(assembled)` once all chunks
    /// for the current generation have arrived; returns `None` otherwise.
    ///
    /// Generation handling is a three-way compare: a newer generation
    /// resets the state (older partials discarded), an older generation
    /// is dropped (newer state kept), equal continues recording.
    /// Malformed chunks are dropped silently: `total == 0`,
    /// `total > MAX_TOTAL_CHUNKS`, or `index >= total`.
    pub fn receive_chunk(
        &self,
        key: &str,
        generation: u64,
        index: u32,
        total: u32,
        data: Vec<u8>,
    ) -> Option<Vec<u8>> {
        if total == 0 || total > MAX_TOTAL_CHUNKS || index >= total {
            return None;
        }

        // Record the chunk under the shard lock. Assembly happens outside
        // the guard so the lock is not held during the allocation+copy of
        // the full reassembled buffer.
        let completed = {
            let mut entry = self
                .assemblies
                .entry(key.to_string())
                .or_insert_with(|| AssemblyState::new(generation, total));

            match generation.cmp(&entry.generation) {
                Ordering::Greater => {
                    *entry = AssemblyState::new(generation, total);
                }
                Ordering::Less => {
                    return None;
                }
                Ordering::Equal => {
                    if entry.chunks.len() != total as usize {
                        return None;
                    }
                }
            }

            if entry.chunks[index as usize].is_none() {
                entry.received_count += 1;
            }
            entry.chunks[index as usize] = Some(data);

            entry.is_complete()
        };

        if !completed {
            self.enforce_bounds();
            return None;
        }

        // Atomically take ownership only if the state is still our
        // generation and still complete. If another thread has moved on
        // (newer generation reset the state after our chunk landed), we
        // return None — the newer generation is what matters now.
        self.assemblies
            .remove_if(key, |_, state| {
                state.generation == generation && state.is_complete()
            })
            .map(|(_, state)| state.assemble())
    }

    /// Evict partials until the receiver-side memory bounds are satisfied.
    /// Over the concurrent-assembly cap: drop the oldest (by created_at).
    /// Over the byte cap: drop the largest (by total bytes held). Checked
    /// after each non-completing receive; the completing path removes its
    /// own entry before returning, so no enforcement is needed there.
    fn enforce_bounds(&self) {
        // Eviction candidates: only partial assemblies. Completed ones are
        // owned by a concurrent receive_chunk that is about to extract
        // them via remove_if; evicting them here would silently swallow a
        // fully-assembled payload.
        while self.assemblies.len() > self.max_concurrent {
            match self.oldest_incomplete_key() {
                Some(k) => {
                    self.assemblies
                        .remove_if(&k, |_, state| !state.is_complete());
                }
                None => break,
            }
        }
        loop {
            let total: usize = self.assemblies.iter().map(|e| e.value().bytes_held()).sum();
            if total <= self.max_bytes {
                break;
            }
            match self.largest_incomplete_key() {
                Some(k) => {
                    self.assemblies
                        .remove_if(&k, |_, state| !state.is_complete());
                }
                None => break,
            }
        }
    }

    fn oldest_incomplete_key(&self) -> Option<String> {
        self.assemblies
            .iter()
            .filter(|e| !e.value().is_complete())
            .min_by_key(|e| e.value().created_at)
            .map(|e| e.key().clone())
    }

    fn largest_incomplete_key(&self) -> Option<String> {
        self.assemblies
            .iter()
            .filter(|e| !e.value().is_complete())
            .max_by_key(|e| e.value().bytes_held())
            .map(|e| e.key().clone())
    }

    /// Drop partial assemblies older than `timeout`. Collect-then-remove
    /// to avoid holding DashMap shard locks during mutation, and each
    /// removal re-checks the timeout so a concurrent receive that reset
    /// the entry to a new generation (fresh created_at) is spared.
    /// Complete assemblies are also skipped — they belong to an in-flight
    /// receive_chunk that is about to extract them via remove_if.
    pub fn gc(&self, timeout: Duration) {
        let expired: Vec<String> = self
            .assemblies
            .iter()
            .filter(|e| e.value().created_at.elapsed() >= timeout && !e.value().is_complete())
            .map(|e| e.key().clone())
            .collect();
        for key in expired {
            self.assemblies.remove_if(&key, |_, state| {
                state.created_at.elapsed() >= timeout && !state.is_complete()
            });
        }
    }

    #[cfg(test)]
    pub(crate) fn in_flight(&self) -> usize {
        self.assemblies.len()
    }

    #[cfg(test)]
    pub(crate) fn total_bytes(&self) -> usize {
        self.assemblies.iter().map(|e| e.value().bytes_held()).sum()
    }
}

impl Default for ChunkAssembler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use super::*;

    #[test]
    fn test_single_chunk_round_trip() {
        let asm = ChunkAssembler::new();
        let out = asm.receive_chunk("k", 1, 0, 1, b"hello".to_vec());
        assert_eq!(out.as_deref(), Some(b"hello".as_slice()));
        assert_eq!(asm.in_flight(), 0, "completed assembly should be removed");
    }

    #[test]
    fn test_multi_chunk_in_order() {
        let asm = ChunkAssembler::new();
        assert!(asm.receive_chunk("k", 1, 0, 3, b"aaa".to_vec()).is_none());
        assert!(asm.receive_chunk("k", 1, 1, 3, b"bbb".to_vec()).is_none());
        let out = asm.receive_chunk("k", 1, 2, 3, b"ccc".to_vec()).unwrap();
        assert_eq!(out, b"aaabbbccc");
        assert_eq!(asm.in_flight(), 0);
    }

    #[test]
    fn test_multi_chunk_out_of_order() {
        let asm = ChunkAssembler::new();
        assert!(asm.receive_chunk("k", 1, 2, 3, b"ccc".to_vec()).is_none());
        assert!(asm.receive_chunk("k", 1, 0, 3, b"aaa".to_vec()).is_none());
        let out = asm.receive_chunk("k", 1, 1, 3, b"bbb".to_vec()).unwrap();
        assert_eq!(out, b"aaabbbccc", "chunks must assemble in index order");
    }

    #[test]
    fn test_generation_reset_discards_older_partials() {
        let asm = ChunkAssembler::new();
        assert!(asm.receive_chunk("k", 5, 0, 3, b"old-0".to_vec()).is_none());
        assert!(asm.receive_chunk("k", 5, 1, 3, b"old-1".to_vec()).is_none());

        assert!(asm.receive_chunk("k", 6, 0, 2, b"new-0".to_vec()).is_none());
        let out = asm.receive_chunk("k", 6, 1, 2, b"new-1".to_vec()).unwrap();
        assert_eq!(out, b"new-0new-1");
    }

    #[test]
    fn test_delayed_older_generation_chunk_is_dropped() {
        let asm = ChunkAssembler::new();
        // gen=6 starts and records one chunk.
        assert!(asm.receive_chunk("k", 6, 0, 2, b"new-0".to_vec()).is_none());

        // A delayed gen=5 chunk arrives — must NOT reset the gen=6 state.
        assert!(asm
            .receive_chunk("k", 5, 0, 3, b"stale-0".to_vec())
            .is_none());

        // Completing gen=6 must still yield the gen=6 payload.
        let out = asm.receive_chunk("k", 6, 1, 2, b"new-1".to_vec()).unwrap();
        assert_eq!(
            out, b"new-0new-1",
            "stale older chunk must not overwrite newer state"
        );
    }

    #[test]
    fn test_gc_removes_stale_partials() {
        let asm = ChunkAssembler::new();
        assert!(asm.receive_chunk("k", 1, 0, 3, b"aaa".to_vec()).is_none());
        assert_eq!(asm.in_flight(), 1);

        thread::sleep(Duration::from_millis(50));
        asm.gc(Duration::from_millis(30));
        assert_eq!(asm.in_flight(), 0, "stale partial should be GC'd");
    }

    #[test]
    fn test_gc_keeps_recent_partials() {
        let asm = ChunkAssembler::new();
        assert!(asm.receive_chunk("k", 1, 0, 3, b"aaa".to_vec()).is_none());
        asm.gc(Duration::from_secs(10));
        assert_eq!(asm.in_flight(), 1, "recent partial should survive gc");
    }

    #[test]
    fn test_gc_skips_complete_assemblies() {
        // A complete assembly sitting in the map (before its owning
        // receive_chunk extracts it) must not be removed by gc even if
        // the timeout would otherwise apply.
        let asm = ChunkAssembler::new();
        let old = Instant::now() - Duration::from_secs(60);
        asm.assemblies.insert(
            "complete".to_string(),
            AssemblyState {
                generation: 1,
                received_count: 2,
                chunks: vec![Some(vec![0u8; 10]), Some(vec![0u8; 10])],
                created_at: old,
            },
        );
        asm.gc(Duration::from_secs(1));
        assert!(
            asm.assemblies.contains_key("complete"),
            "gc must not remove a complete assembly"
        );
    }

    #[test]
    fn test_gc_rechecks_timeout_at_remove() {
        // After the collect phase marks a key as expired, simulate a
        // concurrent receive_chunk replacing that entry with fresh
        // created_at by inserting a young state under the same key
        // between collect and remove. The remove_if predicate re-checks
        // timeout and spares the young entry.
        let asm = ChunkAssembler::new();
        let old = Instant::now() - Duration::from_secs(60);
        asm.assemblies.insert(
            "k".to_string(),
            AssemblyState {
                generation: 1,
                received_count: 0,
                chunks: vec![None, None],
                created_at: old,
            },
        );
        // Simulate the concurrent reset before gc fires by overwriting
        // with a fresh state — this is what the TOCTOU window allowed
        // before the fix.
        asm.assemblies.insert(
            "k".to_string(),
            AssemblyState {
                generation: 2,
                received_count: 0,
                chunks: vec![None, None],
                created_at: Instant::now(),
            },
        );
        asm.gc(Duration::from_secs(1));
        assert!(
            asm.assemblies.contains_key("k"),
            "freshly-reset entry must survive gc"
        );
    }

    #[test]
    fn test_malformed_chunk_is_dropped() {
        let asm = ChunkAssembler::new();
        assert!(asm.receive_chunk("k", 1, 0, 0, b"x".to_vec()).is_none());
        assert!(asm.receive_chunk("k", 1, 5, 3, b"x".to_vec()).is_none());
        assert_eq!(asm.in_flight(), 0);
    }

    #[test]
    fn test_oversized_total_chunks_is_rejected() {
        let asm = ChunkAssembler::new();
        // total = u32::MAX would trigger multi-GB allocation if admitted.
        assert!(asm
            .receive_chunk("k", 1, 0, u32::MAX, b"x".to_vec())
            .is_none());
        // Just past the cap is also rejected.
        assert!(asm
            .receive_chunk("k", 1, 0, MAX_TOTAL_CHUNKS + 1, b"x".to_vec())
            .is_none());
        assert_eq!(
            asm.in_flight(),
            0,
            "oversized total must not allocate an AssemblyState"
        );
    }

    #[test]
    fn test_bounds_evict_oldest_when_too_many_concurrent() {
        let asm = ChunkAssembler::with_limits(3, usize::MAX);
        // 4 partials against a cap of 3 — oldest must be evicted.
        for i in 0..4 {
            assert!(asm
                .receive_chunk(&format!("k{i}"), 1, 0, 2, vec![0u8; 10])
                .is_none());
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(asm.in_flight(), 3, "concurrent cap enforced");
        assert!(
            !asm.assemblies.contains_key("k0"),
            "oldest partial (k0) should be evicted"
        );
    }

    #[test]
    fn test_bounds_evict_largest_when_over_byte_cap() {
        // Use payloads large enough that per-assembly vec overhead
        // (~25 bytes × total_chunks) is negligible relative to payload.
        let cap = 10_000;
        let asm = ChunkAssembler::with_limits(usize::MAX, cap);
        assert!(asm
            .receive_chunk("k_small", 1, 0, 2, vec![0u8; 2_000])
            .is_none());
        assert!(asm
            .receive_chunk("k_med", 1, 0, 2, vec![0u8; 3_000])
            .is_none());
        assert!(asm
            .receive_chunk("k_big", 1, 0, 2, vec![0u8; 6_000])
            .is_none());

        assert!(
            asm.total_bytes() <= cap,
            "byte cap enforced, total = {}",
            asm.total_bytes()
        );
        assert!(
            !asm.assemblies.contains_key("k_big"),
            "largest partial (k_big) should be evicted"
        );
    }

    #[test]
    fn test_enforce_bounds_skips_complete_assemblies() {
        // Simulate the race window: a just-completed assembly sits in
        // the map (before the owning receive_chunk extracts it), and a
        // concurrent receive_chunk on a different key triggers
        // enforce_bounds. The complete one must survive.
        let asm = ChunkAssembler::with_limits(1, usize::MAX);
        let old = Instant::now() - Duration::from_secs(60);
        asm.assemblies.insert(
            "complete".to_string(),
            AssemblyState {
                generation: 1,
                received_count: 2,
                chunks: vec![Some(vec![0u8; 10]), Some(vec![0u8; 10])],
                created_at: old,
            },
        );
        // Add a partial — this is over the cap=1, so enforce_bounds runs
        // (via the !completed branch of receive_chunk).
        assert!(asm
            .receive_chunk("partial", 1, 0, 2, vec![0u8; 10])
            .is_none());

        assert!(
            asm.assemblies.contains_key("complete"),
            "complete assembly must not be evicted"
        );
        assert!(
            !asm.assemblies.contains_key("partial"),
            "partial (the only incomplete candidate) should be evicted"
        );
    }

    #[test]
    fn test_bounds_not_enforced_on_completion() {
        // If receive_chunk completes an assembly, the entry leaves the
        // map before bounds are checked — the just-completed value must
        // still be returned regardless of size.
        let asm = ChunkAssembler::with_limits(usize::MAX, 10);
        let out = asm
            .receive_chunk("k", 1, 0, 1, vec![0u8; 500])
            .expect("single-chunk completion returns bytes");
        assert_eq!(out.len(), 500);
        assert_eq!(asm.in_flight(), 0);
    }

    #[test]
    fn test_multiple_keys_independent() {
        let asm = ChunkAssembler::new();
        assert!(asm.receive_chunk("a", 1, 0, 2, b"a0".to_vec()).is_none());
        assert!(asm.receive_chunk("b", 1, 0, 2, b"b0".to_vec()).is_none());
        assert_eq!(asm.in_flight(), 2);

        let a = asm.receive_chunk("a", 1, 1, 2, b"a1".to_vec()).unwrap();
        assert_eq!(a, b"a0a1");
        assert_eq!(asm.in_flight(), 1, "completing a should not touch b");

        let b = asm.receive_chunk("b", 1, 1, 2, b"b1".to_vec()).unwrap();
        assert_eq!(b, b"b0b1");
        assert_eq!(asm.in_flight(), 0);
    }
}
