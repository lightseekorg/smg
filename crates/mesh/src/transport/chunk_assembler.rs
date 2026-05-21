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

use std::{
    cmp::Ordering,
    mem::size_of,
    time::{Duration, Instant},
};

use bytes::Bytes;
use dashmap::DashMap;

/// Max concurrent in-flight assemblies. Prevents a peer from flooding
/// the map with partial assemblies for unique keys that never complete.
pub const DEFAULT_MAX_CONCURRENT_ASSEMBLIES: usize = 20;

/// Max total bytes held across all in-flight assemblies. Caps the
/// receiver-side memory independently of any sender-side payload choice.
pub const DEFAULT_MAX_ASSEMBLER_BYTES: usize = 512 * 1024 * 1024;

/// Hard cap on the chunk count advertised by a single chunk header.
/// AssemblyState allocates a `chunks` vector sized to `total`, so an
/// unvalidated peer-supplied `total = u32::MAX` would trigger a
/// multi-GB allocation before the byte-cap enforcer can react. 1024
/// chunks × 10 MB/chunk is 10 GB of assembled payload — well past the
/// byte cap, so bounds still catch any realistic traffic, while the
/// per-assembly vector overhead stays under ~25 KB.
pub const MAX_TOTAL_CHUNKS: u32 = 1024;

/// Composite key scoping an assembly by sender peer + application key.
/// Two peers chunking the same key in flight don't collide.
type AssemblyKey = (String, String);

pub struct ChunkAssembler {
    assemblies: DashMap<AssemblyKey, AssemblyState>,
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
    chunks: Vec<Option<Bytes>>,
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
        let overhead = self.chunks.len() * size_of::<Option<Bytes>>();
        payload + overhead
    }

    /// Return the assembled chunks as a fragmented buffer: each chunk
    /// stays in its own `Bytes` handle (no contiguous memcpy of the
    /// full payload).
    fn assemble(self) -> Vec<Bytes> {
        self.chunks.into_iter().flatten().collect()
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
    /// Assemblies are scoped by `(peer_id, key)` so two senders streaming
    /// chunked values under the same key don't collide through the
    /// generation compare. A single node-wide assembler serves all
    /// inbound streams; the peer-scope prevents one peer's newer-
    /// generation reset from wiping another peer's in-flight partial.
    ///
    /// Generation handling is a three-way compare: a newer generation
    /// resets the state (older partials discarded), an older generation
    /// is dropped (newer state kept), equal continues recording.
    /// Malformed chunks are dropped silently: `total == 0`,
    /// `total > MAX_TOTAL_CHUNKS`, or `index >= total`.
    pub fn receive_chunk(
        &self,
        peer_id: &str,
        key: &str,
        generation: u64,
        index: u32,
        total: u32,
        data: Bytes,
    ) -> Option<Vec<Bytes>> {
        if total == 0 || total > MAX_TOTAL_CHUNKS || index >= total {
            return None;
        }

        let asm_key: AssemblyKey = (peer_id.to_string(), key.to_string());

        // Record the chunk under the shard lock. Assembly happens outside
        // the guard so the lock is not held during the allocation+copy of
        // the full reassembled buffer.
        let completed = {
            let mut entry = self
                .assemblies
                .entry(asm_key.clone())
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
            .remove_if(&asm_key, |_, state| {
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
        // Cap checks and eviction target only incomplete assemblies;
        // transiently-complete entries awaiting remove_if must not count.
        while self.incomplete_count() > self.max_concurrent {
            let Some((k, gen)) = self.oldest_incomplete() else {
                break;
            };
            self.try_evict_stale(&k, gen);
        }
        while self.incomplete_bytes() > self.max_bytes {
            let Some((k, gen)) = self.largest_incomplete() else {
                break;
            };
            self.try_evict_stale(&k, gen);
        }
    }

    fn incomplete_count(&self) -> usize {
        self.assemblies
            .iter()
            .filter(|e| !e.value().is_complete())
            .count()
    }

    fn incomplete_bytes(&self) -> usize {
        self.assemblies
            .iter()
            .filter(|e| !e.value().is_complete())
            .map(|e| e.value().bytes_held())
            .sum()
    }

    fn try_evict_stale(&self, key: &AssemblyKey, gen: u64) {
        self.assemblies
            .remove_if(key, |_, s| s.generation == gen && !s.is_complete());
    }

    fn oldest_incomplete(&self) -> Option<(AssemblyKey, u64)> {
        self.assemblies
            .iter()
            .filter(|e| !e.value().is_complete())
            .min_by_key(|e| e.value().created_at)
            .map(|e| (e.key().clone(), e.value().generation))
    }

    fn largest_incomplete(&self) -> Option<(AssemblyKey, u64)> {
        self.assemblies
            .iter()
            .filter(|e| !e.value().is_complete())
            .max_by_key(|e| e.value().bytes_held())
            .map(|e| (e.key().clone(), e.value().generation))
    }

    /// Drop partial assemblies older than `timeout`. Collect-then-remove
    /// to avoid holding DashMap shard locks during mutation, and each
    /// removal re-checks the timeout so a concurrent receive that reset
    /// the entry to a new generation (fresh created_at) is spared.
    /// Complete assemblies are also skipped — they belong to an in-flight
    /// receive_chunk that is about to extract them via remove_if.
    pub fn gc(&self, timeout: Duration) {
        let expired = self.collect_expired(timeout);
        self.remove_stale(&expired, timeout);
    }

    fn collect_expired(&self, timeout: Duration) -> Vec<AssemblyKey> {
        self.assemblies
            .iter()
            .filter(|e| e.value().created_at.elapsed() >= timeout && !e.value().is_complete())
            .map(|e| e.key().clone())
            .collect()
    }

    /// Invalidate any pending partial assembly for `(peer_id, key)`.
    /// The single-chunk fast path calls this so a fresh value replaces
    /// leftover fragments from an older multi-chunk publish under the
    /// same key, instead of waiting for the 30 s GC to clean them up.
    /// No-op if no entry exists.
    pub fn drop_pending(&self, peer_id: &str, key: &str) {
        let asm_key: AssemblyKey = (peer_id.to_string(), key.to_string());
        self.assemblies.remove(&asm_key);
    }

    fn remove_stale(&self, keys: &[AssemblyKey], timeout: Duration) {
        for key in keys {
            self.assemblies.remove_if(key, |_, state| {
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

    /// Flatten a fragmented assembly result into a contiguous Vec<u8>
    /// for test assertions only. Production callers handle fragments
    /// directly (zero-copy fan-out).
    fn flatten(bs: &[Bytes]) -> Vec<u8> {
        bs.iter().flat_map(|b| b.iter().copied()).collect()
    }

    #[test]
    fn test_single_chunk_round_trip() {
        let asm = ChunkAssembler::new();
        let out = asm
            .receive_chunk("peer", "k", 1, 0, 1, Bytes::from_static(b"hello"))
            .unwrap();
        assert_eq!(flatten(&out), b"hello");
        assert_eq!(asm.in_flight(), 0, "completed assembly should be removed");
    }

    #[test]
    fn test_multi_chunk_in_order() {
        let asm = ChunkAssembler::new();
        assert!(asm
            .receive_chunk("peer", "k", 1, 0, 3, Bytes::from_static(b"aaa"))
            .is_none());
        assert!(asm
            .receive_chunk("peer", "k", 1, 1, 3, Bytes::from_static(b"bbb"))
            .is_none());
        let out = asm
            .receive_chunk("peer", "k", 1, 2, 3, Bytes::from_static(b"ccc"))
            .unwrap();
        assert_eq!(flatten(&out), b"aaabbbccc");
        assert_eq!(asm.in_flight(), 0);
    }

    #[test]
    fn test_multi_chunk_out_of_order() {
        let asm = ChunkAssembler::new();
        assert!(asm
            .receive_chunk("peer", "k", 1, 2, 3, Bytes::from_static(b"ccc"))
            .is_none());
        assert!(asm
            .receive_chunk("peer", "k", 1, 0, 3, Bytes::from_static(b"aaa"))
            .is_none());
        let out = asm
            .receive_chunk("peer", "k", 1, 1, 3, Bytes::from_static(b"bbb"))
            .unwrap();
        assert_eq!(
            flatten(&out),
            b"aaabbbccc",
            "chunks must assemble in index order"
        );
    }

    #[test]
    fn test_generation_reset_discards_older_partials() {
        let asm = ChunkAssembler::new();
        assert!(asm
            .receive_chunk("peer", "k", 5, 0, 3, Bytes::from_static(b"old-0"))
            .is_none());
        assert!(asm
            .receive_chunk("peer", "k", 5, 1, 3, Bytes::from_static(b"old-1"))
            .is_none());

        assert!(asm
            .receive_chunk("peer", "k", 6, 0, 2, Bytes::from_static(b"new-0"))
            .is_none());
        let out = asm
            .receive_chunk("peer", "k", 6, 1, 2, Bytes::from_static(b"new-1"))
            .unwrap();
        assert_eq!(flatten(&out), b"new-0new-1");
    }

    #[test]
    fn test_delayed_older_generation_chunk_is_dropped() {
        let asm = ChunkAssembler::new();
        // gen=6 starts and records one chunk.
        assert!(asm
            .receive_chunk("peer", "k", 6, 0, 2, Bytes::from_static(b"new-0"))
            .is_none());

        // A delayed gen=5 chunk arrives — must NOT reset the gen=6 state.
        assert!(asm
            .receive_chunk("peer", "k", 5, 0, 3, Bytes::from_static(b"stale-0"))
            .is_none());

        // Completing gen=6 must still yield the gen=6 payload.
        let out = asm
            .receive_chunk("peer", "k", 6, 1, 2, Bytes::from_static(b"new-1"))
            .unwrap();
        assert_eq!(
            flatten(&out),
            b"new-0new-1",
            "stale older chunk must not overwrite newer state"
        );
    }

    #[test]
    fn test_gc_removes_stale_partials() {
        let asm = ChunkAssembler::new();
        assert!(asm
            .receive_chunk("peer", "k", 1, 0, 3, Bytes::from_static(b"aaa"))
            .is_none());
        assert_eq!(asm.in_flight(), 1);

        thread::sleep(Duration::from_millis(50));
        asm.gc(Duration::from_millis(30));
        assert_eq!(asm.in_flight(), 0, "stale partial should be GC'd");
    }

    #[test]
    fn test_gc_keeps_recent_partials() {
        let asm = ChunkAssembler::new();
        assert!(asm
            .receive_chunk("peer", "k", 1, 0, 3, Bytes::from_static(b"aaa"))
            .is_none());
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
        let k: AssemblyKey = ("peer".into(), "complete".into());
        asm.assemblies.insert(
            k.clone(),
            AssemblyState {
                generation: 1,
                received_count: 2,
                chunks: vec![
                    Some(Bytes::from(vec![0u8; 10])),
                    Some(Bytes::from(vec![0u8; 10])),
                ],
                created_at: old,
            },
        );
        asm.gc(Duration::from_secs(1));
        assert!(
            asm.assemblies.contains_key(&k),
            "gc must not remove a complete assembly"
        );
    }

    #[test]
    fn test_gc_rechecks_timeout_at_remove() {
        // Exercise the real collect→remove race window by driving the
        // two phases manually. Overwrite the entry between phases; the
        // remove_if timeout re-check must spare the fresh state.
        let asm = ChunkAssembler::new();
        let timeout = Duration::from_secs(1);
        let k: AssemblyKey = ("peer".into(), "k".into());

        asm.assemblies.insert(
            k.clone(),
            AssemblyState {
                generation: 1,
                received_count: 0,
                chunks: vec![None, None],
                created_at: Instant::now() - Duration::from_secs(60),
            },
        );

        // Phase 1: collect sees the expired entry.
        let expired = asm.collect_expired(timeout);
        assert_eq!(expired, vec![k.clone()]);

        // Race: a concurrent receive_chunk resets the entry to a fresh
        // generation before gc's remove phase fires.
        asm.assemblies.insert(
            k.clone(),
            AssemblyState {
                generation: 2,
                received_count: 0,
                chunks: vec![None, None],
                created_at: Instant::now(),
            },
        );

        // Phase 2: remove_stale re-checks the timeout predicate; the
        // fresh entry's created_at is <1s ago, so the predicate fails
        // and the fresh state is spared.
        asm.remove_stale(&expired, timeout);

        let survived = asm.assemblies.get(&k).expect("fresh entry must survive");
        assert_eq!(
            survived.generation, 2,
            "gc evicted the fresh replacement instead of sparing it"
        );
    }

    #[test]
    fn test_malformed_chunk_is_dropped() {
        let asm = ChunkAssembler::new();
        assert!(asm
            .receive_chunk("peer", "k", 1, 0, 0, Bytes::from_static(b"x"))
            .is_none());
        assert!(asm
            .receive_chunk("peer", "k", 1, 5, 3, Bytes::from_static(b"x"))
            .is_none());
        assert_eq!(asm.in_flight(), 0);
    }

    #[test]
    fn test_oversized_total_chunks_is_rejected() {
        let asm = ChunkAssembler::new();
        // total = u32::MAX would trigger multi-GB allocation if admitted.
        assert!(asm
            .receive_chunk("peer", "k", 1, 0, u32::MAX, Bytes::from_static(b"x"))
            .is_none());
        // Just past the cap is also rejected.
        assert!(asm
            .receive_chunk(
                "peer",
                "k",
                1,
                0,
                MAX_TOTAL_CHUNKS + 1,
                Bytes::from_static(b"x")
            )
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
                .receive_chunk(
                    "peer",
                    &format!("k{i}"),
                    1,
                    0,
                    2,
                    Bytes::from(vec![0u8; 10])
                )
                .is_none());
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(asm.in_flight(), 3, "concurrent cap enforced");
        assert!(
            !asm.assemblies.contains_key(&("peer".into(), "k0".into())),
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
            .receive_chunk("peer", "k_small", 1, 0, 2, Bytes::from(vec![0u8; 2_000]))
            .is_none());
        assert!(asm
            .receive_chunk("peer", "k_med", 1, 0, 2, Bytes::from(vec![0u8; 3_000]))
            .is_none());
        assert!(asm
            .receive_chunk("peer", "k_big", 1, 0, 2, Bytes::from(vec![0u8; 6_000]))
            .is_none());

        assert!(
            asm.total_bytes() <= cap,
            "byte cap enforced, total = {}",
            asm.total_bytes()
        );
        assert!(
            !asm.assemblies
                .contains_key(&("peer".into(), "k_big".into())),
            "largest partial (k_big) should be evicted"
        );
    }

    #[test]
    fn test_enforce_bounds_skips_complete_assemblies() {
        // cap=1, one complete + two partials. Real incomplete=2 > 1,
        // so enforce_bounds must evict one. The complete must not be
        // picked even though it is the oldest by created_at.
        let asm = ChunkAssembler::with_limits(1, usize::MAX);
        let k_complete: AssemblyKey = ("peer".into(), "complete".into());
        let k_partial_a: AssemblyKey = ("peer".into(), "partial_a".into());
        let k_partial_b: AssemblyKey = ("peer".into(), "partial_b".into());
        asm.assemblies.insert(
            k_complete.clone(),
            AssemblyState {
                generation: 1,
                received_count: 2,
                chunks: vec![
                    Some(Bytes::from(vec![0u8; 10])),
                    Some(Bytes::from(vec![0u8; 10])),
                ],
                created_at: Instant::now() - Duration::from_secs(60),
            },
        );
        asm.assemblies.insert(
            k_partial_a.clone(),
            AssemblyState {
                generation: 1,
                received_count: 0,
                chunks: vec![None, None],
                created_at: Instant::now() - Duration::from_secs(30),
            },
        );
        asm.assemblies.insert(
            k_partial_b.clone(),
            AssemblyState {
                generation: 1,
                received_count: 0,
                chunks: vec![None, None],
                created_at: Instant::now(),
            },
        );

        asm.enforce_bounds();

        assert!(
            asm.assemblies.contains_key(&k_complete),
            "complete (even oldest) must not be evicted"
        );
        assert!(
            !asm.assemblies.contains_key(&k_partial_a),
            "oldest incomplete should be evicted"
        );
        assert!(
            asm.assemblies.contains_key(&k_partial_b),
            "newer incomplete survives"
        );
    }

    #[test]
    fn test_enforce_bounds_cap_counts_only_incomplete() {
        // Transient complete-but-unextracted entries must not push the
        // incomplete population over cap. Here the cap is 1, and we
        // preload a complete entry + a partial. enforce_bounds should
        // see 1 incomplete (<=1), not 2 total, and leave the partial
        // alone.
        let asm = ChunkAssembler::with_limits(1, usize::MAX);
        let k_complete: AssemblyKey = ("peer".into(), "complete".into());
        let k_partial: AssemblyKey = ("peer".into(), "partial".into());
        asm.assemblies.insert(
            k_complete.clone(),
            AssemblyState {
                generation: 1,
                received_count: 2,
                chunks: vec![
                    Some(Bytes::from(vec![0u8; 10])),
                    Some(Bytes::from(vec![0u8; 10])),
                ],
                created_at: Instant::now(),
            },
        );
        asm.assemblies.insert(
            k_partial.clone(),
            AssemblyState {
                generation: 1,
                received_count: 1,
                chunks: vec![Some(Bytes::from(vec![0u8; 10])), None],
                created_at: Instant::now(),
            },
        );

        // Build the initial state manually so we can observe the
        // filter-based count without triggering a receive.
        let incomplete_count = asm
            .assemblies
            .iter()
            .filter(|e| !e.value().is_complete())
            .count();
        assert_eq!(incomplete_count, 1, "only 'partial' is incomplete");

        // Call enforce_bounds directly (via a no-op receive path).
        // With the fix: incomplete=1 <= cap=1, so no eviction.
        asm.enforce_bounds();

        assert!(
            asm.assemblies.contains_key(&k_complete),
            "complete assembly still present"
        );
        assert!(
            asm.assemblies.contains_key(&k_partial),
            "partial must not be evicted — real incomplete count was under cap"
        );
    }

    #[test]
    fn test_enforce_bounds_spares_newer_generation_replacement() {
        // Simulate the race: enforce_bounds selects key "k" with
        // generation 5, but before the remove_if fires, another thread
        // resets "k" to generation 6. The generation-gated predicate
        // must spare the replacement. We emulate the race in-process
        // by splitting selection from removal via the private helper.
        let asm = ChunkAssembler::with_limits(1, usize::MAX);
        let k: AssemblyKey = ("peer".into(), "k".into());

        // Insert a stale generation-5 partial as the eviction target.
        asm.assemblies.insert(
            k.clone(),
            AssemblyState {
                generation: 5,
                received_count: 0,
                chunks: vec![None, None],
                created_at: Instant::now() - Duration::from_secs(60),
            },
        );

        // Enforcement picks this key + generation.
        let (picked_key, gen) = asm.oldest_incomplete().expect("have an incomplete");
        assert_eq!(picked_key, k);
        assert_eq!(gen, 5);

        // Racing thread replaces the state with a fresh generation-6.
        asm.assemblies.insert(
            k.clone(),
            AssemblyState {
                generation: 6,
                received_count: 0,
                chunks: vec![None, None],
                created_at: Instant::now(),
            },
        );

        // The delayed remove_if must NOT evict the generation-6 state.
        asm.assemblies.remove_if(&picked_key, |_, state| {
            state.generation == gen && !state.is_complete()
        });

        let survived = asm
            .assemblies
            .get(&k)
            .expect("newer-generation replacement must survive");
        assert_eq!(
            survived.generation, 6,
            "evicted stale gen 5 instead of sparing gen 6"
        );
    }

    #[test]
    fn test_bounds_not_enforced_on_completion() {
        // If receive_chunk completes an assembly, the entry leaves the
        // map before bounds are checked — the just-completed value must
        // still be returned regardless of size.
        let asm = ChunkAssembler::with_limits(usize::MAX, 10);
        let out = asm
            .receive_chunk("peer", "k", 1, 0, 1, Bytes::from(vec![0u8; 500]))
            .expect("single-chunk completion returns bytes");
        assert_eq!(out.len(), 1, "single-chunk result is one Bytes fragment");
        assert_eq!(out[0].len(), 500);
        assert_eq!(asm.in_flight(), 0);
    }

    #[test]
    fn test_multiple_keys_independent() {
        let asm = ChunkAssembler::new();
        assert!(asm
            .receive_chunk("peer", "a", 1, 0, 2, Bytes::from_static(b"a0"))
            .is_none());
        assert!(asm
            .receive_chunk("peer", "b", 1, 0, 2, Bytes::from_static(b"b0"))
            .is_none());
        assert_eq!(asm.in_flight(), 2);

        let a = asm
            .receive_chunk("peer", "a", 1, 1, 2, Bytes::from_static(b"a1"))
            .unwrap();
        assert_eq!(flatten(&a), b"a0a1");
        assert_eq!(asm.in_flight(), 1, "completing a should not touch b");

        let b = asm
            .receive_chunk("peer", "b", 1, 1, 2, Bytes::from_static(b"b1"))
            .unwrap();
        assert_eq!(flatten(&b), b"b0b1");
        assert_eq!(asm.in_flight(), 0);
    }

    #[test]
    fn test_drop_pending_removes_only_matching_entry() {
        let asm = ChunkAssembler::new();
        // Start a multi-chunk assembly for (peer, "k") and an unrelated
        // one for (peer, "other") + (other_peer, "k").
        assert!(asm
            .receive_chunk("peer", "k", 1, 0, 2, Bytes::from_static(b"aa"))
            .is_none());
        assert!(asm
            .receive_chunk("peer", "other", 1, 0, 2, Bytes::from_static(b"bb"))
            .is_none());
        assert!(asm
            .receive_chunk("other_peer", "k", 1, 0, 2, Bytes::from_static(b"cc"))
            .is_none());
        assert_eq!(asm.in_flight(), 3);

        asm.drop_pending("peer", "k");
        assert_eq!(
            asm.in_flight(),
            2,
            "only (peer, k) should be gone; other entries spared"
        );

        // Completing the spared entries still works.
        let out = asm
            .receive_chunk("peer", "other", 1, 1, 2, Bytes::from_static(b"22"))
            .unwrap();
        assert_eq!(flatten(&out), b"bb22");
    }

    #[test]
    fn test_drop_pending_on_missing_key_is_noop() {
        let asm = ChunkAssembler::new();
        asm.drop_pending("peer", "nonexistent");
        assert_eq!(asm.in_flight(), 0);
    }
}
