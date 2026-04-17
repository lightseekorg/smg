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

use std::time::{Duration, Instant};

use dashmap::DashMap;

pub struct ChunkAssembler {
    assemblies: DashMap<String, AssemblyState>,
}

struct AssemblyState {
    generation: u64,
    received: Vec<bool>,
    chunks: Vec<Option<Vec<u8>>>,
    created_at: Instant,
}

impl AssemblyState {
    fn new(generation: u64, total: u32) -> Self {
        let n = total as usize;
        Self {
            generation,
            received: vec![false; n],
            chunks: vec![None; n],
            created_at: Instant::now(),
        }
    }

    fn is_complete(&self) -> bool {
        self.received.iter().all(|&r| r)
    }

    fn assemble(&self) -> Vec<u8> {
        let total_size: usize = self.chunks.iter().flatten().map(|c| c.len()).sum();
        let mut out = Vec::with_capacity(total_size);
        for chunk in self.chunks.iter().flatten() {
            out.extend_from_slice(chunk);
        }
        out
    }
}

impl ChunkAssembler {
    pub fn new() -> Self {
        Self {
            assemblies: DashMap::new(),
        }
    }

    /// Record an incoming chunk. Returns `Some(assembled)` once all chunks
    /// for the current generation have arrived; returns `None` otherwise.
    ///
    /// A chunk whose `generation` differs from the in-flight state resets
    /// the assembly — older-generation partials are discarded in favour
    /// of the new version. Malformed chunks (total == 0, index >= total)
    /// are dropped silently.
    pub fn receive_chunk(
        &self,
        key: &str,
        generation: u64,
        index: u32,
        total: u32,
        data: Vec<u8>,
    ) -> Option<Vec<u8>> {
        if total == 0 || index >= total {
            return None;
        }

        let assembled = {
            let mut entry = self
                .assemblies
                .entry(key.to_string())
                .or_insert_with(|| AssemblyState::new(generation, total));

            if entry.generation != generation || entry.chunks.len() != total as usize {
                *entry = AssemblyState::new(generation, total);
            }

            entry.received[index as usize] = true;
            entry.chunks[index as usize] = Some(data);

            if entry.is_complete() {
                Some(entry.assemble())
            } else {
                None
            }
        };

        if assembled.is_some() {
            self.assemblies.remove(key);
        }
        assembled
    }

    /// Drop partial assemblies older than `timeout`. Collect-then-remove
    /// to avoid holding DashMap shard locks during mutation.
    pub fn gc(&self, timeout: Duration) {
        let expired: Vec<String> = self
            .assemblies
            .iter()
            .filter(|e| e.value().created_at.elapsed() >= timeout)
            .map(|e| e.key().clone())
            .collect();
        for key in expired {
            self.assemblies.remove(&key);
        }
    }

    #[cfg(test)]
    pub(crate) fn in_flight(&self) -> usize {
        self.assemblies.len()
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
    fn test_malformed_chunk_is_dropped() {
        let asm = ChunkAssembler::new();
        assert!(asm.receive_chunk("k", 1, 0, 0, b"x".to_vec()).is_none());
        assert!(asm.receive_chunk("k", 1, 5, 3, b"x".to_vec()).is_none());
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
