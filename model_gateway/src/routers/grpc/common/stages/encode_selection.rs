//! Per-image encode-worker selection for the EPD pipeline.
//!
//! The encode stage fans a request's images out one Encode RPC per image
//! (Option A: each image is encoded independently and may land on a different
//! encode worker). This module decides which encode worker each image goes to.
//!
//! The default `CacheAffinity` policy pins each image to a stable worker by its
//! `content_hash`. Each encode worker keeps a process-local, per-worker embedding
//! cache keyed by image hash, so re-encoding the same image on the same worker is
//! a cache hit that skips the vision tower (the dominant GPU cost). Plain
//! round-robin spreads an image's repeats across the pool, dropping the hit rate
//! to roughly `1/N`; hash affinity lifts it back toward `1`. Uniform hashing also
//! spreads *distinct* images evenly across the pool on its own, so affinity is
//! already a balancer for the common case; the only concentration it creates is a
//! hot repeated image on its home worker, and that is cheap precisely because it
//! is a cache hit.
//!
//! There is deliberately no gateway-side load balancing. The gateway cannot
//! observe a worker's actual vision-tower backlog: the Encode RPC returns on
//! enqueue and the embedding ships out of band over Mooncake, so any
//! gateway-local load counter is a dispatch-burst proxy, not real backlog.
//! Feeding such a proxy into a rebalancer can route against true backlog and,
//! worse, shed a hot repeated image off the very worker that caches it (a cheap
//! cache hit turned into an expensive recompute). Real encode load balancing
//! needs an engine-reported signal (a future enhancement); until then we route
//! purely by affinity.

use std::sync::OnceLock;

use tracing::warn;

/// How the encode stage assigns each image's Encode RPC to a worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EncodeRoutingPolicy {
    /// Content-hash affinity: the same image always routes to the same worker
    /// (default), so its per-worker embedding cache hits.
    CacheAffinity,
    /// Plain round-robin across the encode pool. Escape hatch / A-B baseline.
    RoundRobin,
}

/// The configured encode routing policy (parsed once).
pub(crate) fn routing_policy() -> EncodeRoutingPolicy {
    static POLICY: OnceLock<EncodeRoutingPolicy> = OnceLock::new();
    *POLICY.get_or_init(|| match std::env::var("SMG_ENCODE_ROUTING_POLICY").as_deref() {
        Ok("round_robin") => EncodeRoutingPolicy::RoundRobin,
        Ok("cache_affinity") | Err(_) => EncodeRoutingPolicy::CacheAffinity,
        Ok(other) => {
            warn!(
                value = other,
                "unknown SMG_ENCODE_ROUTING_POLICY; falling back to cache_affinity"
            );
            EncodeRoutingPolicy::CacheAffinity
        }
    })
}

/// Stable affinity slot for a content hash over `n` candidates. `content_hash`
/// is already a uniform blake3 digest, so its leading bytes are taken directly
/// as the ring position; no re-hash needed. The candidate list must be in a
/// stable order (the caller sorts by URL) so the same image maps to the same
/// worker across requests for a fixed pool.
///
/// This is fixed-slot consistent hashing: on a pool resize most keys remap
/// (unlike a virtual-node ring), but EPD encode pools are fixed at launch, so a
/// resize only costs a one-time cache re-warm, never correctness. Returns 0 for
/// an empty pool or empty hash (deterministic; the caller guards the empty pool).
pub(crate) fn affinity_slot(content_hash: &[u8], n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut buf = [0u8; 8];
    let take = content_hash.len().min(8);
    buf[..take].copy_from_slice(&content_hash[..take]);
    (u64::from_le_bytes(buf) % n as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affinity_is_deterministic_for_same_hash() {
        let hash = [9u8, 8, 7, 6, 5, 4, 3, 2, 1, 0];
        let first = affinity_slot(&hash, 4);
        for _ in 0..20 {
            assert_eq!(affinity_slot(&hash, 4), first);
        }
    }

    #[test]
    fn different_hashes_spread_across_pool() {
        let mut seen = std::collections::HashSet::new();
        for i in 0u64..200 {
            seen.insert(affinity_slot(&i.to_le_bytes(), 4));
        }
        assert!(seen.len() > 1, "distinct hashes should not all collide");
    }

    #[test]
    fn slot_is_always_in_range() {
        for n in 1usize..=8 {
            for i in 0u64..256 {
                let slot = affinity_slot(&i.to_le_bytes(), n);
                assert!(slot < n, "slot {slot} out of range for n={n}");
            }
        }
    }

    #[test]
    fn single_candidate_is_slot_zero() {
        assert_eq!(affinity_slot(&[1, 2, 3], 1), 0);
    }

    #[test]
    fn empty_pool_is_slot_zero() {
        assert_eq!(affinity_slot(&[1, 2, 3], 0), 0);
    }

    #[test]
    fn zero_length_content_hash_is_safe() {
        // EPD images always carry a hash, but guard the empty case: maps to 0.
        assert_eq!(affinity_slot(&[], 4), 0);
    }
}
