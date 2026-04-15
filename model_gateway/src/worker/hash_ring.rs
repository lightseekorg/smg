//! Consistent hash ring for O(log n) worker selection.
//!
//! The ring maps a routing key to a worker URL using consistent hashing over
//! virtual nodes. The registry rebuilds one ring per model when workers are
//! added or removed, so the build cost is amortized — individual lookups only
//! pay an `O(log n)` binary search plus a small bounded dedupe set to skip
//! virtual-node duplicates. See [`HashRing::find_healthy_url`] for details.
//!
//! The type intentionally has no dependency on the `Worker` trait — it is
//! constructed from URLs — so policies and tests can build rings without
//! materializing fake workers.

use std::{collections::HashSet, sync::Arc};

/// Number of virtual nodes per physical worker for even distribution.
/// 150 is a common choice that provides good balance between memory and distribution.
const VIRTUAL_NODES_PER_WORKER: usize = 150;

/// Consistent hash ring for O(log n) worker selection.
///
/// Each worker is placed at multiple positions (virtual nodes) on the ring
/// based on `hash(worker_url + vnode_index)`. This provides:
/// - Even key distribution across workers
/// - Minimal key redistribution when workers are added/removed (~1/N keys move)
/// - O(log n) lookup via binary search
///
/// Uses blake3 for stable, fast hashing that is consistent across Rust versions.
#[derive(Debug, Clone)]
pub struct HashRing {
    /// Sorted list of `(ring_position, worker_url)`.
    ///
    /// Multiple entries per worker (virtual nodes) for even distribution.
    /// Uses `Arc<str>` to share each URL across all of its virtual nodes
    /// (150 refs vs 150 copies).
    entries: Arc<[(u64, Arc<str>)]>,
}

impl HashRing {
    /// Build a hash ring from a collection of worker URLs.
    ///
    /// Creates `VIRTUAL_NODES_PER_WORKER` entries per URL for even distribution.
    /// Accepts any iterable of string-like items, so callers can pass the
    /// output of `workers.iter().map(|w| w.url())` without allocating a Vec.
    pub fn new<I>(urls: I) -> Self
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let iter = urls.into_iter();
        let (lower, _) = iter.size_hint();
        let mut entries: Vec<(u64, Arc<str>)> =
            Vec::with_capacity(lower.saturating_mul(VIRTUAL_NODES_PER_WORKER));

        for url in iter {
            let url: Arc<str> = Arc::from(url.as_ref());

            for vnode in 0..VIRTUAL_NODES_PER_WORKER {
                let vnode_key = format!("{url}#{vnode}");
                let pos = Self::hash_position(&vnode_key);
                entries.push((pos, Arc::clone(&url)));
            }
        }

        entries.sort_unstable_by_key(|(pos, _)| *pos);

        Self {
            entries: Arc::from(entries.into_boxed_slice()),
        }
    }

    /// Hash a string to a ring position using blake3 (stable across versions).
    #[inline]
    #[expect(
        clippy::expect_used,
        reason = "blake3 always produces 32 bytes — converting a fixed 8-byte slice to [u8; 8] is infallible"
    )]
    fn hash_position(s: &str) -> u64 {
        let hash = blake3::hash(s.as_bytes());
        u64::from_le_bytes(
            hash.as_bytes()[..8]
                .try_into()
                .expect("blake3 hash is always 32 bytes, slicing first 8 is infallible"),
        )
    }

    /// Find a worker URL for a key using consistent hashing.
    ///
    /// Returns the first healthy worker URL at or after the key's position
    /// (clockwise). Skips virtual nodes for workers already checked.
    ///
    /// Cost per call: `O(log n)` binary search to find the start position
    /// plus one small `HashSet` allocation bounded by
    /// `min(worker_count(), 16)` slots to dedupe virtual-node hits while
    /// walking clockwise. The dedupe set is dropped before return.
    ///
    /// - `key`: The routing key to hash
    /// - `is_healthy`: Function to check if a worker URL is healthy
    pub fn find_healthy_url<F>(&self, key: &str, is_healthy: F) -> Option<&str>
    where
        F: Fn(&str) -> bool,
    {
        if self.entries.is_empty() {
            return None;
        }

        let key_pos = Self::hash_position(key);

        let start = self.entries.partition_point(|(pos, _)| *pos < key_pos);

        // Walk clockwise from start, wrapping around. Track visited URLs to
        // avoid calling `is_healthy` multiple times for the same worker when
        // we hit its virtual nodes. Capacity is bounded by the physical worker
        // count — typically a handful of entries — so the per-lookup
        // allocation is negligible relative to the hashing itself.
        let mut checked_urls = HashSet::with_capacity(self.worker_count().min(16));

        for i in 0..self.entries.len() {
            let (_, url) = &self.entries[(start + i) % self.entries.len()];
            let url_str: &str = url;

            if !checked_urls.insert(url_str) {
                continue;
            }

            if is_healthy(url_str) {
                return Some(url_str);
            }
        }

        None
    }

    /// Check if the ring is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the number of entries in the ring (including virtual nodes).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Get the number of unique workers in the ring.
    pub fn worker_count(&self) -> usize {
        self.entries.len() / VIRTUAL_NODES_PER_WORKER.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_ring_returns_none() {
        let ring = HashRing::new(std::iter::empty::<&str>());
        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);
        assert_eq!(ring.worker_count(), 0);
        assert_eq!(ring.find_healthy_url("any-key", |_| true), None);
    }

    #[test]
    fn len_scales_with_virtual_nodes() {
        let ring = HashRing::new(["http://a", "http://b", "http://c"]);
        assert!(!ring.is_empty());
        assert_eq!(ring.len(), 3 * VIRTUAL_NODES_PER_WORKER);
        assert_eq!(ring.worker_count(), 3);
    }

    #[test]
    fn find_healthy_url_is_deterministic() {
        let ring = HashRing::new(["http://a", "http://b", "http://c"]);
        let first = ring.find_healthy_url("routing-key", |_| true).unwrap();
        for _ in 0..10 {
            assert_eq!(ring.find_healthy_url("routing-key", |_| true), Some(first));
        }
    }

    #[test]
    fn find_healthy_url_skips_unhealthy() {
        let ring = HashRing::new(["http://a", "http://b", "http://c"]);
        let picked = ring.find_healthy_url("routing-key", |url| url != "http://a");
        assert!(matches!(picked, Some("http://b") | Some("http://c")));
    }

    #[test]
    fn find_healthy_url_returns_none_when_all_unhealthy() {
        let ring = HashRing::new(["http://a", "http://b"]);
        assert_eq!(ring.find_healthy_url("k", |_| false), None);
    }

    #[test]
    fn accepts_owned_string_iterators() {
        let urls = vec!["http://a".to_string(), "http://b".to_string()];
        let ring = HashRing::new(urls);
        assert_eq!(ring.worker_count(), 2);
    }
}
