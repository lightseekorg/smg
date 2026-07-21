use std::collections::HashSet;

use dashmap::DashMap;
use mini_moka::sync::Cache;

pub struct PrefixCache {
    inner: Cache<u64, String>,
    reverse: DashMap<String, HashSet<u64>>,
}

impl PrefixCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Cache::builder().max_capacity(capacity as u64).build(),
            reverse: DashMap::new(),
        }
    }

    pub fn insert(&self, prefix_hash: u64, worker_id: String) {
        self.reverse
            .entry(worker_id.clone())
            .or_default()
            .insert(prefix_hash);

        self.inner.insert(prefix_hash, worker_id);
    }

    pub fn lookup(&self, prefix_hash: u64) -> Option<String> {
        self.inner.get(&prefix_hash)
    }

    pub fn remove_worker(&self, worker_id: &str) {
        if let Some((_, hashes)) = self.reverse.remove(worker_id) {
            for hash in hashes {
                self.inner.invalidate(&hash);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.inner.entry_count() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_lookup() {
        let cache = PrefixCache::new(10);
        cache.insert(0x1234, "worker-a".to_string());
        assert_eq!(cache.lookup(0x1234), Some("worker-a".to_string()));
        assert_eq!(cache.lookup(0x5678), None);
    }

    #[test]
    fn remove_worker() {
        let cache = PrefixCache::new(10);
        cache.insert(0x1234, "worker-a".to_string());
        cache.insert(0x5678, "worker-a".to_string());
        cache.insert(0xABCD, "worker-b".to_string());

        cache.remove_worker("worker-a");
        assert_eq!(cache.lookup(0x1234), None);
        assert_eq!(cache.lookup(0x5678), None);
        assert_eq!(cache.lookup(0xABCD), Some("worker-b".to_string()));
    }

    #[test]
    fn overwrite_cleans_reverse() {
        let cache = PrefixCache::new(10);
        cache.insert(0x1234, "worker-a".to_string());
        cache.insert(0x1234, "worker-b".to_string());
        cache.remove_worker("worker-a");
        assert_eq!(cache.lookup(0x1234), Some("worker-b".to_string()));
    }

    #[test]
    fn lru_eviction() {
        let cache = PrefixCache::new(3);
        cache.insert(1, "w1".to_string());
        cache.insert(2, "w2".to_string());
        cache.insert(3, "w3".to_string());
        assert_eq!(cache.len(), 3);

        cache.insert(4, "w4".to_string());
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.lookup(1), None);
        assert_eq!(cache.lookup(2), Some("w2".to_string()));
        assert_eq!(cache.lookup(3), Some("w3".to_string()));
        assert_eq!(cache.lookup(4), Some("w4".to_string()));
    }
}
