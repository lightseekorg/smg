use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use dashmap::{mapref::entry::Entry, DashMap};

// ============================================================================
// High-Performance In-Memory KV Storage - Concurrent-Safe Implementation Based on DashMap
// ============================================================================

/// Basic KV storage, using DashMap for thread-safe high-performance access.
///
/// Includes a generation counter that increments on every mutation. This allows
/// the incremental update collector to skip expensive full-store scans when
/// nothing has changed.
#[derive(Debug, Clone)]
pub struct KvStore {
    store: Arc<DashMap<String, Vec<u8>>>,
    /// Monotonically increasing counter, bumped on every insert/remove/upsert.
    generation: Arc<AtomicU64>,
}

impl KvStore {
    /// Create new KV storage
    pub fn new() -> Self {
        Self {
            store: Arc::new(DashMap::new()),
            generation: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Current generation (mutation counter).
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Insert or update key-value pair
    pub fn insert(&self, key: String, value: Vec<u8>) -> Option<Vec<u8>> {
        self.generation.fetch_add(1, Ordering::Release);
        self.store.insert(key, value)
    }

    /// Atomically compute and update a key in a single DashMap entry operation.
    pub fn upsert<F>(&self, key: String, updater: F) -> Vec<u8>
    where
        F: FnOnce(Option<&[u8]>) -> Vec<u8>,
    {
        self.generation.fetch_add(1, Ordering::Release);
        match self.store.entry(key) {
            Entry::Occupied(mut entry) => {
                let new_value = updater(Some(entry.get().as_slice()));
                entry.get_mut().clone_from(&new_value);
                new_value
            }
            Entry::Vacant(entry) => {
                let new_value = updater(None);
                entry.insert(new_value.clone());
                new_value
            }
        }
    }

    /// Get value by key
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.store.get(key).map(|v| v.value().clone())
    }

    /// Remove key
    pub fn remove(&self, key: &str) -> Option<Vec<u8>> {
        self.generation.fetch_add(1, Ordering::Release);
        self.store.remove(key).map(|(_, v)| v)
    }

    /// Check if key exists
    pub fn contains_key(&self, key: &str) -> bool {
        self.store.contains_key(key)
    }

    /// Number of keys in the store.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Get all keys without cloning values.
    pub fn keys(&self) -> Vec<String> {
        self.store.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Get all key-value pairs as a BTreeMap
    pub fn all(&self) -> std::collections::BTreeMap<String, Vec<u8>> {
        self.store
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }
}

impl Default for KvStore {
    fn default() -> Self {
        Self::new()
    }
}
