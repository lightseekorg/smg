use std::sync::Arc;

use dashmap::DashMap;

// ============================================================================
// High-Performance In-Memory KV Storage - Concurrent-Safe Implementation Based on DashMap
// ============================================================================

/// Basic KV storage, using DashMap for thread-safe high-performance access
#[derive(Debug, Clone)]
pub struct KvStore {
    store: Arc<DashMap<String, Vec<u8>>>,
}

impl KvStore {
    /// Create new KV storage
    pub fn new() -> Self {
        Self {
            store: Arc::new(DashMap::new()),
        }
    }

    /// Insert or update key-value pair
    pub fn insert(&self, key: String, value: Vec<u8>) -> Option<Vec<u8>> {
        self.store.insert(key, value)
    }

    /// Get value by key
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.store.get(key).map(|v| v.value().clone())
    }

    /// Remove key
    pub fn remove(&self, key: &str) -> Option<Vec<u8>> {
        self.store.remove(key).map(|(_, v)| v)
    }

    /// Check if key exists
    pub fn contains_key(&self, key: &str) -> bool {
        self.store.contains_key(key)
    }

    // /// Get number of key-value pairs
    // pub fn len(&self) -> usize {
    //     self.store.len()
    // }

    // /// Check if storage is empty
    // pub fn is_empty(&self) -> bool {
    //     self.store.is_empty()
    // }

    // /// Clear all data
    // pub fn clear(&self) {
    //     self.store.clear();
    // }

    // /// Get all keys
    // pub fn keys(&self) -> Vec<String> {
    //     self.store.iter().map(|entry| entry.key().clone()).collect()
    // }

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
