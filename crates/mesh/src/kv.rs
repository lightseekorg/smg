//! MeshKV: Generic, application-agnostic mesh transport with explicit namespace handles.
//!
//! Provides two explicit namespace types:
//! - `CrdtNamespace` for durable, mergeable state (workers, policies, rate limits, config)
//! - `StreamNamespace` for ephemeral, lossy, application-regenerated traffic (tenant deltas, tree repair)
//!

use std::{collections::HashMap, sync::Arc, time::Instant};

use bytes::Bytes;
use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::sync::mpsc;

use crate::crdt_kv::CrdtOrMap;

// ============================================================================
// Type Definitions
// ============================================================================

/// Merge strategy for CRDT namespaces. Determines how conflicts are resolved
/// when two nodes write the same key concurrently.
#[derive(Debug, Clone)]
#[expect(clippy::enum_variant_names)]
pub enum MergeStrategy {
    /// Higher (version, replica_id) wins. Used for worker:*, policy:*, config:*.
    LastWriterWins,
    /// Higher numeric value wins (simple max). Reserved for future use.
    MaxValueWins,
    /// Compare epochs first, then max within same epoch.
    /// The mesh crate implements this internally — no application callback needed.
    /// Values MUST be exactly 16 bytes: epoch (u64 big-endian) + count (i64 big-endian).
    /// The adapter is responsible for serializing RateLimitValue to this fixed format.
    ///
    /// If either local or remote value is not exactly 16 bytes (corrupt/truncated message),
    /// the merge keeps the well-formed value. If both are malformed, keeps local.
    EpochMaxWins,
}

/// Routing mode for stream namespaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamRouting {
    /// Send to all connected peers (e.g., tenant deltas).
    Broadcast,
    /// Send to exactly one peer (e.g., tree repair requests/pages).
    Targeted,
}

/// Configuration for a stream namespace.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum bytes buffered before backpressure (oldest entries dropped).
    pub max_buffer_bytes: usize,
    /// Routing mode: broadcast to all peers or targeted to one peer.
    pub routing: StreamRouting,
}

/// Mode selection for a namespace prefix. Internal enum used during configuration.
#[derive(Debug)]
#[expect(dead_code)] // Fields read during later steps (gossip integration)
enum StoreMode {
    Crdt {
        merge_strategy: MergeStrategy,
    },
    Stream {
        max_buffer_bytes: usize,
        routing: StreamRouting,
    },
}

/// CRDT mode entry metadata.
#[derive(Debug, Clone)]
#[expect(dead_code)] // Constructed in later steps (CRDT merge integration)
pub(crate) struct ValueEntry {
    /// The actual data (opaque bytes — mesh doesn't interpret).
    pub value: Vec<u8>,
    /// Lamport timestamp (logical clock).
    pub version: u64,
    /// Deterministic hash of server_name, computed once at MeshKV startup.
    pub replica_id: u64,
    /// Soft-deleted? Tombstones propagate via gossip and win by higher version.
    pub tombstone: bool,
    /// Monotonic timestamp for tombstone GC. Tombstones younger than
    /// `tombstone_grace` (default 5 min) are not garbage collected.
    pub created_at: Instant,
}

// ============================================================================
// Subscription
// ============================================================================

/// A subscription to changes for keys matching a prefix.
/// Delivers via a bounded mpsc channel. Capacity is per-prefix:
/// - worker:  1000
/// - policy:  100
/// - rl:      100
/// - config:  100
/// - td:      50
/// - tree:req: 100
/// - tree:page: 32
pub struct Subscription {
    /// Receives (key, value) pairs. `None` value means the key was deleted.
    pub receiver: mpsc::Receiver<SubscriptionEvent>,
}

/// Function signature for stream drain callbacks. Called exactly once per gossip
/// round. Returns accumulated entries to be sent in this round's batch.
pub type StreamDrainFn = Box<dyn Fn() -> Vec<(String, Vec<u8>)> + Send + Sync>;

/// Handle returned by `register_drain`. Dropping unregisters the drain callback.
/// Use `drop(handle)` to explicitly unregister.
pub struct DrainHandle {
    prefix: String,
    drain_registry: Arc<DrainRegistry>,
}

impl Drop for DrainHandle {
    fn drop(&mut self) {
        self.drain_registry.remove(&self.prefix);
    }
}

// ============================================================================
// Subscriber Registry (internal)
// ============================================================================

/// A single subscription event: (key, value). `None` value means deletion.
type SubscriptionEvent = (String, Option<Bytes>);

/// Tracks all active subscriptions by prefix.
struct SubscriberRegistry {
    /// prefix -> list of senders. Multiple subscribers can watch the same prefix.
    subscribers: DashMap<String, Vec<mpsc::Sender<SubscriptionEvent>>>,
}

impl SubscriberRegistry {
    fn new() -> Self {
        Self {
            subscribers: DashMap::new(),
        }
    }

    fn register(&self, prefix: &str, tx: mpsc::Sender<SubscriptionEvent>) {
        self.subscribers
            .entry(prefix.to_string())
            .or_default()
            .push(tx);
    }

    /// Notify all subscribers whose prefix matches the given key.
    /// Uses try_send to never block the gossip loop.
    /// `value` is `Some(bytes)` for puts, `None` for deletes.
    fn notify(&self, key: &str, value: Option<Bytes>) {
        for entry in &self.subscribers {
            let prefix = entry.key();
            if key.starts_with(prefix.as_str()) {
                // try_send: never block. If full, entry is dropped.
                // For CRDT: watermark not advanced, resent next round.
                // For Stream: dropped permanently (ephemeral).
                for tx in entry.value() {
                    let _ = tx.try_send((key.to_string(), value.clone()));
                }
            }
        }
    }

    /// Remove closed senders individually. If all senders for a prefix are
    /// gone, remove the prefix entry entirely.
    #[expect(dead_code)] // Called by gossip GC cycle in later steps
    fn gc_closed(&self) {
        for mut entry in self.subscribers.iter_mut() {
            entry.value_mut().retain(|tx| !tx.is_closed());
        }
        // Atomically remove only if still empty, avoiding a race where
        // a concurrent register() adds a sender between iter_mut and remove.
        self.subscribers.retain(|_, senders| !senders.is_empty());
    }
}

// ============================================================================
// Drain Registry (internal)
// ============================================================================

/// Tracks registered drain callbacks for stream namespaces.
struct DrainRegistry {
    drains: DashMap<String, Arc<StreamDrainFn>>,
}

impl DrainRegistry {
    fn new() -> Self {
        Self {
            drains: DashMap::new(),
        }
    }

    fn register(&self, prefix: &str, drain: StreamDrainFn) {
        assert!(
            !self.drains.contains_key(prefix),
            "drain already registered for prefix '{prefix}'"
        );
        self.drains.insert(prefix.to_string(), Arc::new(drain));
    }

    fn remove(&self, prefix: &str) {
        self.drains.remove(prefix);
    }

    /// Call all registered drains. Returns accumulated entries.
    /// Called exactly once per gossip round.
    #[expect(dead_code)] // Called by centralized gossip loop in Step 2
    fn drain_all(&self) -> Vec<(String, Vec<u8>)> {
        let mut all_entries = Vec::new();
        for entry in &self.drains {
            let drain_fn = entry.value();
            all_entries.extend(drain_fn());
        }
        all_entries
    }
}

// ============================================================================
// Prefix Configuration (internal)
// ============================================================================

/// Per-prefix subscriber channel capacity.
fn subscriber_capacity_for_prefix(prefix: &str) -> usize {
    match prefix {
        "worker:" => 1000,
        "policy:" => 100,
        "rl:" => 100,
        "config:" => 100,
        "td:" => 50,
        "tree:req:" => 100,
        "tree:page:" => 32,
        _ => 100, // default for unknown prefixes
    }
}

// ============================================================================
// CrdtNamespace
// ============================================================================

/// Handle for durable, mergeable state. Scoped to a key prefix.
/// Provides put/get/delete/keys/subscribe.
pub struct CrdtNamespace {
    prefix: String,
    store: Arc<CrdtOrMap>,
    subscriber_registry: Arc<SubscriberRegistry>,
    merge_strategy: MergeStrategy,
}

impl CrdtNamespace {
    /// Insert or update a key-value pair. The key must start with this
    /// namespace's prefix.
    pub fn put(&self, key: &str, value: Vec<u8>) {
        assert!(
            key.starts_with(&self.prefix),
            "key '{key}' does not match prefix '{}'",
            self.prefix
        );
        self.store.insert(key.to_string(), value.clone());
        self.subscriber_registry
            .notify(key, Some(Bytes::from(value)));
    }

    /// Get the current value for a key, or None if not present or tombstoned.
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        assert!(
            key.starts_with(&self.prefix),
            "key '{key}' does not match prefix '{}'",
            self.prefix
        );
        self.store.get(key)
    }

    /// Delete a key by writing a tombstone. The tombstone propagates via gossip
    /// and wins by higher version. Actual removal happens after GC grace period.
    /// Subscribers receive `(key, None)` to indicate deletion.
    pub fn delete(&self, key: &str) {
        assert!(
            key.starts_with(&self.prefix),
            "key '{key}' does not match prefix '{}'",
            self.prefix
        );
        self.store.remove(key);
        self.subscriber_registry.notify(key, None);
    }

    /// List all live keys matching a sub-prefix within this namespace.
    pub fn keys(&self, sub_prefix: &str) -> Vec<String> {
        let full_prefix = format!("{}{}", self.prefix, sub_prefix);
        self.store
            .keys()
            .into_iter()
            .filter(|k| k.starts_with(&full_prefix))
            .collect()
    }

    /// Subscribe to changes for keys matching a sub-prefix within this namespace.
    /// Channel capacity is determined by the namespace prefix.
    pub fn subscribe(&self, sub_prefix: &str) -> Subscription {
        let full_prefix = format!("{}{}", self.prefix, sub_prefix);
        let capacity = subscriber_capacity_for_prefix(&self.prefix);
        let (tx, rx) = mpsc::channel(capacity);
        self.subscriber_registry.register(&full_prefix, tx);
        Subscription { receiver: rx }
    }

    /// Get the merge strategy for this namespace.
    pub fn merge_strategy(&self) -> &MergeStrategy {
        &self.merge_strategy
    }

    /// Get the prefix for this namespace.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

// ============================================================================
// StreamNamespace
// ============================================================================

/// Handle for ephemeral, lossy, application-regenerated traffic. Scoped to a
/// key prefix with a fixed routing mode (Broadcast or Targeted).
pub struct StreamNamespace {
    prefix: String,
    routing: StreamRouting,
    #[expect(dead_code)] // Used for backpressure enforcement in Step 2
    max_buffer_bytes: usize,
    /// Ephemeral send buffer, drained every gossip round.
    buffer: Arc<DashMap<String, Bytes>>,
    /// Targeted entries: (target_peer_id, key, value).
    targeted_buffer: Arc<parking_lot::Mutex<Vec<(String, String, Bytes)>>>,
    subscriber_registry: Arc<SubscriberRegistry>,
    drain_registry: Arc<DrainRegistry>,
}

impl StreamNamespace {
    /// Publish a value to all connected peers (Broadcast namespaces only).
    pub fn publish(&self, key: &str, value: Bytes) {
        assert_eq!(
            self.routing,
            StreamRouting::Broadcast,
            "publish() is only valid on Broadcast namespaces, not Targeted (prefix: '{}')",
            self.prefix
        );
        assert!(
            key.starts_with(&self.prefix),
            "key '{key}' does not match prefix '{}'",
            self.prefix
        );
        self.buffer.insert(key.to_string(), value);
    }

    /// Publish a value to exactly one peer (Targeted namespaces only).
    pub fn publish_to(&self, peer_id: &str, key: &str, value: Bytes) {
        assert_eq!(
            self.routing,
            StreamRouting::Targeted,
            "publish_to() is only valid on Targeted namespaces, not Broadcast (prefix: '{}')",
            self.prefix
        );
        assert!(
            key.starts_with(&self.prefix),
            "key '{key}' does not match prefix '{}'",
            self.prefix
        );
        self.targeted_buffer
            .lock()
            .push((peer_id.to_string(), key.to_string(), value));
    }

    /// Subscribe to messages for keys matching a sub-prefix within this namespace.
    pub fn subscribe(&self, sub_prefix: &str) -> Subscription {
        let full_prefix = format!("{}{}", self.prefix, sub_prefix);
        let capacity = subscriber_capacity_for_prefix(&self.prefix);
        let (tx, rx) = mpsc::channel(capacity);
        self.subscriber_registry.register(&full_prefix, tx);
        Subscription { receiver: rx }
    }

    /// Register a drain callback. Called exactly once per gossip round by the
    /// centralized collector. Returns a DrainHandle for
    /// unregistration.
    pub fn register_drain(&self, drain: StreamDrainFn) -> DrainHandle {
        self.drain_registry.register(&self.prefix, drain);
        DrainHandle {
            prefix: self.prefix.clone(),
            drain_registry: self.drain_registry.clone(),
        }
    }

    /// Get the routing mode for this namespace.
    pub fn routing(&self) -> StreamRouting {
        self.routing
    }

    /// Get the prefix for this namespace.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

// ============================================================================
// MeshKV
// ============================================================================

/// Generic, application-agnostic mesh transport. Provides explicit namespace
/// handles for CRDT and stream modes. Application code MUST use namespace
/// handles, not MeshKV directly.
pub struct MeshKV {
    /// CRDT store shared across all CRDT namespaces.
    store: Arc<CrdtOrMap>,
    /// Tracks configured prefixes to enforce fail-closed semantics.
    configured_prefixes: RwLock<HashMap<String, StoreMode>>,
    /// Shared subscriber registry.
    subscriber_registry: Arc<SubscriberRegistry>,
    /// Shared drain registry.
    drain_registry: Arc<DrainRegistry>,
    /// Server name for this node (used to derive replica_id).
    server_name: String,
    /// Replica ID: hash(server_name) as u64.
    replica_id: u64,
}

impl MeshKV {
    /// Create a new MeshKV instance.
    pub fn new(server_name: String) -> Self {
        let replica_id = Self::derive_replica_id(&server_name);
        Self {
            store: Arc::new(CrdtOrMap::new()),
            configured_prefixes: RwLock::new(HashMap::new()),
            subscriber_registry: Arc::new(SubscriberRegistry::new()),
            drain_registry: Arc::new(DrainRegistry::new()),
            server_name,
            replica_id,
        }
    }

    /// Derive replica_id from server_name using blake3 hash truncated to u64.
    /// Collision risk: 2^-64 per pair — negligible.
    fn derive_replica_id(server_name: &str) -> u64 {
        let hash = blake3::hash(server_name.as_bytes());
        // blake3::Hash::as_bytes() returns &[u8; 32], so first_chunk is infallible.
        let bytes: &[u8; 8] = hash.as_bytes().first_chunk().unwrap_or(&[0; 8]);
        u64::from_le_bytes(*bytes)
    }

    /// Configure a CRDT namespace for a key prefix. Returns a handle scoped to
    /// that prefix.
    ///
    /// # Panics
    /// Panics if the prefix is already configured (fail-closed).
    pub fn configure_crdt_prefix(
        &self,
        prefix: &str,
        merge_strategy: MergeStrategy,
    ) -> Arc<CrdtNamespace> {
        {
            let mut prefixes = self.configured_prefixes.write();
            assert!(
                !prefixes.contains_key(prefix),
                "Prefix '{prefix}' is already configured. Each prefix must be configured exactly once."
            );
            prefixes.insert(
                prefix.to_string(),
                StoreMode::Crdt {
                    merge_strategy: merge_strategy.clone(),
                },
            );
        }

        Arc::new(CrdtNamespace {
            prefix: prefix.to_string(),
            store: self.store.clone(),
            subscriber_registry: self.subscriber_registry.clone(),
            merge_strategy,
        })
    }

    /// Configure a stream namespace for a key prefix. Returns a handle scoped to
    /// that prefix.
    ///
    /// # Panics
    /// Panics if the prefix is already configured (fail-closed).
    pub fn configure_stream_prefix(
        &self,
        prefix: &str,
        config: StreamConfig,
    ) -> Arc<StreamNamespace> {
        {
            let mut prefixes = self.configured_prefixes.write();
            assert!(
                !prefixes.contains_key(prefix),
                "Prefix '{prefix}' is already configured. Each prefix must be configured exactly once."
            );
            prefixes.insert(
                prefix.to_string(),
                StoreMode::Stream {
                    max_buffer_bytes: config.max_buffer_bytes,
                    routing: config.routing,
                },
            );
        }

        Arc::new(StreamNamespace {
            prefix: prefix.to_string(),
            routing: config.routing,
            max_buffer_bytes: config.max_buffer_bytes,
            buffer: Arc::new(DashMap::new()),
            targeted_buffer: Arc::new(parking_lot::Mutex::new(Vec::new())),
            subscriber_registry: self.subscriber_registry.clone(),
            drain_registry: self.drain_registry.clone(),
        })
    }

    /// Get the server name for this node.
    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    /// Get the replica ID for this node.
    pub fn replica_id(&self) -> u64 {
        self.replica_id
    }

    /// Check if a prefix has been configured.
    pub fn is_prefix_configured(&self, prefix: &str) -> bool {
        self.configured_prefixes.read().contains_key(prefix)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_derive_replica_id_deterministic() {
        let id1 = MeshKV::derive_replica_id("gateway-1");
        let id2 = MeshKV::derive_replica_id("gateway-1");
        assert_eq!(id1, id2, "Same server_name must produce same replica_id");
    }

    #[test]
    fn test_derive_replica_id_different_names() {
        let id1 = MeshKV::derive_replica_id("gateway-1");
        let id2 = MeshKV::derive_replica_id("gateway-2");
        assert_ne!(
            id1, id2,
            "Different server_names should produce different replica_ids"
        );
    }

    #[test]
    fn test_configure_crdt_prefix() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
        assert_eq!(ns.prefix(), "worker:");
        assert!(kv.is_prefix_configured("worker:"));
    }

    #[test]
    fn test_configure_stream_prefix() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_stream_prefix(
            "td:",
            StreamConfig {
                max_buffer_bytes: 1024,
                routing: StreamRouting::Broadcast,
            },
        );
        assert_eq!(ns.prefix(), "td:");
        assert_eq!(ns.routing(), StreamRouting::Broadcast);
        assert!(kv.is_prefix_configured("td:"));
    }

    #[test]
    #[should_panic(expected = "already configured")]
    fn test_duplicate_prefix_panics() {
        let kv = MeshKV::new("test-node".to_string());
        kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
        kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins); // panics
    }

    #[test]
    #[should_panic(expected = "already configured")]
    fn test_duplicate_prefix_across_modes_panics() {
        let kv = MeshKV::new("test-node".to_string());
        kv.configure_crdt_prefix("data:", MergeStrategy::LastWriterWins);
        kv.configure_stream_prefix(
            "data:",
            StreamConfig {
                max_buffer_bytes: 1024,
                routing: StreamRouting::Broadcast,
            },
        ); // panics
    }

    #[test]
    fn test_crdt_put_get() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

        ns.put("worker:7", b"healthy".to_vec());
        let val = ns.get("worker:7");
        assert_eq!(val, Some(b"healthy".to_vec()));
    }

    #[test]
    fn test_crdt_delete() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

        ns.put("worker:7", b"healthy".to_vec());
        ns.delete("worker:7");
        assert_eq!(ns.get("worker:7"), None);
    }

    #[test]
    fn test_crdt_keys() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

        ns.put("worker:1", b"a".to_vec());
        ns.put("worker:2", b"b".to_vec());
        ns.put("worker:3", b"c".to_vec());

        let mut keys = ns.keys("");
        keys.sort();
        assert_eq!(keys, vec!["worker:1", "worker:2", "worker:3"]);
    }

    #[test]
    #[should_panic(expected = "does not match prefix")]
    fn test_crdt_put_wrong_prefix_panics() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
        ns.put("policy:1", b"wrong".to_vec()); // panics
    }

    #[test]
    #[should_panic(expected = "only valid on Broadcast")]
    fn test_stream_publish_on_targeted_panics() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_stream_prefix(
            "tree:req:",
            StreamConfig {
                max_buffer_bytes: 1024,
                routing: StreamRouting::Targeted,
            },
        );
        ns.publish("tree:req:model-x", Bytes::from("data")); // panics
    }

    #[test]
    #[should_panic(expected = "only valid on Targeted")]
    fn test_stream_publish_to_on_broadcast_panics() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_stream_prefix(
            "td:",
            StreamConfig {
                max_buffer_bytes: 1024,
                routing: StreamRouting::Broadcast,
            },
        );
        ns.publish_to("peer-1", "td:model-x", Bytes::from("data")); // panics
    }

    #[tokio::test]
    async fn test_crdt_subscribe() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

        let mut sub = ns.subscribe("");
        ns.put("worker:7", b"healthy".to_vec());

        let (key, value) = tokio::time::timeout(Duration::from_millis(100), sub.receiver.recv())
            .await
            .expect("timeout")
            .expect("channel closed");

        assert_eq!(key, "worker:7");
        assert_eq!(value.as_deref(), Some(b"healthy".as_slice()));
    }

    #[tokio::test]
    async fn test_crdt_subscribe_delete() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

        let mut sub = ns.subscribe("");
        ns.put("worker:7", b"healthy".to_vec());
        ns.delete("worker:7");

        // First event: put
        let (key, value) = tokio::time::timeout(Duration::from_millis(100), sub.receiver.recv())
            .await
            .expect("timeout")
            .expect("channel closed");
        assert_eq!(key, "worker:7");
        assert!(value.is_some());

        // Second event: delete
        let (key, value) = tokio::time::timeout(Duration::from_millis(100), sub.receiver.recv())
            .await
            .expect("timeout")
            .expect("channel closed");
        assert_eq!(key, "worker:7");
        assert!(value.is_none(), "delete should notify with None");
    }
}
