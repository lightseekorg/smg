//! Tree operation definitions for mesh synchronization
//!
//! Defines serializable tree operations that can be synchronized across mesh cluster nodes

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TreeKey {
    Text(String),
    Tokens(Vec<u32>),
}

/// Tree insert operation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TreeInsertOp {
    pub key: TreeKey,
    pub tenant: String, // worker URL
}

/// Tree remove operation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TreeRemoveOp {
    pub tenant: String, // worker URL
}

/// Tree operation type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TreeOperation {
    Insert(TreeInsertOp),
    Remove(TreeRemoveOp),
}

/// Delta encoding for tree state synchronization.
/// Contains only the new operations since the last successful sync, rather than the full tree state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TreeStateDelta {
    pub model_id: String,
    pub operations: Vec<TreeOperation>,
    /// Tree state version before these operations were applied.
    pub base_version: u64,
    /// Tree state version after these operations are applied.
    pub new_version: u64,
}

impl TreeStateDelta {
    /// Serialize to bincode.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| format!("Failed to serialize TreeStateDelta: {e}"))
    }

    /// Deserialize from bincode bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        bincode::deserialize(bytes)
            .map_err(|e| format!("Failed to deserialize TreeStateDelta: {e}"))
    }
}

// ── Tenant delta types for efficient two-layer sync ─────────────────

/// Lightweight tenant change set for high-frequency sync (every gossip round).
/// Contains only which tenants changed at which tree nodes — no tree structure,
/// no prompt text. ~100 bytes per insert vs ~200KB for full TreeOperation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TenantDelta {
    pub model_id: String,
    pub version: u64,
    pub inserts: Vec<TenantInsert>,
    pub evictions: Vec<TenantEvict>,
}

/// A tenant was added or refreshed at a tree node.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TenantInsert {
    /// Blake3 hash of the full prefix path from tree root to this node.
    /// 8 bytes instead of 80k+ chars. Receiver looks up node by hash;
    /// if unknown, buffers until next structure snapshot.
    pub node_path_hash: u64,
    /// Worker URL that cached this prefix.
    pub worker_url: String,
    /// Epoch (timestamp) of the cache event. Max-epoch-wins on merge.
    pub epoch: u64,
}

pub use crate::hash::{hash_node_path, hash_token_path, GLOBAL_EVICTION_HASH};

/// A tenant was evicted from a tree node.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TenantEvict {
    /// Blake3 hash of the prefix path where the tenant was evicted.
    /// Use [`GLOBAL_EVICTION_HASH`] (0) to evict from all nodes.
    pub node_path_hash: u64,
    /// Worker URL that evicted this prefix.
    pub worker_url: String,
}

impl TenantDelta {
    pub fn new(model_id: String, version: u64) -> Self {
        Self {
            model_id,
            version,
            inserts: Vec::new(),
            evictions: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inserts.is_empty() && self.evictions.is_empty()
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| format!("Failed to serialize TenantDelta: {e}"))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        bincode::deserialize(bytes).map_err(|e| format!("Failed to deserialize TenantDelta: {e}"))
    }
}

// ── Compression helpers for structure snapshots ─────────────────────

/// Compress bytes with LZ4 for wire efficiency.
/// Radix tree data compresses well (repetitive edge labels, worker URLs).
pub fn lz4_compress(data: &[u8]) -> Vec<u8> {
    lz4_flex::compress_prepend_size(data)
}

/// Decompress LZ4-compressed bytes with a size safety check.
/// Rejects payloads claiming > 256 MB decompressed size to prevent
/// OOM from corrupted or malicious size headers.
pub fn lz4_decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    const MAX_DECOMPRESSED_SIZE: usize = 256 * 1024 * 1024; // 256 MB
    if data.len() >= 4 {
        let claimed_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if claimed_size > MAX_DECOMPRESSED_SIZE {
            return Err(format!(
                "LZ4 claimed decompressed size {claimed_size} exceeds limit {MAX_DECOMPRESSED_SIZE}"
            ));
        }
    }
    lz4_flex::decompress_size_prepended(data).map_err(|e| format!("LZ4 decompression failed: {e}"))
}

// ── Legacy types (still used for periodic structure snapshots) ───────

/// Maximum number of operations stored in a TreeState before compaction.
/// Prevents unbounded growth of the operation log, especially with token payloads.
const MAX_TREE_OPERATIONS: usize = 2048;

/// Tree state for a specific model
/// Contains a sequence of operations that can be applied to reconstruct the tree
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub struct TreeState {
    pub model_id: String,
    pub operations: Vec<TreeOperation>,
    pub version: u64,
}

impl TreeState {
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            operations: Vec::new(),
            version: 0,
        }
    }

    pub fn add_operation(&mut self, operation: TreeOperation) {
        self.operations.push(operation);
        self.version += 1;
        if self.operations.len() > MAX_TREE_OPERATIONS {
            // Keep the most recent half — oldest operations are least relevant for routing
            let drain_count = self.operations.len() - MAX_TREE_OPERATIONS / 2;
            self.operations.drain(..drain_count);
        }
    }

    /// Serialize to bincode (compact binary format).
    /// A Vec<u32> of 1000 tokens is ~4KB in bincode vs ~7KB in JSON.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| format!("Failed to serialize TreeState: {e}"))
    }

    /// Deserialize from bincode bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        bincode::deserialize(bytes).map_err(|e| format!("Failed to deserialize TreeState: {e}"))
    }

    /// Reconstruct a `TreeState` from a compact [`kv_index::snapshot::TreeSnapshot`].
    ///
    /// Walks the pre-order node list, rebuilding full prefix paths and emitting
    /// an `Insert` operation for each `(tenant, prefix)` pair. This is the
    /// inverse of [`CacheAwarePolicy::export_tree_state`] and is used on the
    /// receiver side to convert compact snapshots back into the `TreeState`
    /// format that `apply_remote_tree_operation` expects.
    #[expect(
        clippy::unwrap_used,
        reason = "pop() after last_mut().is_some() is infallible"
    )]
    pub fn from_snapshot(
        model_id: String,
        snapshot: &kv_index::snapshot::TreeSnapshot,
        version: u64,
    ) -> Self {
        let mut tree_state = Self::new(model_id);
        let mut path_stack: Vec<(String, u32)> = Vec::new();
        let mut current_prefix = String::new();

        for node in &snapshot.nodes {
            // Pop completed parents from the stack
            while let Some((_, remaining)) = path_stack.last_mut() {
                if *remaining == 0 {
                    let (parent_prefix, _) = path_stack.pop().unwrap();
                    current_prefix = parent_prefix;
                } else {
                    *remaining -= 1;
                    break;
                }
            }

            // Build this node's full prefix
            let node_prefix = format!("{}{}", current_prefix, node.edge);

            // Emit an Insert operation for each tenant at this node
            for (tenant_url, _epoch) in &node.tenants {
                if !node_prefix.is_empty() {
                    tree_state.add_operation(TreeOperation::Insert(TreeInsertOp {
                        key: TreeKey::Text(node_prefix.clone()),
                        tenant: tenant_url.clone(),
                    }));
                }
            }

            // Push this node onto the stack for its children
            if node.child_count > 0 {
                path_stack.push((current_prefix.clone(), node.child_count));
                current_prefix = node_prefix;
            }
        }

        tree_state.version = version;
        tree_state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_insert_op_creation() {
        let op = TreeInsertOp {
            key: TreeKey::Text("test_text".to_string()),
            tenant: "http://worker1:8000".to_string(),
        };
        assert_eq!(op.key, TreeKey::Text("test_text".to_string()));
        assert_eq!(op.tenant, "http://worker1:8000");
    }

    #[test]
    fn test_tree_remove_op_creation() {
        let op = TreeRemoveOp {
            tenant: "http://worker1:8000".to_string(),
        };
        assert_eq!(op.tenant, "http://worker1:8000");
    }

    #[test]
    fn test_tree_operation_insert() {
        let insert_op = TreeInsertOp {
            key: TreeKey::Text("test_text".to_string()),
            tenant: "http://worker1:8000".to_string(),
        };
        let operation = TreeOperation::Insert(insert_op.clone());

        match &operation {
            TreeOperation::Insert(op) => {
                assert_eq!(op.key, TreeKey::Text("test_text".to_string()));
                assert_eq!(op.tenant, "http://worker1:8000");
            }
            TreeOperation::Remove(_) => panic!("Expected Insert operation"),
        }
    }

    #[test]
    fn test_tree_operation_remove() {
        let remove_op = TreeRemoveOp {
            tenant: "http://worker1:8000".to_string(),
        };
        let operation = TreeOperation::Remove(remove_op.clone());

        match &operation {
            TreeOperation::Insert(_) => panic!("Expected Remove operation"),
            TreeOperation::Remove(op) => {
                assert_eq!(op.tenant, "http://worker1:8000");
            }
        }
    }

    #[test]
    fn test_tree_operation_serialization() {
        let insert_op = TreeInsertOp {
            key: TreeKey::Text("test_text".to_string()),
            tenant: "http://worker1:8000".to_string(),
        };
        let operation = TreeOperation::Insert(insert_op);

        let serialized = serde_json::to_string(&operation).unwrap();
        let deserialized: TreeOperation = serde_json::from_str(&serialized).unwrap();

        match (&operation, &deserialized) {
            (TreeOperation::Insert(a), TreeOperation::Insert(b)) => {
                assert_eq!(a.key, b.key);
                assert_eq!(a.tenant, b.tenant);
            }
            _ => panic!("Operations should match"),
        }
    }

    #[test]
    fn test_tree_operation_token_serialization() {
        let insert_op = TreeInsertOp {
            key: TreeKey::Tokens(vec![1, 2, 3, 4]),
            tenant: "http://worker1:8000".to_string(),
        };
        let operation = TreeOperation::Insert(insert_op);

        let serialized = serde_json::to_string(&operation).unwrap();
        let deserialized: TreeOperation = serde_json::from_str(&serialized).unwrap();

        match (&operation, &deserialized) {
            (TreeOperation::Insert(a), TreeOperation::Insert(b)) => {
                assert_eq!(a.key, b.key);
                assert_eq!(a.tenant, b.tenant);
            }
            _ => panic!("Operations should match"),
        }
    }

    #[test]
    fn test_tree_state_bincode_round_trip_with_tokens() {
        let tokens = vec![12345u32, 67890, 0, u32::MAX, 42];
        let mut state = TreeState::new("test-model".to_string());
        state.add_operation(TreeOperation::Insert(TreeInsertOp {
            key: TreeKey::Tokens(tokens.clone()),
            tenant: "http://worker1:8000".to_string(),
        }));
        state.add_operation(TreeOperation::Insert(TreeInsertOp {
            key: TreeKey::Text("text_key".to_string()),
            tenant: "http://worker2:8000".to_string(),
        }));
        state.add_operation(TreeOperation::Remove(TreeRemoveOp {
            tenant: "http://worker3:8000".to_string(),
        }));

        let bytes = state.to_bytes().unwrap();
        let restored = TreeState::from_bytes(&bytes).unwrap();

        assert_eq!(restored.model_id, "test-model");
        assert_eq!(restored.version, state.version);
        assert_eq!(restored.operations.len(), 3);

        match &restored.operations[0] {
            TreeOperation::Insert(op) => {
                assert_eq!(op.key, TreeKey::Tokens(tokens));
                assert_eq!(op.tenant, "http://worker1:8000");
            }
            TreeOperation::Remove(_) => panic!("Expected Insert"),
        }
        match &restored.operations[1] {
            TreeOperation::Insert(op) => {
                assert_eq!(op.key, TreeKey::Text("text_key".to_string()));
            }
            TreeOperation::Remove(_) => panic!("Expected Insert"),
        }
        match &restored.operations[2] {
            TreeOperation::Remove(op) => {
                assert_eq!(op.tenant, "http://worker3:8000");
            }
            TreeOperation::Insert(_) => panic!("Expected Remove"),
        }
    }

    #[test]
    fn test_tree_state_bincode_round_trip_large_tokens() {
        let mut state = TreeState::new("large-model".to_string());
        for i in 0..100 {
            let tokens: Vec<u32> = (0..1000).map(|j| (i * 1000 + j) as u32).collect();
            state.add_operation(TreeOperation::Insert(TreeInsertOp {
                key: TreeKey::Tokens(tokens),
                tenant: format!("http://worker-{i}:8000"),
            }));
        }

        let bytes = state.to_bytes().unwrap();
        let restored = TreeState::from_bytes(&bytes).unwrap();

        assert_eq!(restored.operations.len(), 100);
        assert_eq!(restored.version, state.version);

        // Spot-check exact token preservation
        match &restored.operations[0] {
            TreeOperation::Insert(op) => {
                if let TreeKey::Tokens(tokens) = &op.key {
                    assert_eq!(tokens.len(), 1000);
                    assert_eq!(tokens[0], 0);
                    assert_eq!(tokens[999], 999);
                } else {
                    panic!("Expected Tokens key");
                }
            }
            TreeOperation::Remove(_) => panic!("Expected Insert"),
        }
        match &restored.operations[99] {
            TreeOperation::Insert(op) => {
                if let TreeKey::Tokens(tokens) = &op.key {
                    assert_eq!(tokens[0], 99000);
                    assert_eq!(tokens[999], 99999);
                } else {
                    panic!("Expected Tokens key");
                }
            }
            TreeOperation::Remove(_) => panic!("Expected Insert"),
        }
    }

    #[test]
    fn test_tree_operation_remove_serialization() {
        let remove_op = TreeRemoveOp {
            tenant: "http://worker1:8000".to_string(),
        };
        let operation = TreeOperation::Remove(remove_op);

        let serialized = serde_json::to_string(&operation).unwrap();
        let deserialized: TreeOperation = serde_json::from_str(&serialized).unwrap();

        match (&operation, &deserialized) {
            (TreeOperation::Remove(a), TreeOperation::Remove(b)) => {
                assert_eq!(a.tenant, b.tenant);
            }
            _ => panic!("Operations should match"),
        }
    }

    #[test]
    fn test_tree_state_new() {
        let state = TreeState::new("model1".to_string());
        assert_eq!(state.model_id, "model1");
        assert_eq!(state.operations.len(), 0);
        assert_eq!(state.version, 0);
    }

    #[test]
    fn test_tree_state_default() {
        let state = TreeState::default();
        assert_eq!(state.model_id, "");
        assert_eq!(state.operations.len(), 0);
        assert_eq!(state.version, 0);
    }

    #[test]
    fn test_tree_state_add_operation() {
        let mut state = TreeState::new("model1".to_string());

        let insert_op = TreeInsertOp {
            key: TreeKey::Text("text1".to_string()),
            tenant: "http://worker1:8000".to_string(),
        };
        state.add_operation(TreeOperation::Insert(insert_op));

        assert_eq!(state.operations.len(), 1);
        assert_eq!(state.version, 1);

        let remove_op = TreeRemoveOp {
            tenant: "http://worker1:8000".to_string(),
        };
        state.add_operation(TreeOperation::Remove(remove_op));

        assert_eq!(state.operations.len(), 2);
        assert_eq!(state.version, 2);
    }

    #[test]
    fn test_tree_state_add_multiple_operations() {
        let mut state = TreeState::new("model1".to_string());

        for i in 0..5 {
            let insert_op = TreeInsertOp {
                key: TreeKey::Text(format!("text_{i}")),
                tenant: format!("http://worker{i}:8000"),
            };
            state.add_operation(TreeOperation::Insert(insert_op));
        }

        assert_eq!(state.operations.len(), 5);
        assert_eq!(state.version, 5);
    }

    #[test]
    fn test_tree_state_serialization() {
        let mut state = TreeState::new("model1".to_string());

        let insert_op = TreeInsertOp {
            key: TreeKey::Text("test_text".to_string()),
            tenant: "http://worker1:8000".to_string(),
        };
        state.add_operation(TreeOperation::Insert(insert_op));

        let remove_op = TreeRemoveOp {
            tenant: "http://worker1:8000".to_string(),
        };
        state.add_operation(TreeOperation::Remove(remove_op));

        let serialized = serde_json::to_string(&state).unwrap();
        let deserialized: TreeState = serde_json::from_str(&serialized).unwrap();

        assert_eq!(state.model_id, deserialized.model_id);
        assert_eq!(state.operations.len(), deserialized.operations.len());
        assert_eq!(state.version, deserialized.version);
    }

    #[test]
    fn test_tree_state_clone() {
        let mut state = TreeState::new("model1".to_string());

        let insert_op = TreeInsertOp {
            key: TreeKey::Text("test_text".to_string()),
            tenant: "http://worker1:8000".to_string(),
        };
        state.add_operation(TreeOperation::Insert(insert_op));

        let cloned = state.clone();
        assert_eq!(state.model_id, cloned.model_id);
        assert_eq!(state.operations.len(), cloned.operations.len());
        assert_eq!(state.version, cloned.version);
    }

    #[test]
    fn test_tree_state_equality() {
        let mut state1 = TreeState::new("model1".to_string());
        let mut state2 = TreeState::new("model1".to_string());

        let insert_op = TreeInsertOp {
            key: TreeKey::Text("test_text".to_string()),
            tenant: "http://worker1:8000".to_string(),
        };
        state1.add_operation(TreeOperation::Insert(insert_op.clone()));
        state2.add_operation(TreeOperation::Insert(insert_op));

        assert_eq!(state1, state2);
    }

    #[test]
    fn test_tree_operation_hash() {
        use std::collections::HashSet;

        let insert_op1 = TreeInsertOp {
            key: TreeKey::Text("text1".to_string()),
            tenant: "http://worker1:8000".to_string(),
        };
        let insert_op2 = TreeInsertOp {
            key: TreeKey::Text("text1".to_string()),
            tenant: "http://worker1:8000".to_string(),
        };

        let op1 = TreeOperation::Insert(insert_op1);
        let op2 = TreeOperation::Insert(insert_op2);

        let mut set = HashSet::new();
        set.insert(op1.clone());
        set.insert(op2.clone());

        // Same operations should be considered equal
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_tenant_delta_round_trip() {
        let path_hash = hash_node_path("Hello world, how are");
        let mut delta = TenantDelta::new("model1".to_string(), 42);
        delta.inserts.push(TenantInsert {
            node_path_hash: path_hash,
            worker_url: "grpc://w1:8000".to_string(),
            epoch: 1000,
        });
        delta.evictions.push(TenantEvict {
            node_path_hash: path_hash,
            worker_url: "grpc://w2:8000".to_string(),
        });

        assert!(!delta.is_empty());

        let bytes = delta.to_bytes().unwrap();
        let restored = TenantDelta::from_bytes(&bytes).unwrap();

        assert_eq!(restored.model_id, "model1");
        assert_eq!(restored.version, 42);
        assert_eq!(restored.inserts.len(), 1);
        assert_eq!(restored.inserts[0].worker_url, "grpc://w1:8000");
        assert_eq!(restored.inserts[0].node_path_hash, path_hash);
        assert_eq!(restored.inserts[0].epoch, 1000);
        assert_eq!(restored.evictions.len(), 1);
        assert_eq!(restored.evictions[0].worker_url, "grpc://w2:8000");
    }

    #[test]
    fn test_tenant_delta_empty() {
        let delta = TenantDelta::new("model1".to_string(), 0);
        assert!(delta.is_empty());
    }

    #[test]
    fn test_tenant_delta_size_vs_tree_operation() {
        // A TenantInsert with a hash is ~30 bytes (8 + ~20 URL + 8 epoch)
        let insert = TenantInsert {
            node_path_hash: hash_node_path(&"a".repeat(100)),
            worker_url: "grpc://worker1:8000".to_string(),
            epoch: 12345,
        };
        let delta = TenantDelta {
            model_id: "model1".to_string(),
            version: 1,
            inserts: vec![insert],
            evictions: vec![],
        };
        let delta_bytes = delta.to_bytes().unwrap();

        // A TreeOperation with a 20k-char prompt is ~20KB+
        let tree_op = TreeOperation::Insert(TreeInsertOp {
            key: TreeKey::Text("x".repeat(20_000)),
            tenant: "grpc://worker1:8000".to_string(),
        });
        let tree_state = TreeState {
            model_id: "model1".to_string(),
            operations: vec![tree_op],
            version: 1,
        };
        let tree_bytes = tree_state.to_bytes().unwrap();

        // TenantDelta should be orders of magnitude smaller
        assert!(
            delta_bytes.len() < tree_bytes.len() / 10,
            "TenantDelta ({} bytes) should be much smaller than TreeState ({} bytes)",
            delta_bytes.len(),
            tree_bytes.len()
        );
    }
}
