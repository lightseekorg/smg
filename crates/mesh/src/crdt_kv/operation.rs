use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::replica::ReplicaId;

// ============================================================================
// Operation Type Definition - Atomic Unit of State Change
// ============================================================================

/// CRDT operation type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Operation {
    /// Insert operation: key, value, timestamp, replica_id
    Insert {
        key: String,
        value: Vec<u8>,
        timestamp: u64,
        replica_id: ReplicaId,
    },
    /// Remove operation: key, timestamp, replica_id
    Remove {
        key: String,
        timestamp: u64,
        replica_id: ReplicaId,
    },
}

impl Operation {
    /// Create insert operation
    pub fn insert(key: String, value: Vec<u8>, timestamp: u64, replica_id: ReplicaId) -> Self {
        Self::Insert {
            key,
            value,
            timestamp,
            replica_id,
        }
    }

    /// Create remove operation
    pub fn remove(key: String, timestamp: u64, replica_id: ReplicaId) -> Self {
        Self::Remove {
            key,
            timestamp,
            replica_id,
        }
    }

    /// Get the key of the operation
    pub fn key(&self) -> &str {
        match self {
            Self::Insert { key, .. } => key,
            Self::Remove { key, .. } => key,
        }
    }

    /// Get the timestamp of the operation
    pub fn timestamp(&self) -> u64 {
        match self {
            Self::Insert { timestamp, .. } => *timestamp,
            Self::Remove { timestamp, .. } => *timestamp,
        }
    }

    /// Get the replica ID of the operation
    pub fn replica_id(&self) -> ReplicaId {
        match self {
            Self::Insert { replica_id, .. } => *replica_id,
            Self::Remove { replica_id, .. } => *replica_id,
        }
    }

    fn operation_id(&self) -> (ReplicaId, u64) {
        (self.replica_id(), self.timestamp())
    }
}

// ============================================================================
// Operation Log - State Operation Pipeline
// ============================================================================

/// Operation log, recording all state changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationLog {
    operations: Vec<Operation>,
}

impl OperationLog {
    fn decode_counter_payload(value: &[u8]) -> Option<i64> {
        bincode::deserialize::<i64>(value).ok().or_else(|| {
            bincode::deserialize::<HashMap<String, i64>>(value)
                .ok()
                .and_then(|map| map.get("value").copied())
        })
    }

    /// Create empty operation log
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    /// Append operation to log
    pub fn append(&mut self, operation: Operation) {
        self.operations.push(operation);
    }

    /// Get all operations
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    /// Serialize to bincode bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Box<bincode::ErrorKind>> {
        bincode::serialize(self)
    }

    /// Deserialize from bincode bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Box<bincode::ErrorKind>> {
        bincode::deserialize(bytes)
    }

    /// Get number of operations
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    fn latest_operations_by_key(&self) -> HashMap<String, Operation> {
        let mut latest_by_key: HashMap<String, Operation> = HashMap::new();

        for operation in &self.operations {
            let key = operation.key().to_string();
            match latest_by_key.get(&key) {
                Some(current)
                    if (current.timestamp(), current.replica_id())
                        >= (operation.timestamp(), operation.replica_id()) => {}
                _ => {
                    latest_by_key.insert(key, operation.clone());
                }
            }
        }

        latest_by_key
    }

    /// Keep only latest operation per key to bound log growth.
    ///
    /// This uses `latest_operations_by_key` LWW tie-breaking by `(timestamp, ReplicaId)`.
    /// As a result, concurrent operations may be compacted away deterministically, so
    /// `compact()` + `merge()` can be non-idempotent in raw log contents even though
    /// `apply_operation` and `operation_id` guards keep state semantics safe.
    /// Stronger concurrency retention would require vector-clock/version-vector metadata.
    pub fn compact(&mut self) {
        self.operations = self
            .latest_operations_by_key()
            .into_values()
            .collect::<Vec<_>>();
        self.operations
            .sort_by_key(|operation| (operation.timestamp(), operation.replica_id()));
    }

    /// Drop operations with timestamp <= watermark.
    pub fn compact_up_to(&mut self, watermark: u64) {
        self.operations
            .retain(|operation| operation.timestamp() > watermark);
    }

    /// Build a latest-state snapshot and clear the operation log.
    pub fn snapshot_and_truncate(&mut self) -> HashMap<String, Operation> {
        let snapshot = self.latest_operations_by_key();
        self.operations.clear();
        snapshot
    }

    /// Decode the latest known counter value for a key from log payloads.
    pub fn latest_counter_value(&self, key: &str) -> Option<i64> {
        let latest = self
            .operations
            .iter()
            .filter(|operation| operation.key() == key)
            .max_by_key(|operation| (operation.timestamp(), operation.replica_id()))?;

        match latest {
            Operation::Insert { value, .. } => Self::decode_counter_payload(value),
            Operation::Remove { .. } => None,
        }
    }

    /// Decode the latest known counter value, regardless of key.
    pub fn latest_counter_value_any(&self) -> Option<i64> {
        let latest = self
            .operations
            .iter()
            .max_by_key(|operation| (operation.timestamp(), operation.replica_id()))?;

        match latest {
            Operation::Insert { value, .. } => Self::decode_counter_payload(value),
            Operation::Remove { .. } => None,
        }
    }

    /// Merge another operation log.
    ///
    /// INVARIANT: `Operation::operation_id()` (`ReplicaId`, `timestamp`) is unique per operation
    /// because each replica's `LamportClock::tick()` is monotonic and never repeats a timestamp.
    pub fn merge(&mut self, other: &OperationLog) {
        let mut seen_ids: HashSet<(ReplicaId, u64)> = self
            .operations
            .iter()
            .map(Operation::operation_id)
            .collect();

        for operation in &other.operations {
            if seen_ids.insert(operation.operation_id()) {
                self.operations.push(operation.clone());
            }
        }
    }
}

impl Default for OperationLog {
    fn default() -> Self {
        Self::new()
    }
}
