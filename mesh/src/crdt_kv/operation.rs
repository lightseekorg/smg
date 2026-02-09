use serde::{Deserialize, Serialize};

use super::replica::ReplicaId;

// ============================================================================
// Operation Type Definition - Atomic Unit of State Change
// ============================================================================

/// CRDT operation type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize to JSON bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Deserialize from binary
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    /// Get number of operations
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Merge another operation log
    pub fn merge(&mut self, other: &OperationLog) {
        self.operations.extend_from_slice(&other.operations);
    }
}

impl Default for OperationLog {
    fn default() -> Self {
        Self::new()
    }
}
