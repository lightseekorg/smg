// ============================================================================
// CRDT OR-Map - High-Performance Transparent CRDT KV Storage
// ============================================================================

mod crdt;
mod epoch_max_wins;
mod kv_store;
mod merge_strategy;
mod operation;
mod replica;

// Export core types
pub use crdt::CrdtOrMap;
pub use epoch_max_wins::{decode, encode, EpochCount, EPOCH_MAX_WINS_ENCODED_LEN};
pub use merge_strategy::MergeStrategy;
pub use operation::{Operation, OperationLog};
pub use replica::ReplicaId;

#[cfg(test)]
mod tests;
