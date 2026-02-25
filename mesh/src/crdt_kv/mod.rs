// ============================================================================
// CRDT OR-Map - High-Performance Transparent CRDT KV Storage
// ============================================================================

mod crdt;
mod kv_store;
mod operation;
mod replica;

// Export core types
pub use crdt::CrdtOrMap;
pub use operation::{Operation, OperationLog};
pub use replica::ReplicaId;

#[cfg(test)]
mod tests;
