// ============================================================================
// CRDT OR-Map - High-Performance Transparent CRDT KV Storage
// ============================================================================

mod crdt;
mod kv_store;
mod operation;
mod replica;

// Export core types
pub use crdt::CrdtOrMap;
pub use operation::OperationLog;
// pub use replica::ReplicaId;
// pub use kv_store::KvStore;
// pub use operation::Operation;
// pub use replica::LamportClock;

#[cfg(test)]
mod tests;
