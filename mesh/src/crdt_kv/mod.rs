// ============================================================================
// CRDT OR-Map - High-Performance Transparent CRDT KV Storage
// ============================================================================

mod crdt;
mod kv_store;
mod operation;
mod replica;

// Export core types
pub use crdt::CrdtOrMap;
#[expect(
    unused_imports,
    reason = "public re-export for external API ergonomics"
)]
pub use operation::{Operation, OperationLog};
#[expect(
    unused_imports,
    reason = "public re-export for external API ergonomics"
)]
pub use replica::ReplicaId;

#[cfg(test)]
mod tests;
