// ============================================================================
// CRDT OR-Map - High-Performance Transparent CRDT KV Storage
// ============================================================================

mod crdt;
// `epoch_max_wins` hosts the rate-limit value merge helper. Nothing
// outside the crate imports it yet — the RateLimitSyncAdapter PR (the
// first consumer) will add the `pub use` at that point so the dead-code
// warnings don't fire on this standalone merge module.
mod epoch_max_wins;
mod kv_store;
mod operation;
mod replica;

// Export core types
pub use crdt::CrdtOrMap;
pub use operation::{Operation, OperationLog};
pub use replica::ReplicaId;

#[cfg(test)]
mod tests;
