/// Merge strategy for CRDT namespaces. Determines how conflicts are resolved
/// when two nodes write the same key concurrently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Higher (version, replica_id) wins. Used for worker:*, policy:*, config:*.
    LastWriterWins,
    /// Compare epochs first, then max within same epoch.
    /// Values MUST be exactly 16 bytes: epoch (u64 big-endian) + count (i64 big-endian).
    EpochMaxWins,
}
