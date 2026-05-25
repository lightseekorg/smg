/// Merge strategy for CRDT namespaces. Determines how conflicts are resolved
/// when two nodes write the same key concurrently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Higher (version, replica_id) wins. Used for worker:*, policy:*, config:*.
    LastWriterWins,
    /// Compare epochs first, then max within same epoch.
    ///
    /// The raw write payload at the put boundary MUST be exactly 16 bytes:
    /// epoch (u64 big-endian) followed by count (i64 big-endian). The CRDT
    /// normalizes stored and replicated values into a richer
    /// `RateLimitShard` form internally (live-point frontier plus an
    /// embedded tombstone boundary), so values observed by `get`, gossip
    /// peers, and subscribers after normalization are larger than 16
    /// bytes.
    EpochMaxWins,
}
