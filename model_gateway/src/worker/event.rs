//! Events emitted by [`WorkerRegistry`] on state mutations.

/// Events broadcast when worker state changes.
#[derive(Debug, Clone)]
pub enum WorkerEvent {
    /// A worker was registered.
    Registered { url: String },
    /// A worker was removed.
    Removed { url: String },
    /// A worker's health status changed.
    HealthChanged { url: String, healthy: bool },
}
