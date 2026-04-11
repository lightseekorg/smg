use std::{fmt, sync::Arc};

/// Health checker handle with graceful shutdown.
///
/// The checker sleeps until the next worker is due for a health check,
/// so it wakes only when there is actual work to do.
pub(crate) struct HealthChecker {
    handle: Option<tokio::task::JoinHandle<()>>,
    shutdown_notify: Arc<tokio::sync::Notify>,
}

impl fmt::Debug for HealthChecker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HealthChecker").finish()
    }
}

impl HealthChecker {
    pub fn new(
        handle: tokio::task::JoinHandle<()>,
        shutdown_notify: Arc<tokio::sync::Notify>,
    ) -> Self {
        Self {
            handle: Some(handle),
            shutdown_notify,
        }
    }

    /// Shutdown the health checker gracefully.
    /// Wakes the sleeping task immediately so it can exit cleanly.
    /// Prefer this over dropping when you can `.await` — it lets the
    /// current health-check iteration finish instead of aborting mid-flight.
    #[expect(
        dead_code,
        reason = "Drop::drop handles abort; this exists for graceful shutdown when an async context is available"
    )]
    pub async fn shutdown(&mut self) {
        self.shutdown_notify.notify_one();
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

impl Drop for HealthChecker {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}
