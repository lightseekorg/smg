//! Background-mode shared handlers and execution driver.

pub mod create;
pub mod scheduler;
pub mod supervisor;
pub mod worker;

use std::sync::Arc;

pub use scheduler::{BackgroundScheduler, BackgroundSchedulerHandle};
use smg_data_connector::BackgroundResponseRepository;
pub use worker::{BackgroundWorker, UnavailableBackgroundWorker, BACKGROUND_EXECUTION_UNAVAILABLE};

use crate::config::BackgroundConfig;

#[derive(Clone)]
pub struct BackgroundServices {
    repository: Arc<dyn BackgroundResponseRepository>,
    config: Arc<BackgroundConfig>,
}

impl BackgroundServices {
    pub fn new(
        repository: Arc<dyn BackgroundResponseRepository>,
        config: BackgroundConfig,
    ) -> Self {
        Self {
            repository,
            config: Arc::new(config),
        }
    }

    pub fn repository(&self) -> &Arc<dyn BackgroundResponseRepository> {
        &self.repository
    }

    pub fn config(&self) -> &BackgroundConfig {
        &self.config
    }
}

/// Construct and spawn the background scheduler driver over `repository`.
///
/// Wires the default [`UnavailableBackgroundWorker`] (BGM-PR-07 swaps in the
/// real worker), runs the startup claim pass, and starts the supervised
/// claim-tick and sweeper loops. The returned [`BackgroundSchedulerHandle`] owns
/// the spawned tasks and the scheduler `Arc`; the caller must hold it for the
/// process lifetime, otherwise the loops stop when it drops.
///
/// Callers gate this on `background_repository` being present; when no durable
/// (or memory) background repository is configured, nothing is spawned.
pub async fn start_background_scheduler(
    repository: Arc<dyn BackgroundResponseRepository>,
    config: BackgroundConfig,
) -> BackgroundSchedulerHandle {
    let worker: Arc<dyn BackgroundWorker> =
        Arc::new(UnavailableBackgroundWorker::new(Arc::clone(&repository)));
    let scheduler = BackgroundScheduler::new(repository, worker, config);
    scheduler.spawn().await
}
