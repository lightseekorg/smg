//! Background-mode shared handlers and execution driver.

pub mod create;
pub mod driver;
pub mod supervisor;
pub mod worker;

use std::sync::Arc;

pub use driver::{BackgroundDriver, BackgroundDriverHandle};
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

// NOTE: BGM-PR-06 deliberately does *not* start the [`driver::BackgroundDriver`]
// at process startup. The only [`BackgroundWorker`] that exists today is
// [`UnavailableBackgroundWorker`], which finalizes every claimed job as
// `failed`; running the driver with it would regress #1614's durable `queued`
// contract (a `background=true` response that clients can poll) into immediate
// failure for every gateway with a background repository — including the default
// `history_backend=memory`. So PR-06 lands the driver + its tests but leaves it
// unspawned; `background=true` requests stay durably `queued` (per #1614).
//
// BGM-PR-07 (#1221) wires `BackgroundDriver::spawn(...)` with the real
// `BackgroundWorker` and gates it on `background_repository` being present.
