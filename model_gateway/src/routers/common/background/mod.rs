//! Background-mode shared handlers and execution driver.

pub mod create;
pub mod driver;
pub mod supervisor;
pub mod worker;

use std::sync::Arc;

pub use driver::{BackgroundDriver, BackgroundDriverHandle};
use smg_data_connector::BackgroundResponseRepository;
pub use worker::{BackgroundWorker, HeadlessResponses, RealBackgroundWorker};

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

// The [`driver::BackgroundDriver`] is started at process startup whenever
// background mode is enabled (a background repository is configured); it runs
// claimed jobs via [`RealBackgroundWorker`], dispatching per-model through the
// router manager. See `server.rs`.
