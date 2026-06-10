//! Background-mode shared handlers and execution driver.

pub mod create;
pub mod driver;
pub mod supervisor;
pub mod worker;

use std::sync::Arc;

pub use driver::{BackgroundDriver, BackgroundDriverHandle};
use smg_data_connector::BackgroundResponseRepository;
pub(crate) use worker::run_job;
pub use worker::BackgroundWorker;

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

// The [`driver::BackgroundDriver`] is started whenever background mode is
// enabled (a background repository is configured). The gRPC router constructs
// and starts it with itself as the [`BackgroundWorker`]; see
// `routers::grpc::router`.
