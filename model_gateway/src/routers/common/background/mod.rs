//! Background-mode shared handler scaffolding.
//!
//! BGM-PR-03 ships only [`BackgroundServices`]. Concrete create / cancel /
//! resume handlers land in later PRs.

use std::sync::Arc;

use smg_data_connector::BackgroundResponseRepository;

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
