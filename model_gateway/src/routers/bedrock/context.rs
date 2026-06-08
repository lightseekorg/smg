use std::sync::Arc;

use super::signing::AwsSigner;
use crate::{config::types::BedrockConfig, worker::WorkerRegistry};

pub(crate) struct RouterContext {
    pub worker_registry: Arc<WorkerRegistry>,
    pub http_client: reqwest::Client,
    pub bedrock: BedrockConfig,
    pub signer: AwsSigner,
}

impl RouterContext {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        http_client: reqwest::Client,
        bedrock: BedrockConfig,
        signer: AwsSigner,
    ) -> Self {
        Self {
            worker_registry,
            http_client,
            bedrock,
            signer,
        }
    }
}
