use openai_protocol::worker::{
    WorkerApiResponse, WorkerInfo, WorkerListResponse, WorkerSpec, WorkerUpdateRequest,
};

use crate::{transport::Transport, SmgError};

/// Workers API (`/workers`).
pub struct Workers {
    transport: Transport,
}

impl Workers {
    pub(crate) fn new(transport: Transport) -> Self {
        Self { transport }
    }

    /// Register a new worker.
    pub async fn create(&self, spec: &WorkerSpec) -> Result<WorkerApiResponse, SmgError> {
        let resp = self.transport.post("/workers", spec).await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }

    /// List all registered workers.
    pub async fn list(&self) -> Result<WorkerListResponse, SmgError> {
        let resp = self.transport.get("/workers").await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }

    /// Get details for a specific worker.
    pub async fn get(&self, worker_id: &str) -> Result<WorkerInfo, SmgError> {
        let resp = self.transport.get(&format!("/workers/{worker_id}")).await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }

    /// Update a worker's configuration.
    pub async fn update(
        &self,
        worker_id: &str,
        request: &WorkerUpdateRequest,
    ) -> Result<WorkerApiResponse, SmgError> {
        let resp = self
            .transport
            .put(&format!("/workers/{worker_id}"), request)
            .await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }

    /// Remove a worker.
    pub async fn delete(&self, worker_id: &str) -> Result<WorkerApiResponse, SmgError> {
        let resp = self
            .transport
            .delete(&format!("/workers/{worker_id}"))
            .await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }
}
