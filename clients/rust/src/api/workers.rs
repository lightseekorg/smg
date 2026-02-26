use openai_protocol::worker::{WorkerInfo, WorkerSpec, WorkerUpdateRequest};

use crate::{transport::Transport, SmgError};

/// Workers API (`/workers`).
///
/// Note: worker mutation endpoints (create/update/delete) and list return
/// `serde_json::Value` because the server response shapes (`{status, worker_id, ...}`)
/// differ from the protocol types (`WorkerApiResponse`, `WorkerListResponse`).
pub struct Workers {
    transport: Transport,
}

impl Workers {
    pub(crate) fn new(transport: Transport) -> Self {
        Self { transport }
    }

    /// Register a new worker.
    ///
    /// Returns 202 Accepted with `{status, worker_id, url, location}`.
    pub async fn create(&self, spec: &WorkerSpec) -> Result<serde_json::Value, SmgError> {
        let resp = self.transport.post("/workers", spec).await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }

    /// List all registered workers.
    ///
    /// Returns `{workers: [WorkerInfo], total, stats: {prefill_count, decode_count, regular_count}}`.
    pub async fn list(&self) -> Result<serde_json::Value, SmgError> {
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
    ///
    /// Returns 202 Accepted with `{status, worker_id, message}`.
    pub async fn update(
        &self,
        worker_id: &str,
        request: &WorkerUpdateRequest,
    ) -> Result<serde_json::Value, SmgError> {
        let resp = self
            .transport
            .put(&format!("/workers/{worker_id}"), request)
            .await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }

    /// Remove a worker.
    ///
    /// Returns 202 Accepted with `{status, worker_id, message}`.
    pub async fn delete(&self, worker_id: &str) -> Result<serde_json::Value, SmgError> {
        let resp = self
            .transport
            .delete(&format!("/workers/{worker_id}"))
            .await?;
        let body = resp.text().await.map_err(SmgError::Connection)?;
        serde_json::from_str(&body).map_err(SmgError::from)
    }
}
