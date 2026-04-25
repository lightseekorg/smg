//! [`MockBackend`] — in-memory [`ContainerBackend`] for hermetic tests.
//!
//! Used by SMG e2e tests in CB-3 / CB-4 to exercise the dispatch flow
//! without touching the network. Two `MockBackend` instances do NOT share
//! state — each owns its own map.
//!
//! # Behaviour
//!
//! - `create` allocates ids of the form `cntr_mock_<n>` (monotonic per
//!   instance) and returns a fresh [`Container`] in [`ContainerStatus::Running`].
//! - `retrieve` returns the stored container or [`BackendError::NotFound`].
//! - `delete` removes by id; returns `Ok(())` whether or not the id existed
//!   (matches the wire surface — `DELETE` is idempotent on the OpenAI side).
//! - `list` returns every stored container in arbitrary order, with
//!   `has_more: false` and no cursor — pagination is intentionally not
//!   modelled here.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::types::{
    Container, ContainerStatus, CreateContainerParams, ListQuery, Page,
};
use crate::{BackendError, ContainerBackend};

/// In-memory [`ContainerBackend`] for hermetic tests.
///
/// `Clone` shares the same backing map (rc-counted), so callers that want
/// independent state should construct separate instances.
#[derive(Debug, Default, Clone)]
pub struct MockBackend {
    inner: Arc<Mutex<HashMap<String, Container>>>,
    seq: Arc<Mutex<u64>>,
}

impl MockBackend {
    /// Construct an empty backend.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ContainerBackend for MockBackend {
    async fn create(&self, params: CreateContainerParams) -> Result<Container, BackendError> {
        let id = {
            let mut seq = self
                .seq
                .lock()
                .map_err(|e| BackendError::Backend {
                    status: 0,
                    message: format!("mock seq lock poisoned: {e}"),
                })?;
            *seq += 1;
            format!("cntr_mock_{}", *seq)
        };
        let container = Container {
            id: id.clone(),
            object: "container".into(),
            created_at: chrono::Utc::now().timestamp(),
            status: ContainerStatus::Running,
            name: params.name,
            last_active_at: None,
            expires_after: params.expires_after,
            memory_limit: params.memory_limit,
            network_policy: params.network_policy,
            file_ids: params.file_ids,
            skills: params.skills,
        };
        self.inner
            .lock()
            .map_err(|e| BackendError::Backend {
                status: 0,
                message: format!("mock map lock poisoned: {e}"),
            })?
            .insert(id, container.clone());
        Ok(container)
    }

    async fn retrieve(&self, id: &str) -> Result<Container, BackendError> {
        self.inner
            .lock()
            .map_err(|e| BackendError::Backend {
                status: 0,
                message: format!("mock map lock poisoned: {e}"),
            })?
            .get(id)
            .cloned()
            .ok_or_else(|| BackendError::NotFound(id.to_owned()))
    }

    async fn delete(&self, id: &str) -> Result<(), BackendError> {
        self.inner
            .lock()
            .map_err(|e| BackendError::Backend {
                status: 0,
                message: format!("mock map lock poisoned: {e}"),
            })?
            .remove(id);
        Ok(())
    }

    async fn list(&self, _q: ListQuery) -> Result<Page<Container>, BackendError> {
        let data: Vec<Container> = self
            .inner
            .lock()
            .map_err(|e| BackendError::Backend {
                status: 0,
                message: format!("mock map lock poisoned: {e}"),
            })?
            .values()
            .cloned()
            .collect();
        Ok(Page {
            data,
            object: "list".into(),
            first_id: None,
            last_id: None,
            has_more: false,
        })
    }
}
