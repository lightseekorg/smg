//! `OpenAiApiKeyAuth` — Bearer-token outbound auth.
//!
//! Mirrors Java's API_KEY branch in
//! `AbstractContainerToolProcessor.java:106-119` (`createOpenAiClient`,
//! API_KEY case) plus
//! `:128-146` (`resolveContainerRequestHeaders`, API_KEY case where the
//! OpenAI Java SDK auto-attaches `Authorization: Bearer <key>` and the
//! caller adds `OpenAI-Project: <projectId>` when present).

use async_trait::async_trait;
use bytes::Bytes;
use http::Request;
use secrecy::{ExposeSecret, SecretString};

use crate::{AuthError, OutboundAuth};

/// API-key based outbound auth.
///
/// On `apply`, sets:
/// - `Authorization: Bearer <api_key>`
/// - `OpenAI-Project: <project_id>` if a project id was configured.
#[derive(Debug, Clone)]
pub struct OpenAiApiKeyAuth {
    api_key: SecretString,
    project_id: Option<String>,
}

impl OpenAiApiKeyAuth {
    /// Construct with just an API key. No project header attached.
    pub fn new(api_key: impl Into<SecretString>) -> Self {
        Self {
            api_key: api_key.into(),
            project_id: None,
        }
    }

    /// Attach an `OpenAI-Project` header.
    pub fn with_project(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }
}

#[async_trait]
impl OutboundAuth for OpenAiApiKeyAuth {
    async fn apply(&self, req: &mut Request<Bytes>) -> Result<(), AuthError> {
        let bearer = format!("Bearer {}", self.api_key.expose_secret());
        let bearer_val: http::HeaderValue = bearer
            .parse()
            .map_err(|_| AuthError::InvalidCredential("api_key contains invalid header chars"))?;
        req.headers_mut().insert("authorization", bearer_val);
        if let Some(pid) = &self.project_id {
            let pid_val: http::HeaderValue = pid
                .parse()
                .map_err(|_| AuthError::InvalidCredential("project_id contains invalid header chars"))?;
            req.headers_mut().insert("openai-project", pid_val);
        }
        Ok(())
    }
}
