use http::HeaderValue;
use reqwest::RequestBuilder;
use serde_json::Value;

use super::{types::strip_sglang_fields, ProviderError};
use crate::worker::{Endpoint, ProviderType};

/// Default `transform_request` strips SGLang fields.
pub trait Provider: Send + Sync {
    fn provider_type(&self) -> ProviderType;

    fn transform_request(
        &self,
        payload: &mut Value,
        _endpoint: Endpoint,
    ) -> Result<(), ProviderError> {
        strip_sglang_fields(payload);
        Ok(())
    }

    fn transform_response(
        &self,
        _response: &mut Value,
        _endpoint: Endpoint,
    ) -> Result<(), ProviderError> {
        Ok(())
    }

    fn upstream_url(
        &self,
        worker_url: &str,
        provided_upstream_url: Option<&str>,
    ) -> Result<String, ProviderError> {
        Ok(provided_upstream_url
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("{worker_url}/v1/responses")))
    }

    fn apply_headers(
        &self,
        builder: RequestBuilder,
        _auth_header: Option<&HeaderValue>,
    ) -> RequestBuilder {
        builder
    }
}
