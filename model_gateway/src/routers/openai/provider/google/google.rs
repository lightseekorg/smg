use std::any::Any;

use http::HeaderValue;
use reqwest::RequestBuilder;
use serde_json::Value;

use super::{
    super::{Provider, ProviderError},
    streaming,
};
use crate::worker::{Endpoint, ProviderType};

pub struct GoogleProvider;

impl Provider for GoogleProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::Gemini
    }

    fn transform_request(
        &self,
        payload: &mut Value,
        endpoint: Endpoint,
    ) -> Result<(), ProviderError> {
        Self::transform_request_impl(payload, endpoint);
        Ok(())
    }

    fn transform_response(
        &self,
        response: &mut Value,
        endpoint: Endpoint,
    ) -> Result<(), ProviderError> {
        Self::transform_response_impl(response, endpoint);
        Ok(())
    }

    fn transform_stream_event(
        &self,
        event: &Value,
        state: Option<&mut (dyn Any + Send)>,
        endpoint: Endpoint,
    ) -> Result<Vec<Value>, ProviderError> {
        Ok(Self::transform_stream_event_with_state_impl(
            event, state, endpoint,
        ))
    }

    fn new_stream_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(streaming::GoogleStreamState::new()))
    }

    fn is_openai_response_shape(&self, response: &Value, endpoint: Endpoint) -> bool {
        !(endpoint == Endpoint::Responses
            && (response.get("candidates").is_some()
                || response.get("modelVersion").is_some()
                || response.get("responseId").is_some()
                || response.get("usageMetadata").is_some()))
    }

    fn upstream_url(
        &self,
        _worker_url: &str,
        provided_upstream_url: Option<&str>,
    ) -> Result<String, ProviderError> {
        let upstream_url = provided_upstream_url
            .map(str::trim)
            .filter(|url| !url.is_empty())
            .ok_or_else(|| {
                ProviderError::TransformError(
                    "Google upstream URL is required and must be provided via x-provider-endpoint header"
                        .to_string(),
                )
            })?;

        Ok(upstream_url.to_string())
    }

    fn apply_headers(
        &self,
        mut builder: RequestBuilder,
        auth_header: Option<&HeaderValue>,
    ) -> RequestBuilder {
        if let Some(auth) = auth_header {
            if let Ok(auth_str) = auth.to_str() {
                let trimmed = auth_str.trim();
                if !trimmed.is_empty() {
                    if trimmed.split_once(' ').is_some_and(|(scheme, token)| {
                        scheme.eq_ignore_ascii_case("bearer") && !token.trim().is_empty()
                    }) {
                        builder = builder.header("Authorization", trimmed);
                    } else {
                        builder = builder.header("x-goog-api-key", trimmed);
                    }
                }
            }
        }
        builder
    }
}
