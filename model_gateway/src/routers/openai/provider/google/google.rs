use reqwest::RequestBuilder;
use serde_json::Value;

use super::super::{Provider, ProviderError};
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
        GoogleProvider::transform_request_impl(payload, endpoint);
        Ok(())
    }

    fn transform_response(
        &self,
        response: &mut Value,
        endpoint: Endpoint,
    ) -> Result<(), ProviderError> {
        GoogleProvider::transform_response_impl(response, endpoint);
        Ok(())
    }

    fn apply_headers(&self, builder: RequestBuilder) -> RequestBuilder {
        builder
    }
}
