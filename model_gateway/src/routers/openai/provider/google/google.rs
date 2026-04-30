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
        _payload: &mut Value,
        _endpoint: Endpoint,
    ) -> Result<(), ProviderError> {
        Ok(())
    }

    fn transform_response(
        &self,
        _response: &mut Value,
        _endpoint: Endpoint,
    ) -> Result<(), ProviderError> {
        Ok(())
    }

    fn apply_headers(&self, builder: RequestBuilder) -> RequestBuilder {
        builder
    }
}
