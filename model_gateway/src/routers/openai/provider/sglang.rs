use serde_json::Value;

use super::{Provider, ProviderError};
use crate::worker::{Endpoint, ProviderType};

pub struct SGLangProvider;

impl Provider for SGLangProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenAI
    }

    fn transform_request(
        &self,
        _payload: &mut Value,
        _endpoint: Endpoint,
    ) -> Result<(), ProviderError> {
        Ok(())
    }
}
