use super::Provider;
use crate::worker::ProviderType;

pub struct OpenAIProvider;

impl Provider for OpenAIProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenAI
    }
}
