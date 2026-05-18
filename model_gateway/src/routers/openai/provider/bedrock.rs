use super::Provider;
use crate::worker::ProviderType;

pub struct BedrockProvider;

impl Provider for BedrockProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::Bedrock
    }
}
