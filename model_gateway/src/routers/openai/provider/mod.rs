//! Provider abstractions for vendor-specific API transformations.

mod anthropic;
mod google;
mod openai;
mod provider_trait;
mod registry;
mod sglang;
mod types;
mod xai;

pub use provider_trait::Provider;
pub use registry::ProviderRegistry;
pub use types::ProviderError;

#[cfg(test)]
mod tests;
