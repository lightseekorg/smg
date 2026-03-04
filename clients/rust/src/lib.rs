pub mod api;
mod client;
pub mod config;
mod error;
pub mod streaming;
mod transport;

pub use client::SmgClient;
pub use config::ClientConfig;
pub use error::SmgError;
// Re-export the protocol types so users don't need to depend on openai-protocol directly.
pub use openai_protocol;
