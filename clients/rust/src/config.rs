use std::time::Duration;

/// Configuration for the SMG client.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Base URL of the SMG server (e.g. `http://localhost:30000`).
    pub base_url: String,
    /// Optional API key for authentication. Falls back to `SMG_API_KEY` env var.
    pub api_key: Option<String>,
    /// Request timeout. Defaults to 60 seconds.
    pub timeout: Duration,
    /// Maximum number of retries for transient errors. Defaults to 2.
    pub max_retries: u32,
}

impl ClientConfig {
    /// Create a new config with just a base URL. API key is read from `SMG_API_KEY`.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: std::env::var("SMG_API_KEY").ok(),
            timeout: Duration::from_secs(60),
            max_retries: 2,
        }
    }

    /// Set the API key explicitly.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the maximum number of retries.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }
}
