use crate::{
    api::{
        chat::Chat, classify::Classify, completions::Completions, embeddings::Embeddings,
        messages::Messages, models::Models, parser::Parser, rerank::Rerank, responses::Responses,
        workers::Workers,
    },
    config::ClientConfig,
    transport::Transport,
    SmgError,
};

/// Async HTTP client for SMG (Shepherd Model Gateway).
///
/// # Example
///
/// ```no_run
/// use smg_client::{SmgClient, ClientConfig};
///
/// # async fn example() -> Result<(), smg_client::SmgError> {
/// let client = SmgClient::new(ClientConfig::new("http://localhost:30000"))?;
///
/// // List models
/// let models = client.models().list().await?;
/// println!("Available models: {:?}", models.data.len());
///
/// // Chat completion
/// let req = openai_protocol::chat::ChatCompletionRequest {
///     model: "llama-3.1-8b".to_string(),
///     ..Default::default()
/// };
/// let resp = client.chat().create(&req).await?;
/// # Ok(())
/// # }
/// ```
pub struct SmgClient {
    transport: Transport,
}

impl SmgClient {
    /// Create a new client with the given configuration.
    pub fn new(config: ClientConfig) -> Result<Self, SmgError> {
        let transport = Transport::new(&config)?;
        Ok(Self { transport })
    }

    /// Access the chat completions API (`/v1/chat/completions`).
    pub fn chat(&self) -> Chat {
        Chat::new(self.transport.clone())
    }

    /// Access the legacy completions API (`/v1/completions`).
    pub fn completions(&self) -> Completions {
        Completions::new(self.transport.clone())
    }

    /// Access the embeddings API (`/v1/embeddings`).
    pub fn embeddings(&self) -> Embeddings {
        Embeddings::new(self.transport.clone())
    }

    /// Access the Anthropic messages API (`/v1/messages`).
    pub fn messages(&self) -> Messages {
        Messages::new(self.transport.clone())
    }

    /// Access the models API (`/v1/models`).
    pub fn models(&self) -> Models {
        Models::new(self.transport.clone())
    }

    /// Access the rerank API (`/v1/rerank`).
    pub fn rerank(&self) -> Rerank {
        Rerank::new(self.transport.clone())
    }

    /// Access the classify API (`/v1/classify`).
    pub fn classify(&self) -> Classify {
        Classify::new(self.transport.clone())
    }

    /// Access the responses API (`/v1/responses`).
    pub fn responses(&self) -> Responses {
        Responses::new(self.transport.clone())
    }

    /// Access the parser API (`/parse/*`).
    pub fn parser(&self) -> Parser {
        Parser::new(self.transport.clone())
    }

    /// Access the workers API (`/workers`).
    pub fn workers(&self) -> Workers {
        Workers::new(self.transport.clone())
    }
}
