//! Messages API request handler (non-streaming)
//!
//! This module handles non-streaming requests to the `/v1/messages` endpoint.
//! It coordinates with routing policies, worker selection, and response processing.

use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};

use crate::protocols::messages::CreateMessageRequest;

/// Handler for Messages API requests
pub struct MessagesHandler;

impl MessagesHandler {
    pub async fn handle_non_streaming(
        _headers: Option<&HeaderMap>,
        _request: &CreateMessageRequest,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Messages API non-streaming handler coming in PR #2",
        )
            .into_response()
    }

    pub async fn handle_streaming(
        _headers: Option<&HeaderMap>,
        _request: &CreateMessageRequest,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Messages API streaming handler coming in PR #3",
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_handler_exists() {
        // Verify handler can be instantiated
        let _handler = MessagesHandler;
    }

    #[tokio::test]
    async fn test_non_streaming_returns_not_implemented() {
        use crate::protocols::messages::{CreateMessageRequest, InputContent, InputMessage, Role};

        let request = CreateMessageRequest {
            model: "test-model".to_string(),
            messages: vec![InputMessage {
                role: Role::User,
                content: InputContent::String("test".to_string()),
            }],
            max_tokens: 100,
            metadata: None,
            service_tier: None,
            stop_sequences: None,
            stream: None,
            system: None,
            temperature: None,
            thinking: None,
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
            container: None,
            mcp_servers: None,
        };

        let response = MessagesHandler::handle_non_streaming(None, &request).await;
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    }
}
