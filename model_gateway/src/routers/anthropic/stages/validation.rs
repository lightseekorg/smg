//! Validation stage for Anthropic router pipeline
//!
//! This stage validates the request and extracts key information:
//! - Validates required fields (model, messages, max_tokens)
//! - Extracts model ID for routing
//! - Checks message format validity
//! - Extracts streaming flag

use async_trait::async_trait;
use axum::{http::StatusCode, response::IntoResponse};
use tracing::debug;

use super::{PipelineStage, StageResult};
use crate::routers::anthropic::context::{RequestContext, ValidationOutput};

/// Maximum allowed message count to prevent DoS
const MAX_MESSAGE_COUNT: usize = 1000;

/// Validation stage
pub(crate) struct ValidationStage;

impl ValidationStage {
    /// Create a new validation stage
    pub fn new() -> Self {
        Self
    }
}

impl Default for ValidationStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for ValidationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> StageResult {
        debug!(
            model = %ctx.input.model_id,
            "Validating Messages API request"
        );

        let request = &ctx.input.request;

        // Validate model is not empty
        if request.model.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                "model field is required and cannot be empty",
            )
                .into_response());
        }

        // Validate messages array is not empty
        if request.messages.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                "messages array is required and cannot be empty",
            )
                .into_response());
        }

        // Validate message count (DoS prevention)
        if request.messages.len() > MAX_MESSAGE_COUNT {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "messages array exceeds maximum of {} messages",
                    MAX_MESSAGE_COUNT
                ),
            )
                .into_response());
        }

        // Validate max_tokens is reasonable
        if request.max_tokens == 0 {
            return Err(
                (StatusCode::BAD_REQUEST, "max_tokens must be greater than 0").into_response(),
            );
        }

        // Extract validation output
        let validation_output = ValidationOutput {
            is_streaming: request.stream.unwrap_or(false),
            max_tokens: request.max_tokens,
        };

        debug!(
            model_id = %ctx.input.model_id,
            is_streaming = %validation_output.is_streaming,
            max_tokens = %validation_output.max_tokens,
            "Request validation successful"
        );

        ctx.state.validation = Some(validation_output);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "validation"
    }
}
