//! Request execution stage for Anthropic router pipeline
//!
//! This stage sends the HTTP request to the worker:
//! - Sends request with timeout
//! - Handles connection errors
//!
//! Note: Circuit breaker outcome is recorded in ResponseProcessingStage
//! after full response consumption to avoid misreporting success when
//! later parsing or streaming fails.

use std::time::Duration;

use async_trait::async_trait;
use tracing::{debug, error, warn};

use super::{PipelineStage, StageResult};
use crate::routers::{anthropic::context::RequestContext, error};

/// Request execution stage
pub(crate) struct RequestExecutionStage {
    http_client: reqwest::Client,
    timeout: Duration,
}

impl RequestExecutionStage {
    /// Create a new request execution stage
    pub fn new(http_client: reqwest::Client, timeout: Duration) -> Self {
        Self {
            http_client,
            timeout,
        }
    }
}

#[async_trait]
impl PipelineStage for RequestExecutionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> StageResult {
        // Ensure we have HTTP request state
        let http_request = ctx.state.http_request.as_ref().ok_or_else(|| {
            error!("Request execution stage called without HTTP request state");
            error::internal_error("no_http_request", "Internal error: no HTTP request built")
        })?;

        // Get worker for circuit breaker recording
        let worker = ctx.state.worker.as_ref().ok_or_else(|| {
            error!("Request execution stage called without worker");
            error::internal_error("no_worker", "Internal error: no worker selected")
        })?;

        let url = &http_request.url;
        debug!(url = %url, "Sending request to worker");

        // Increment worker load - will be decremented in response_processing stage
        // when the response is fully consumed (for streaming) or parsed (for non-streaming)
        worker.increment_load();

        // Build and send request (reqwest handles JSON serialization)
        let mut request_builder = self
            .http_client
            .post(url)
            .json(&*ctx.input.request)
            .timeout(self.timeout);

        // Add propagated headers
        for (key, value) in &http_request.headers {
            request_builder = request_builder.header(key, value);
        }

        let result = request_builder.send().await;

        // Note: Worker load is NOT decremented here because:
        // - For streaming: the worker remains busy while body is being streamed
        // - For non-streaming: the response body still needs to be read
        // Load is decremented in response_processing stage when fully done.

        match result {
            Ok(response) => {
                let status = response.status();
                debug!(
                    url = %url,
                    status = %status,
                    "Received response from worker"
                );

                // Note: Circuit breaker outcome is deferred to ResponseProcessingStage
                // to avoid misreporting success when parsing/streaming fails later

                // Store response for next stage
                ctx.state.response.status_code = Some(status.as_u16());
                ctx.state.response.worker_response = Some(response);
                Ok(None)
            }
            Err(e) => {
                warn!(url = %url, error = %e, "Request to worker failed");

                // Decrement load on connection/send failure (no response to process)
                worker.decrement_load();

                // Record circuit breaker failure
                worker.record_outcome(false);

                // Determine error type and return appropriate response
                if e.is_timeout() {
                    Err(error::gateway_timeout(
                        "timeout",
                        format!("Request timeout: {}", e),
                    ))
                } else if e.is_connect() {
                    Err(error::bad_gateway(
                        "connection_failed",
                        format!("Connection failed: {}", e),
                    ))
                } else {
                    Err(error::bad_gateway(
                        "request_failed",
                        format!("Request failed: {}", e),
                    ))
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "request_execution"
    }
}
