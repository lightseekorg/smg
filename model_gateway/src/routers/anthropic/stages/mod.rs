//! Pipeline stages for Anthropic router
//!
//! This module provides a trait-based pipeline architecture where each stage
//! performs a specific part of request processing. Stages are executed in sequence
//! and can short-circuit by returning an early response (for errors or streaming).
//!
//! ## Stage Order
//!
//! 1. **Validation** - Validate request fields and extract model ID
//! 2. **Worker Selection** - Select appropriate worker for the request
//! 3. **Request Building** - Build HTTP request for worker
//! 4. **Request Execution** - Send request to worker
//! 5. **Response Processing** - Parse response and record metrics

mod request_building;
mod request_execution;
mod response_processing;
mod validation;
mod worker_selection;

use async_trait::async_trait;
use axum::response::Response;
pub(crate) use request_building::RequestBuildingStage;
pub(crate) use request_execution::RequestExecutionStage;
pub(crate) use response_processing::ResponseProcessingStage;
pub(crate) use validation::ValidationStage;
pub(crate) use worker_selection::WorkerSelectionStage;

use super::context::RequestContext;

/// Result type for pipeline stage execution
///
/// - `Ok(None)` - Stage completed successfully, continue to next stage
/// - `Ok(Some(response))` - Stage completed with early response (streaming or redirect)
/// - `Err(response)` - Stage failed with error response
pub(crate) type StageResult = Result<Option<Response>, Response>;

/// A single stage in the request processing pipeline
///
/// Each stage receives a mutable reference to the request context,
/// can read/modify state, and either continues to the next stage
/// or returns an early response.
#[async_trait]
pub(crate) trait PipelineStage: Send + Sync {
    /// Execute this pipeline stage
    ///
    /// # Arguments
    /// * `ctx` - The request context with accumulated state
    ///
    /// # Returns
    /// * `Ok(None)` - Continue to next stage
    /// * `Ok(Some(response))` - Return early with this response (e.g., streaming)
    /// * `Err(response)` - Return error response
    async fn execute(&self, ctx: &mut RequestContext) -> StageResult;

    /// Get the name of this stage (for logging and metrics)
    fn name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test that the stage modules are properly exported
    #[test]
    fn test_stage_names() {
        let validation = ValidationStage::new();
        assert_eq!(validation.name(), "validation");

        let request_building = RequestBuildingStage::new();
        assert_eq!(request_building.name(), "request_building");
    }
}
