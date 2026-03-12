//! Completion endpoint pipeline stages
//!
//! Completions are processed by converting to a GenerateRequest, running
//! through the generate preparation pipeline, then wrapping the generate
//! response as a CompletionResponse or CompletionStreamResponse.

mod preparation;
mod response_processing;

pub(crate) use preparation::CompletionPreparationStage;
pub(crate) use response_processing::CompletionResponseProcessingStage;
