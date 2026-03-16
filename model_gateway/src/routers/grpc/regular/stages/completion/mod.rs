//! Completion endpoint pipeline stages
//!
//! These stages handle completion-specific preprocessing, request building, and
//! response processing. `CompletionRequest` flows natively as
//! `RequestType::Completion` through every pipeline stage — preparation, worker
//! selection, client acquisition, request building, execution, and response
//! processing — following the same architecture as chat and generate.

mod preparation;
mod request_building;
mod response_processing;

pub(crate) use preparation::CompletionPreparationStage;
pub(crate) use request_building::CompletionRequestBuildingStage;
pub(crate) use response_processing::CompletionResponseProcessingStage;
