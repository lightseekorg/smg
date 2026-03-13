//! Messages API endpoint pipeline stages
//!
//! These stages handle Messages API-specific preprocessing, request building,
//! and response processing. Each stage is wired into a dedicated Messages
//! pipeline via `RequestPipeline::new_messages()`.

mod preparation;
mod request_building;
mod response_processing;

pub(crate) use preparation::MessagePreparationStage;
pub(crate) use request_building::MessageRequestBuildingStage;
pub(crate) use response_processing::MessageResponseProcessingStage;
