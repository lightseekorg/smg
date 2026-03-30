//! Completion API endpoint pipeline stages
//!
//! This module contains the `/v1/completions` stage stack: preparation (#907),
//! request building (#915), and response processing. The shared stages
//! (worker selection, client acquisition, dispatch, execution) are reused
//! from the existing pipeline.

mod preparation;
mod request_building;
mod response_processing;

pub(crate) use preparation::CompletionPreparationStage;
pub(crate) use request_building::CompletionRequestBuildingStage;
pub(crate) use response_processing::CompletionResponseProcessingStage;
