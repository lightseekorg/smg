//! Completion API endpoint pipeline stages
//!
//! This module continues the native `/v1/completions` stage stack. With scaffolding
//! in PR #840,preparation (#907), request building (#915), and now response processing,
//! three of the four endpoint-specific pipeline stages are complete. The shared
//! stages (worker selection, client acquisition, dispatch, execution) are reused
//! from the existing pipeline.

mod preparation;
mod request_building;
mod response_processing;

pub(crate) use preparation::CompletionPreparationStage;
#[expect(unused_imports, reason = "wired in pipeline factory follow-up PR")]
pub(crate) use request_building::CompletionRequestBuildingStage;
#[expect(unused_imports, reason = "wired in pipeline factory follow-up PR")]
pub(crate) use response_processing::CompletionResponseProcessingStage;
