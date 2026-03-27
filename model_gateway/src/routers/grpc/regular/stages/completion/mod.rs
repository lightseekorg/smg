//! Completion API endpoint pipeline stages
//!
//! This module continues the native `/v1/completions` stage stack after the
//! scaffolding in PR #840 and preparation in PR #907. It adds completion-specific
//! request building, with response processing deferred to a follow-up PR.

mod preparation;
mod request_building;

pub(crate) use preparation::CompletionPreparationStage;
#[expect(unused_imports, reason = "wired in pipeline factory follow-up PR")]
pub(crate) use request_building::CompletionRequestBuildingStage;
