//! Messages API endpoint pipeline stages
//!
//! These stages handle Messages API-specific preprocessing and request building.
//! Response processing will be added in a follow-up PR.

mod preparation;
mod request_building;

#[expect(unused_imports, reason = "wired in follow-up PR (pipeline factory)")]
pub(crate) use preparation::MessagePreparationStage;
#[expect(unused_imports, reason = "wired in follow-up PR (pipeline factory)")]
pub(crate) use request_building::MessageRequestBuildingStage;
