//! Messages API endpoint pipeline stages
//!
//! These stages handle Messages API-specific preprocessing.
//! Request building and response processing will be added in follow-up PRs.

mod preparation;

#[expect(unused_imports, reason = "wired in follow-up PR (pipeline factory)")]
pub(crate) use preparation::MessagePreparationStage;
