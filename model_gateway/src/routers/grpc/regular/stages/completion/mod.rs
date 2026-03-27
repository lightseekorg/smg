//! Completion API endpoint pipeline stages
//!
//! This module is the next stacked step after the native completion typing work
//! landed in PR #840. That PR introduced `RequestType::Completion`,
//! `FinalResponse::Completion`, and `execute_completion()`. This branch begins
//! the endpoint-specific stage stack with `CompletionPreparationStage`.
//!
//! Later follow-up PRs can add completion-specific request building and
//! response processing here, similar to the Messages API pipeline.

mod preparation;

pub(crate) use preparation::CompletionPreparationStage;
