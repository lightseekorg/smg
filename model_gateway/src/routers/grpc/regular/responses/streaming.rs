//! Legacy streaming module — fully replaced by
//! `agent_streaming_adapter::RegularStreamingAdapter` plus
//! `handlers::route_responses_streaming`.
//!
//! Kept as an empty file so the surrounding `mod streaming;`
//! declaration in `mod.rs` continues to type-check during the rollout
//! period; the next reorganization can remove the module entry
//! outright.
