// SPDX-License-Identifier: Apache-2.0

//! vLLM EngineCore wire protocol.
//!
//! Clean-room port of vLLM's Apache-2.0 `vllm-engine-core-client`
//! (vllm-project/vllm, `rust/src/engine-core-client/src/protocol`). Struct
//! shapes, field order, and `array_like` positional-tuple encoding are the wire
//! contract with Python `EngineCoreProc` — do not reorder fields.
//!
//! Text generation is typed fully. Multimodal features, structured outputs
//! (guided decoding), and pooling params are carried as [`crate::codec::OpaqueValue`]
//! for now — they serialize as `nil` on the text path and get strongly typed in
//! the multimodal phase.

pub mod handshake;
pub mod logprobs;
pub mod lora;
pub mod output;
pub mod request;
pub mod sampling;
pub mod stats;
