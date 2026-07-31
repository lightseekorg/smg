// SPDX-License-Identifier: Apache-2.0

//! Per-engine wire protocol modules. vLLM's EngineCore protocol is first; the
//! sglang-family msgpack protocol (issues #2003/#2006) will live alongside it.

pub mod vllm;
