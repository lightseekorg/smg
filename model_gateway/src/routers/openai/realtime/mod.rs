//! OpenAI Realtime API gateway implementation.
//!
//! Supports three transport mechanisms:
//! - **WebSocket** (server-to-server): Bidirectional WS proxy with transparent MCP interception
//! - **WebRTC** (browser-to-server): SDP signaling proxy; media + data channel flow directly
//! - **REST**: Ephemeral token generation (`client_secrets`, `sessions`, `transcription_sessions`)

pub mod registry;
pub mod rest;

pub use registry::RealtimeRegistry;
