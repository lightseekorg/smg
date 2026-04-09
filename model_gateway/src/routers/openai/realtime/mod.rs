//! OpenAI Realtime API gateway implementation.
//!
//! Supports three transport mechanisms:
//! - **WebSocket** (server-to-server): Bidirectional WS proxy with transparent MCP interception
//! - **WebRTC** (browser-to-server): Dual peer-connection relay; SMG terminates both sides
//! - **REST**: Ephemeral token generation (`client_secrets`, `sessions`, `transcription_sessions`)

pub mod proxy;
pub mod registry;
pub mod rest;
pub mod webrtc_bridge;
pub mod ws;

pub use registry::RealtimeRegistry;
