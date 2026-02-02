//! OpenAI Realtime API protocol types.
//!
//! This module provides type definitions for the OpenAI Realtime API,
//! which enables real-time audio and text conversations with AI models.
//!
//! # Overview
//!
//! The Realtime API uses WebSocket or WebRTC connections for bidirectional
//! communication. This module defines:
//!
//! - **Session types**: Configuration and state for realtime sessions
//! - **Conversation types**: Messages, function calls, and content parts
//! - **Response types**: Response configuration and streaming state
//! - **Client events**: Events sent from client to server (9 types)
//! - **Server events**: Events received from server (28 types)
//!
//! # Example
//!
//! ```rust,ignore
//! use protocols::realtime::{
//!     RealtimeClientEvent, RealtimeServerEvent,
//!     SessionConfig, ConversationItem, Voice,
//! };
//!
//! // Create a session update event
//! let event = RealtimeClientEvent::session_update(SessionConfig {
//!     voice: Some(Voice::Alloy),
//!     instructions: Some("Be helpful and concise.".to_string()),
//!     ..Default::default()
//! });
//!
//! // Serialize to JSON for sending over WebSocket
//! let json = serde_json::to_string(&event).unwrap();
//!
//! // Parse incoming server event
//! let server_event: RealtimeServerEvent = serde_json::from_str(&json_str).unwrap();
//! if let RealtimeServerEvent::SessionCreated { session, .. } = server_event {
//!     println!("Session created: {}", session.id);
//! }
//! ```
//!
//! # Event Types
//!
//! ## Client Events (sent to server)
//!
//! | Event Type | Description |
//! |------------|-------------|
//! | `session.update` | Update session configuration |
//! | `input_audio_buffer.append` | Add audio to input buffer |
//! | `input_audio_buffer.commit` | Commit buffered audio as message |
//! | `input_audio_buffer.clear` | Clear the audio buffer |
//! | `conversation.item.create` | Add a conversation item |
//! | `conversation.item.truncate` | Truncate audio in an item |
//! | `conversation.item.delete` | Delete a conversation item |
//! | `response.create` | Request a model response |
//! | `response.cancel` | Cancel an in-progress response |
//!
//! ## Server Events (received from server)
//!
//! Session and conversation events, response streaming events, audio events,
//! function call events, and rate limit updates. See [`RealtimeServerEvent`]
//! for the full list.

pub mod client_events;
pub mod conversation;
pub mod response;
pub mod server_events;
pub mod session;

// Re-export all public types for convenience
pub use client_events::RealtimeClientEvent;
pub use conversation::{ContentPart, ConversationItem, ItemStatus, Role};
pub use response::{
    InputTokenDetails, OutputTokenDetails, Response, ResponseConfig, ResponseError,
    ResponseStatus, ResponseStatusDetails, ResponseUsage,
};
pub use server_events::{ApiError, Conversation, RateLimit, RealtimeServerEvent};
pub use session::{
    AudioFormat, InputAudioTranscription, MaxResponseOutputTokens, Modality, RealtimeTool,
    Session, SessionConfig, ToolChoice, ToolChoiceFunction, ToolChoiceFunctionName,
    ToolChoiceMode, TurnDetection, Voice,
};
