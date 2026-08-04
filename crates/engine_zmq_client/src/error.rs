// Error variants for the vLLM EngineCore protocol are adapted from the
// Apache-2.0 reference `vllm-engine-core-client` (vllm-project/vllm,
// rust/src/engine-core-client/src/error.rs). See crate root for provenance.

use std::{sync::Arc, time::Duration};

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for the engine ZMQ client.
#[derive(Debug, Error)]
pub enum Error {
    #[error("messagepack encode failed for {target_type}: {message}")]
    Encode {
        target_type: &'static str,
        message: String,
    },
    #[error("messagepack decode failed for {target_type}: {message}")]
    Decode {
        target_type: &'static str,
        message: String,
    },
    #[error("messagepack value decode failed")]
    ValueDecode(#[from] rmpv::decode::Error),
    #[error("messagepack ext value decode failed: {message}")]
    ExtValueDecode { message: String },
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("transport error")]
    Transport(#[from] zeromq::ZmqError),
    #[error("ZMQ runtime task failed")]
    ZmqRuntimeTask(#[from] tokio::task::JoinError),
    #[error("engine core reported fatal failure")]
    EngineCoreDead,
    #[error("startup handshake timed out while waiting for {stage} after {timeout:?}")]
    HandshakeTimeout {
        stage: &'static str,
        timeout: Duration,
    },
    #[error("engine input registration timed out after {timeout:?}")]
    InputRegistrationTimeout { timeout: Duration },
    #[error("unexpected engine id in startup handshake: expected {expected:?}, got {actual:?}")]
    UnexpectedHandshakeIdentity { expected: Vec<u8>, actual: Vec<u8> },
    #[error("unexpected startup handshake message: {message}")]
    UnexpectedHandshakeMessage { message: String },
    #[error("unsupported auxiliary frame(s): expected {expected} frame(s), got {frame_count}")]
    UnsupportedAuxFrames { expected: usize, frame_count: usize },
    #[error("unsupported field `{field}` in {context}")]
    UnsupportedField {
        context: &'static str,
        field: &'static str,
    },
    #[error("request `{request_id}` is already in flight")]
    DuplicateRequestId { request_id: String },
    #[error("data parallel rank {rank} is out of range for {num_engines} engine(s)")]
    InvalidDataParallelRank { rank: u32, num_engines: u32 },
    #[error("request output stream for `{request_id}` closed unexpectedly")]
    RequestStreamClosed { request_id: String },
    #[error("engine-core output dispatcher closed: {message}")]
    DispatcherClosed { message: String },
    #[error("engine ZMQ client is closed: {message}")]
    ClientClosed { message: String },

    /// Allows cloning the same error across multiple request streams.
    #[error(transparent)]
    Shared(Arc<Self>),
}

impl Error {
    /// Construct an [`Error::Encode`] for `T`.
    pub fn encode<T>(err: impl std::fmt::Display) -> Self {
        Self::Encode {
            target_type: std::any::type_name::<T>(),
            message: err.to_string(),
        }
    }

    /// Construct an [`Error::Decode`] for `T`.
    pub fn decode<T>(err: impl std::fmt::Display) -> Self {
        Self::Decode {
            target_type: std::any::type_name::<T>(),
            message: err.to_string(),
        }
    }
}
