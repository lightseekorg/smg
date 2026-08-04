// SPDX-License-Identifier: Apache-2.0

//! Per-engine wire protocol modules plus the engine-neutral seam the transport
//! and connector are generic over.
//!
//! vLLM's EngineCore protocol and TokenSpeed's native msgpack protocol
//! (issues #2003/#2006) live alongside each other. Both are msgpack-over-ZMQ
//! with a single-byte request-type frame and share the same transport — only
//! the message struct shapes differ, so the transport and connector are
//! parameterized over the [`EngineProtocol`] trait and each engine family
//! provides one implementation.

use bytes::Bytes;
use serde::de::{Deserialize, Error as _, IgnoredAny, SeqAccess};

use crate::Result;

pub mod handshake;
pub mod sglang;
pub mod tokenspeed;
pub mod vllm;

/// Read the next positional element of a msgspec `array_like` struct, failing
/// loudly when the array is shorter than the modeled prefix (every modeled field
/// is required on decode).
pub(crate) fn next_field<'de, A, T>(
    seq: &mut A,
    name: &'static str,
) -> std::result::Result<T, A::Error>
where
    A: SeqAccess<'de>,
    T: Deserialize<'de>,
{
    seq.next_element::<T>()?
        .ok_or_else(|| A::Error::custom(format!("missing positional field `{name}`")))
}

/// Validate the msgspec tag string at element 0. A wrong tag means the payload
/// is a different message type — fail loudly instead of misreading fields.
pub(crate) fn expect_tag<'de, A>(
    seq: &mut A,
    expected: &'static str,
) -> std::result::Result<(), A::Error>
where
    A: SeqAccess<'de>,
{
    let tag: String = next_field(seq, "_tag")?;
    if tag != expected {
        return Err(A::Error::custom(format!(
            "wrong msgspec tag: expected `{expected}`, got `{tag}`"
        )));
    }
    Ok(())
}

/// Drain positional elements beyond the modeled prefix. Engines append new
/// fields at the end of their structs over time, so unknown trailing elements
/// are skipped rather than treated as a decode error.
pub(crate) fn drain_trailing<'de, A>(seq: &mut A) -> std::result::Result<(), A::Error>
where
    A: SeqAccess<'de>,
{
    while seq.next_element::<IgnoredAny>()?.is_some() {}
    Ok(())
}

/// Engine-neutral per-rank load signal. Each protocol maps its native scheduler
/// stats into this shape (vLLM maps `SchedulerStats`; TokenSpeed carries none
/// yet), so the connector and gateway consume one load type regardless of
/// engine. Extend it as more engines report load (task #11).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EngineLoad {
    /// Requests currently in model-execution batches.
    pub num_running: u64,
    /// Requests waiting in the scheduler queue.
    pub num_waiting: u64,
    /// KV-cache usage ratio (`1.0` == 100%).
    pub kv_cache_usage: f64,
}

/// A single per-request output decoded from one engine tick. Lets the
/// engine-neutral connector route outputs to their request streams without
/// knowing the concrete protocol payload.
pub trait EngineOutput {
    /// The request id this output belongs to (the registry routing key).
    fn request_id(&self) -> &str;
    /// Whether this output is terminal for the request.
    fn finished(&self) -> bool;
}

/// One decoded engine tick: the per-request outputs plus the piggybacked load
/// signal and any out-of-band finished ids. The connector consumes this shape
/// for every protocol.
pub struct EngineBatch<O> {
    /// The DP rank / ZMQ index the batch came from.
    pub engine_index: u32,
    /// Per-request outputs produced in this tick.
    pub outputs: Vec<O>,
    /// Request ids the engine reported finished out-of-band.
    pub finished_request_ids: Vec<String>,
    /// Per-rank load signal, when the protocol carries it (`None` otherwise).
    pub load: Option<EngineLoad>,
}

impl<O> Default for EngineBatch<O> {
    fn default() -> Self {
        Self {
            engine_index: 0,
            outputs: Vec::new(),
            finished_request_ids: Vec::new(),
            load: None,
        }
    }
}

/// A per-engine wire protocol spoken over the shared ZMQ transport.
///
/// The transport and connector are generic over this trait; each engine family
/// (vLLM EngineCore, TokenSpeed) provides one zero-sized implementation. Only
/// the struct shapes and framing differ — the request registry, streaming,
/// auto-abort, and load tracking are shared.
pub trait EngineProtocol: Send + Sync + 'static {
    /// The typed add-request the connector submits.
    type Request: Send + 'static;
    /// The typed per-request output streamed back.
    type Output: EngineOutput + Send + 'static;

    /// The single-byte request-type frame prepended to an add-request, or
    /// `None` for protocols that dispatch by payload tag alone (no type frame).
    fn add_frame() -> Option<Bytes>;
    /// The single-byte request-type frame prepended to an abort, or `None` for
    /// protocols with no type frame.
    fn abort_frame() -> Option<Bytes>;

    /// The request's id (the registry routing key).
    fn request_id(request: &Self::Request) -> &str;
    /// SMG-pinned DP rank, if any (else the sole engine is used).
    fn data_parallel_rank(request: &Self::Request) -> Option<u32>;
    /// Reject fields this client cannot represent on the wire.
    fn validate(request: &Self::Request) -> Result<()>;
    /// Encode the add payload plus any aux tensor frames (empty on the text path).
    fn encode_add(request: &Self::Request) -> Result<(Vec<u8>, Vec<Bytes>)>;
    /// Encode the abort payload for one request id.
    fn encode_abort(request_id: &str) -> Result<Vec<u8>>;
    /// Decode one output message (frame 0 plus ordered aux frames) into a batch.
    fn decode_batch(frames: &[Bytes]) -> Result<EngineBatch<Self::Output>>;
}
