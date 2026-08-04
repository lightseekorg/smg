//! SGLang wire protocol.
//!
//! Clean-room port of SGLang's native inter-process messages
//! (`sglang/srt/managers/io_struct.py`, `sglang/srt/sampling/sampling_params.py`).
//! Every data-plane message is a Python `msgspec.Struct(tag=True, kw_only=True,
//! array_like=True)`: it rides the wire as a positional msgpack array whose
//! element 0 is the class-name tag string, followed by the fields in declaration
//! order. Field order is the wire contract — append only, never reorder. Decoders
//! here validate the tag and skip trailing fields they do not model (SGLang
//! appends fields over time); the encoder emits the shortest valid prefix, since
//! `msgspec` fills missing trailing fields from their defaults.
//!
//! Three structural differences from the TokenSpeed protocol drive this module:
//!
//! - **No request-type frame.** SGLang dispatches purely by the msgspec tag
//!   (element 0), so [`SglangProtocol::add_frame`] / [`SglangProtocol::abort_frame`]
//!   return `None` and the abort is a tagged [`AbortReq`] struct, not a bare
//!   id list.
//! - **`SamplingParams` is an untagged positional array.** It is
//!   `array_like=True` with no tag, so it rides the wire as a bare msgpack array
//!   in field order (not keyed by name and no leading tag element); SMG emits
//!   every field, carrying SGLang defaults for the ones it does not set, and the
//!   scheduler normalizes on receipt.
//! - **Token ids are typed arrays.** `input_ids` / `output_ids` use SGLang's
//!   `array.array('q', ...)` form — the 2-tuple `["q", <int64-LE bytes>]` — not
//!   a plain msgpack integer list (see [`token_ids`]).
//!
//! Only the text-generation (skip-tokenizer) path is typed.

pub mod output;
pub mod request;
pub mod sampling;
pub mod token_ids;

use bytes::Bytes;

use crate::{
    codec::{decode_msgpack, encode_msgpack},
    error::Result,
    protocol::{
        sglang::{
            output::{BatchTokenIDOutput, SglangOutput},
            request::{AbortReq, TokenizedGenerateReqInput},
        },
        EngineBatch, EngineProtocol,
    },
};

/// The SGLang engine protocol: drives [`TokenizedGenerateReqInput`] over the
/// tag-dispatched ZMQ transport and decodes [`BatchTokenIDOutput`] back.
pub struct SglangProtocol;

impl EngineProtocol for SglangProtocol {
    type Request = TokenizedGenerateReqInput;
    type Output = SglangOutput;

    fn add_frame() -> Option<Bytes> {
        // SGLang dispatches by the msgspec tag alone — no request-type frame.
        None
    }

    fn abort_frame() -> Option<Bytes> {
        None
    }

    fn request_id(request: &Self::Request) -> &str {
        &request.rid
    }

    fn data_parallel_rank(_request: &Self::Request) -> Option<u32> {
        // The modeled request prefix carries no DP-rank field, so requests route
        // to the sole engine (single-engine ZMQ). DP fan-out is future work.
        None
    }

    fn validate(_request: &Self::Request) -> Result<()> {
        // The tokenized text path has no fields this client cannot represent.
        Ok(())
    }

    fn encode_add(request: &Self::Request) -> Result<(Vec<u8>, Vec<Bytes>)> {
        // Text path carries no aux tensor frames.
        Ok((encode_msgpack(request)?, Vec::new()))
    }

    fn encode_abort(request_id: &str) -> Result<Vec<u8>> {
        // The abort is a tagged AbortReq struct (not a bare id list): the
        // scheduler matches `rid` against in-flight requests.
        encode_msgpack(&AbortReq::new(request_id))
    }

    fn decode_batch(frames: &[Bytes]) -> Result<EngineBatch<Self::Output>> {
        // Output messages are `[payload, aux...]`. The token-id batch carries no
        // tensor fields today, so aux frames are unexpected but not fatal.
        if frames.len() > 1 {
            tracing::debug!(
                aux_frames = frames.len() - 1,
                "ignoring aux frames on an SGLang output (BatchTokenIDOutput has \
                 no tensor fields on the text path)"
            );
        }
        let payload = frames.first().map(AsRef::as_ref).unwrap_or_default();
        let batch: BatchTokenIDOutput = decode_msgpack(payload)?;
        let outputs = batch.into_outputs()?;
        let finished_request_ids = outputs
            .iter()
            .filter(|output| output.finish_reason.is_some())
            .map(|output| output.request_id.clone())
            .collect();
        Ok(EngineBatch {
            // Single-engine ZMQ: SGLang batches carry no engine index and no
            // piggybacked scheduler load on this path.
            engine_index: 0,
            outputs,
            finished_request_ids,
            load: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::decode_msgpack;

    /// The pinned finished-variant output vector (see `output.rs`): one request
    /// "req-00000001" with `finished_reasons = [{"type":"stop","matched":2}]`.
    const PYTHON_OUTPUT_FINISHED_VECTOR: &str =
        "dc0032b24261746368546f6b656e49444f757470757491ac7265712d3030303030303031c091\
         82a474797065a473746f70a76d6174636865640291a09192a171c408883c0000000000009100\
         9192a171c408883c00000000000091c391c391c29103910091019100c0c0c0c0c0c0c0c0c0c0\
         c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0";

    fn from_hex(hex: &str) -> Vec<u8> {
        let hex: String = hex.chars().filter(|c| !c.is_whitespace()).collect();
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn no_request_type_frames() {
        assert!(SglangProtocol::add_frame().is_none());
        assert!(SglangProtocol::abort_frame().is_none());
    }

    #[test]
    fn encode_abort_emits_a_tagged_abort_req() {
        let payload = SglangProtocol::encode_abort("req-1").unwrap();
        let decoded: AbortReq = decode_msgpack(&payload).unwrap();
        assert_eq!(decoded, AbortReq::new("req-1"));
    }

    #[test]
    fn decode_batch_maps_outputs_and_finished_ids() {
        let frames = vec![Bytes::from(from_hex(PYTHON_OUTPUT_FINISHED_VECTOR))];
        let decoded = SglangProtocol::decode_batch(&frames).unwrap();
        assert_eq!(decoded.outputs.len(), 1);
        assert_eq!(decoded.outputs[0].request_id, "req-00000001");
        assert_eq!(decoded.outputs[0].output_ids, vec![15496]);
        assert_eq!(
            decoded.finished_request_ids,
            vec!["req-00000001".to_string()]
        );
        assert!(decoded.load.is_none());
    }

    #[test]
    fn decode_batch_tolerates_aux_frames() {
        let frames = vec![
            Bytes::from(from_hex(PYTHON_OUTPUT_FINISHED_VECTOR)),
            Bytes::from_static(b"opaque-aux-frame"),
        ];
        let decoded = SglangProtocol::decode_batch(&frames).unwrap();
        assert_eq!(decoded.outputs.len(), 1);
    }
}
