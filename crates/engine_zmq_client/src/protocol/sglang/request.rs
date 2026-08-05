// SGLang tokenized generate request and abort — the native
// `TokenizedGenerateReqInput` and `AbortReq` from `io_struct.py`, tagged
// `msgspec.Struct(array_like=True)`: each rides the wire as a positional
// msgpack array whose element 0 is the class-name tag string, followed by the
// fields in declaration order. **Field order is the wire contract** — do not
// reorder. Unlike TokenSpeed, SGLang dispatches purely by this tag (there is no
// single-byte request-type frame), so the abort is a tagged struct rather than
// a bare list of ids.

use serde::{
    de::{IgnoredAny, SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::protocol::{
    expect_tag, next_field,
    sglang::{sampling::SamplingParams, token_ids::TokenIdArray},
};

/// The msgspec tag for [`TokenizedGenerateReqInput`] (element 0 on the wire).
pub const TOKENIZED_GENERATE_REQ_INPUT_TAG: &str = "TokenizedGenerateReqInput";
/// The msgspec tag for [`AbortReq`] (element 0 on the wire).
pub const ABORT_REQ_TAG: &str = "AbortReq";

/// SGLang tokenized generate request sent from frontend to scheduler.
///
/// Models the leading prefix of the Python class, through `stream` — the fields
/// SMG sets on the token-id (skip-tokenizer) path. The encoder emits exactly
/// this 14-element array (tag + 13 fields); the scheduler's decoder fills every
/// later field from its defaults, since msgspec tolerates missing trailing
/// fields. The decoder here accepts full-length arrays and skips the unmodeled
/// trailing fields.
///
/// The three fields between `input_ids` and `sampling_params` — `input_embeds`,
/// `mm_inputs`, `token_type_ids` — are unused on the text path; they are emitted
/// as `nil` and skipped on decode, but must be present to keep the positional
/// layout aligned.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizedGenerateReqInput {
    /// Request id (the routing/registry key).
    pub rid: String,
    /// In-process HTTP-worker return address; unused on this transport.
    pub http_worker_ipc: Option<String>,
    /// Original prompt text. `None` on the token-id path (SMG detokenizes
    /// downstream of the engine, so only ids are sent).
    pub input_text: Option<String>,
    /// Pre-tokenized prompt token ids (SMG tokenizes upstream), encoded as
    /// SGLang's `array.array('q', ...)` wire form.
    pub input_ids: TokenIdArray,
    /// Sampling parameters (nested positional array).
    pub sampling_params: SamplingParams,
    /// Whether to return the sampled token's logprob for this request.
    pub return_logprob: bool,
    /// Prompt-logprob start offset. Neutral `-1`: prompt logprobs are not
    /// supported on this wire.
    pub logprob_start_len: i32,
    /// Output top-k logprob count. Neutral `0`: only the sampled token's
    /// logprob is materialized.
    pub top_logprobs_num: u32,
    /// Token ids to report logprobs for. Neutral `None`: not supported.
    pub token_ids_logprob: Option<Vec<u32>>,
    /// Whether to stream outputs incrementally.
    pub stream: bool,
}

impl Default for TokenizedGenerateReqInput {
    fn default() -> Self {
        Self {
            rid: String::new(),
            http_worker_ipc: None,
            input_text: None,
            input_ids: TokenIdArray::default(),
            sampling_params: SamplingParams::default(),
            return_logprob: false,
            logprob_start_len: -1,
            top_logprobs_num: 0,
            token_ids_logprob: None,
            stream: false,
        }
    }
}

impl Serialize for TokenizedGenerateReqInput {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(14)?;
        tuple.serialize_element(TOKENIZED_GENERATE_REQ_INPUT_TAG)?;
        tuple.serialize_element(&self.rid)?;
        tuple.serialize_element(&self.http_worker_ipc)?;
        tuple.serialize_element(&self.input_text)?;
        tuple.serialize_element(&self.input_ids)?;
        // input_embeds / mm_inputs / token_type_ids: unused on the text path.
        tuple.serialize_element(&None::<()>)?;
        tuple.serialize_element(&None::<()>)?;
        tuple.serialize_element(&None::<()>)?;
        tuple.serialize_element(&self.sampling_params)?;
        tuple.serialize_element(&self.return_logprob)?;
        tuple.serialize_element(&self.logprob_start_len)?;
        tuple.serialize_element(&self.top_logprobs_num)?;
        tuple.serialize_element(&self.token_ids_logprob)?;
        tuple.serialize_element(&self.stream)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for TokenizedGenerateReqInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ReqVisitor;

        impl<'de> Visitor<'de> for ReqVisitor {
            type Value = TokenizedGenerateReqInput;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a tagged TokenizedGenerateReqInput positional array")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                expect_tag(&mut seq, TOKENIZED_GENERATE_REQ_INPUT_TAG)?;
                let rid = next_field(&mut seq, "rid")?;
                let http_worker_ipc = next_field(&mut seq, "http_worker_ipc")?;
                let input_text = next_field(&mut seq, "input_text")?;
                let input_ids = next_field(&mut seq, "input_ids")?;
                // Consume the unused middle fields to keep positions aligned.
                next_field::<_, IgnoredAny>(&mut seq, "input_embeds")?;
                next_field::<_, IgnoredAny>(&mut seq, "mm_inputs")?;
                next_field::<_, IgnoredAny>(&mut seq, "token_type_ids")?;
                let request = TokenizedGenerateReqInput {
                    rid,
                    http_worker_ipc,
                    input_text,
                    input_ids,
                    sampling_params: next_field(&mut seq, "sampling_params")?,
                    return_logprob: next_field(&mut seq, "return_logprob")?,
                    logprob_start_len: next_field(&mut seq, "logprob_start_len")?,
                    top_logprobs_num: next_field(&mut seq, "top_logprobs_num")?,
                    token_ids_logprob: next_field(&mut seq, "token_ids_logprob")?,
                    stream: next_field(&mut seq, "stream")?,
                };
                // SGLang appends fields over time; skip everything past `stream`.
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(request)
            }
        }

        deserializer.deserialize_seq(ReqVisitor)
    }
}

/// SGLang abort request. On the tag-dispatched wire an abort is a tagged
/// [`AbortReq`] struct (not a bare id list): the scheduler matches `rid` against
/// in-flight requests. `abort_all` and the two message fields are unused by SMG.
#[derive(Debug, Clone, PartialEq)]
pub struct AbortReq {
    /// The request id to abort.
    pub rid: String,
    /// Whether to abort every in-flight request (SMG always aborts one rid).
    pub abort_all: bool,
}

impl AbortReq {
    /// An abort for a single request id.
    pub fn new(rid: impl Into<String>) -> Self {
        Self {
            rid: rid.into(),
            abort_all: false,
        }
    }
}

impl Serialize for AbortReq {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(6)?;
        tuple.serialize_element(ABORT_REQ_TAG)?;
        tuple.serialize_element(&self.rid)?;
        // http_worker_ipc / finished_reason / abort_message: unused by SMG.
        tuple.serialize_element(&None::<()>)?;
        tuple.serialize_element(&self.abort_all)?;
        tuple.serialize_element(&None::<()>)?;
        tuple.serialize_element(&None::<()>)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for AbortReq {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct AbortVisitor;

        impl<'de> Visitor<'de> for AbortVisitor {
            type Value = AbortReq;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a tagged AbortReq positional array")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                expect_tag(&mut seq, ABORT_REQ_TAG)?;
                let rid = next_field(&mut seq, "rid")?;
                next_field::<_, IgnoredAny>(&mut seq, "http_worker_ipc")?;
                let abort_all = next_field(&mut seq, "abort_all")?;
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(AbortReq { rid, abort_all })
            }
        }

        deserializer.deserialize_seq(AbortVisitor)
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;
    use crate::{
        codec::{decode_msgpack, decode_value, encode_msgpack},
        protocol::sglang::sampling::SamplingParams,
    };

    /// The pinned request vector captured from the Python encoder: rid
    /// "req-00000001", input_text "Hello, world", input_ids [9906, 11, 1917],
    /// sampling {max_new_tokens 64, temperature 0.7, top_p 0.9, top_k 50},
    /// stream true. The Python side emits all 45 elements (sampling_params is a
    /// nested 30-element positional array at index 8).
    const PYTHON_REQUEST_VECTOR: &str =
        "dc002db9546f6b656e697a656447656e6572617465526571496e707574ac7265712d30303030\
         30303031c0ac48656c6c6f2c20776f726c6492a171c418b2260000000000000b000000000000\
         007d07000000000000c0c0c0dc001e40c0c0c0cb3fe6666666666666cb3feccccccccccccd32c\
         b0000000000000000cb0000000000000000cb0000000000000000cb3ff0000000000000000\
         1c0c0c0c0c2c3c3c2c0c0c0c0c0c00000c2c2ff00c0c3c2c2c2c200c2c0c0c0c0c0c0c0c0c0c0\
         c0c0c0c2c0c0c2c2c2c0c0c0c0c0c0";

    fn from_hex(hex: &str) -> Vec<u8> {
        let hex: String = hex.chars().filter(|c| !c.is_whitespace()).collect();
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }

    fn vector_request() -> TokenizedGenerateReqInput {
        TokenizedGenerateReqInput {
            rid: "req-00000001".to_string(),
            input_text: Some("Hello, world".to_string()),
            input_ids: TokenIdArray(vec![9906, 11, 1917]),
            sampling_params: SamplingParams {
                max_new_tokens: Some(64),
                temperature: 0.7,
                top_p: 0.9,
                top_k: 50,
                ..SamplingParams::default()
            },
            stream: true,
            ..TokenizedGenerateReqInput::default()
        }
    }

    #[test]
    fn python_request_vector_decodes() {
        let decoded: TokenizedGenerateReqInput =
            decode_msgpack(&from_hex(PYTHON_REQUEST_VECTOR)).unwrap();
        assert_eq!(decoded, vector_request());
        assert_eq!(decoded.input_ids, TokenIdArray(vec![9906, 11, 1917]));
        assert_eq!(decoded.logprob_start_len, -1);
        assert!(decoded.stream);
        assert!(!decoded.return_logprob);
    }

    #[test]
    fn encoder_emits_tagged_prefix_through_stream() {
        let encoded = encode_msgpack(&vector_request()).unwrap();
        let Value::Array(array) = decode_value(&encoded).unwrap() else {
            panic!("expected positional array");
        };
        // tag + 13 fields; the three unused middle fields are nil.
        assert_eq!(array.len(), 14);
        assert_eq!(array[0], Value::from(TOKENIZED_GENERATE_REQ_INPUT_TAG));
        assert_eq!(array[1], Value::from("req-00000001"));
        assert_eq!(array[5], Value::Nil); // input_embeds
        assert_eq!(array[6], Value::Nil); // mm_inputs
        assert_eq!(array[7], Value::Nil); // token_type_ids
        assert_eq!(array[13], Value::from(true)); // stream

        // Round-trip: the prefix encoding decodes to the same request as the
        // full-length Python vector.
        let roundtripped: TokenizedGenerateReqInput = decode_msgpack(&encoded).unwrap();
        assert_eq!(roundtripped, vector_request());
        let from_vector: TokenizedGenerateReqInput =
            decode_msgpack(&from_hex(PYTHON_REQUEST_VECTOR)).unwrap();
        assert_eq!(roundtripped, from_vector);
    }

    #[test]
    fn request_decode_rejects_wrong_tag() {
        let mut bytes = from_hex(PYTHON_REQUEST_VECTOR);
        // Corrupt one tag byte inside "TokenizedGenerateReqInput".
        bytes[4] = b'X';
        let error = decode_msgpack::<TokenizedGenerateReqInput>(&bytes).unwrap_err();
        assert!(error.to_string().contains("wrong msgspec tag"), "{error}");
    }

    #[test]
    fn abort_req_matches_python_bytes() {
        // The pinned abort vector: [tag, "req-00000001", nil, false, nil, nil].
        const PYTHON_ABORT_VECTOR: &str = "96a841626f7274526571ac7265712d3030303030303031c0c2c0c0";
        let encoded = encode_msgpack(&AbortReq::new("req-00000001")).unwrap();
        assert_eq!(encoded, from_hex(PYTHON_ABORT_VECTOR));

        let decoded: AbortReq = decode_msgpack(&encoded).unwrap();
        assert_eq!(decoded, AbortReq::new("req-00000001"));
        assert!(!decoded.abort_all);
    }
}
