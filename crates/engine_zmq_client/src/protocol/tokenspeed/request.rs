// TokenSpeed tokenized generate request — the native `TokenizedGenerateReqInput`
// from `io_struct.py`, a tagged `msgspec.Struct(array_like=True)`: on the wire
// it is a positional msgpack array with the class-name tag string as element 0.
// **Field order is the wire contract** — do not reorder.

use bytes::Bytes;
use serde::{
    de::{SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::protocol::tokenspeed::{
    drain_trailing, expect_tag, next_field, sampling::SamplingParams,
};

/// The msgspec tag for [`TokenizedGenerateReqInput`] (element 0 on the wire).
pub const TOKENIZED_GENERATE_REQ_INPUT_TAG: &str = "TokenizedGenerateReqInput";

/// Request types are single-byte protocol constants sent as a raw ZMQ frame
/// ahead of the msgpack payload (TokenSpeed's `REQ_TYPE_ADD`/`REQ_TYPE_ABORT`),
/// so the receiver can dispatch without decoding first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TokenSpeedRequestType {
    Add = 0,
    Abort = 1,
}

impl TokenSpeedRequestType {
    /// Decode the single-byte request-type frame. `None` for unrecognized values.
    pub fn from_frame(frame: &[u8]) -> Option<Self> {
        let [value] = frame else {
            return None;
        };
        match value {
            0 => Some(Self::Add),
            1 => Some(Self::Abort),
            _ => None,
        }
    }

    /// Encode as the single-byte frame used on the engine input socket.
    pub fn to_frame(self) -> Bytes {
        Bytes::from_static(match self {
            Self::Add => b"\x00",
            Self::Abort => b"\x01",
        })
    }
}

/// TokenSpeed tokenized generate request sent from frontend to engine.
///
/// Models the leading prefix of the Python class, through `stream` — the
/// fields SMG sets, plus the neutral logprob carry-over values between them.
/// The encoder emits exactly this 11-element array (tag + 10 fields); the
/// engine's decoder fills every later field (`input_embeds`, `session_params`,
/// multimodal payloads, ...) from its defaults, since msgspec tolerates
/// missing trailing fields. The decoder here accepts full-length arrays and
/// skips the unmodeled trailing fields.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizedGenerateReqInput {
    /// Request id (the routing/registry key).
    pub rid: String,
    /// In-process HTTP-worker return address; unused on this transport.
    pub http_worker_ipc: Option<String>,
    /// Original prompt text. `None` on the token-id path (SMG detokenizes
    /// downstream of the engine, so only ids are sent).
    pub input_text: Option<String>,
    /// Pre-tokenized prompt token ids (SMG tokenizes upstream).
    pub input_ids: Vec<u32>,
    /// Sampling parameters (nested untagged positional array).
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
            input_ids: Vec::new(),
            sampling_params: SamplingParams::default(),
            return_logprob: false,
            // Neutral logprob carry-over values, mirroring the engine's own
            // input processor.
            logprob_start_len: -1,
            top_logprobs_num: 0,
            token_ids_logprob: None,
            stream: false,
        }
    }
}

impl Serialize for TokenizedGenerateReqInput {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(11)?;
        tuple.serialize_element(TOKENIZED_GENERATE_REQ_INPUT_TAG)?;
        tuple.serialize_element(&self.rid)?;
        tuple.serialize_element(&self.http_worker_ipc)?;
        tuple.serialize_element(&self.input_text)?;
        tuple.serialize_element(&self.input_ids)?;
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
                let request = TokenizedGenerateReqInput {
                    rid: next_field(&mut seq, "rid")?,
                    http_worker_ipc: next_field(&mut seq, "http_worker_ipc")?,
                    input_text: next_field(&mut seq, "input_text")?,
                    input_ids: next_field(&mut seq, "input_ids")?,
                    sampling_params: next_field(&mut seq, "sampling_params")?,
                    return_logprob: next_field(&mut seq, "return_logprob")?,
                    logprob_start_len: next_field(&mut seq, "logprob_start_len")?,
                    top_logprobs_num: next_field(&mut seq, "top_logprobs_num")?,
                    token_ids_logprob: next_field(&mut seq, "token_ids_logprob")?,
                    stream: next_field(&mut seq, "stream")?,
                };
                drain_trailing(&mut seq)?;
                Ok(request)
            }
        }

        deserializer.deserialize_seq(ReqVisitor)
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;
    use crate::{
        codec::{decode_msgpack, decode_value, encode_msgpack},
        protocol::tokenspeed::sampling::TOP_K_DISABLED,
    };

    /// A full-length (24-element) request array captured from the Python
    /// encoder: rid "vec-1", input_ids [1, 2, 3], normalized SamplingParams
    /// (temperature 0.5, top_p 0.9, top_k disabled, max_new_tokens 8,
    /// stop_token_ids {2}, seed 42), return_logprob, stream.
    const PYTHON_REQUEST_VECTOR: &str =
        "dc0018b9546f6b656e697a656447656e6572617465526571496e707574a57665632d31c0c0\
         93010203dc001d08c09102cb3fe0000000000000cb3feccccccccccccdce40000000cb0000\
         000000000000cb0000000000000000cb0000000000000000cb3ff000000000000000c0c0c0\
         c0c2c3c3c2c0c0c0c02ac0019000c3c3ff00c0c3c0c0c0c2cb0000000000000000c0c0c0c0\
         c0c0c0c0";

    fn python_request_bytes() -> Vec<u8> {
        let hex: String = PYTHON_REQUEST_VECTOR
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }

    /// The logical request behind [`PYTHON_REQUEST_VECTOR`], built through the
    /// same normalization path SMG uses.
    fn vector_request() -> TokenizedGenerateReqInput {
        let mut sampling_params = SamplingParams {
            max_new_tokens: Some(8),
            stop_token_ids: Some(vec![2]),
            temperature: 0.5,
            top_p: 0.9,
            seed: Some(42),
            ..SamplingParams::default()
        };
        sampling_params.normalize();
        TokenizedGenerateReqInput {
            rid: "vec-1".to_string(),
            input_ids: vec![1, 2, 3],
            sampling_params,
            return_logprob: true,
            stream: true,
            ..TokenizedGenerateReqInput::default()
        }
    }

    #[test]
    fn request_type_frames_roundtrip() {
        for ty in [TokenSpeedRequestType::Add, TokenSpeedRequestType::Abort] {
            assert_eq!(TokenSpeedRequestType::from_frame(&ty.to_frame()), Some(ty));
        }
        assert_eq!(TokenSpeedRequestType::Add.to_frame().as_ref(), b"\x00");
        assert_eq!(TokenSpeedRequestType::Abort.to_frame().as_ref(), b"\x01");
        assert_eq!(TokenSpeedRequestType::from_frame(b"\x09"), None);
        assert_eq!(TokenSpeedRequestType::from_frame(b""), None);
    }

    /// The pinned cross-language vector decodes field-for-field: the Python
    /// side emits all 24 elements, and the modeled 11-element prefix plus
    /// trailing-field skip must recover every field SMG cares about.
    #[test]
    fn python_request_vector_decodes() {
        let decoded: TokenizedGenerateReqInput = decode_msgpack(&python_request_bytes()).unwrap();
        assert_eq!(decoded.rid, "vec-1");
        assert_eq!(decoded.http_worker_ipc, None);
        assert_eq!(decoded.input_text, None);
        assert_eq!(decoded.input_ids, vec![1, 2, 3]);
        assert!(decoded.return_logprob);
        assert_eq!(decoded.logprob_start_len, -1);
        assert_eq!(decoded.top_logprobs_num, 0);
        assert_eq!(decoded.token_ids_logprob, None);
        assert!(decoded.stream);

        let sp = &decoded.sampling_params;
        assert_eq!(sp.max_new_tokens, Some(8));
        assert_eq!(sp.stop_token_ids, Some(vec![2]));
        assert_eq!(sp.temperature, 0.5);
        assert_eq!(sp.top_p, 0.9);
        assert_eq!(sp.top_k, TOP_K_DISABLED);
        assert_eq!(sp.seed, Some(42));
        assert_eq!(sp.n, 1);
        assert_eq!(sp.stop_strs, Vec::<String>::new());
        assert!(sp.is_normalized);

        // Full struct equality against the same logical request built in Rust.
        assert_eq!(decoded, vector_request());
    }

    /// The encoder emits the shortest valid prefix: an 11-element array
    /// (tag + fields through `stream`) that decodes back to the same request
    /// the full-length Python vector decodes to. The nested sampling params
    /// are always sent in full (29 elements), since `is_normalized` — the
    /// last field — is always set.
    #[test]
    fn encoder_emits_tagged_prefix_through_stream() {
        let request = vector_request();
        let encoded = encode_msgpack(&request).unwrap();

        let Value::Array(array) = decode_value(&encoded).unwrap() else {
            panic!("expected positional array");
        };
        assert_eq!(array.len(), 11);
        assert_eq!(array[0], Value::from(TOKENIZED_GENERATE_REQ_INPUT_TAG));
        assert_eq!(array[1], Value::from("vec-1"));
        let Value::Array(sampling) = &array[5] else {
            panic!("expected nested sampling params array, got {:?}", array[5]);
        };
        assert_eq!(sampling.len(), 29);
        assert_eq!(array[10], Value::from(true)); // stream

        // Round-trip: the prefix encoding is semantically identical to the
        // full-length Python encoding.
        let roundtripped: TokenizedGenerateReqInput = decode_msgpack(&encoded).unwrap();
        assert_eq!(roundtripped, request);
        let from_vector: TokenizedGenerateReqInput =
            decode_msgpack(&python_request_bytes()).unwrap();
        assert_eq!(roundtripped, from_vector);
    }

    #[test]
    fn decode_rejects_wrong_tag() {
        let mut request_bytes = python_request_bytes();
        // Corrupt one tag byte: "TokenizedGenerateReqInput" -> "TOkenized...".
        request_bytes[5] = b'O';
        let error = decode_msgpack::<TokenizedGenerateReqInput>(&request_bytes).unwrap_err();
        let rendered = error.to_string();
        assert!(rendered.contains("wrong msgspec tag"), "{rendered}");
        assert!(rendered.contains("TokenizedGenerateReqInput"), "{rendered}");
    }

    #[test]
    fn decode_rejects_truncated_prefix() {
        // Missing modeled fields (here: everything after input_ids) must fail
        // loudly, not silently default.
        let truncated = Value::Array(vec![
            Value::from(TOKENIZED_GENERATE_REQ_INPUT_TAG),
            Value::from("r1"),
            Value::Nil,
            Value::Nil,
            Value::Array(vec![Value::from(1)]),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &truncated).unwrap();
        let error = decode_msgpack::<TokenizedGenerateReqInput>(&bytes).unwrap_err();
        assert!(
            error.to_string().contains("missing positional field"),
            "{error}"
        );
    }
}
