// SGLang `SamplingParams` — a Python `msgspec.Struct(kw_only=True,
// array_like=True)` (sglang/srt/sampling/sampling_params.py), so it rides the
// wire as a positional msgpack array (no tag element), fields in declaration
// order. **Field order is the wire contract** — do not reorder.
//
// This is a faithful codec: [`SamplingParams::default`] mirrors SGLang's own
// class defaults so the positional array is always well-formed, and a producer
// sets only the fields it means to drive. Whether the internal (tokenizer-
// derived) fields are pre-resolved is the producer's policy, not the codec's —
// on the direct path SMG owns the tokenizer and sends them already resolved.

use std::collections::HashMap;

use serde::{
    de::{SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::codec::OpaqueValue;

/// SGLang's "consider the whole vocabulary" `top_k` sentinel (`1 << 30`). SMG
/// forwards this when no explicit cutoff is requested, matching the SGLang
/// default so the scheduler samples over the full distribution.
pub const TOP_K_ALL: i32 = 1 << 30;

/// The number of positional fields SGLang's `SamplingParams` encodes (the full
/// `array_like` struct is emitted, no `omit_defaults`).
const FIELD_COUNT: usize = 30;

/// Engine-facing sampling parameters for SGLang text generation.
///
/// Field order and count mirror the Python `SamplingParams` exactly: the struct
/// rides the wire as a 30-element positional array. Genuinely-optional SGLang
/// fields are `Option` (encoded as `nil` when unset); the rest carry concrete
/// values defaulting to the SGLang default.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    /// Maximum number of tokens to generate (SGLang default `128`, but
    /// `Optional`; `None` leaves it unbounded by this field).
    pub max_new_tokens: Option<u32>,
    /// Stop strings. SMG rejects stop strings upstream on this wire, so unset.
    pub stop: Option<Vec<String>>,
    /// Token ids that stop generation (a Python set; encoded as an array).
    pub stop_token_ids: Option<Vec<u32>>,
    /// Stop regexes. SMG rejects constraints upstream, so unset.
    pub stop_regex: Option<Vec<String>>,
    /// Controls randomness (SGLang default `1.0`).
    pub temperature: f64,
    /// Cumulative probability threshold for nucleus sampling (default `1.0`).
    pub top_p: f64,
    /// Maximum number of top tokens to consider. Defaults to [`TOP_K_ALL`].
    pub top_k: i32,
    /// Minimum probability threshold for token sampling (default `0.0`).
    pub min_p: f64,
    /// Frequency penalty applied by the sampler (default `0.0`).
    pub frequency_penalty: f64,
    /// Presence penalty applied by the sampler (default `0.0`).
    pub presence_penalty: f64,
    /// Repetition penalty applied by the sampler (default `1.0`).
    pub repetition_penalty: f64,
    /// Minimum number of tokens to generate before EOS / stop handling
    /// (default `0`).
    pub min_new_tokens: u32,
    /// OpenAI-compat fanout count. SMG fans out `n > 1` itself, so this is
    /// always `1` on the wire (default `1`).
    pub n: u32,
    /// Structured-output JSON schema. SMG rejects constraints upstream.
    pub json_schema: Option<String>,
    /// Structured-output regex. SMG rejects constraints upstream.
    pub regex: Option<String>,
    /// Structured-output EBNF grammar. SMG rejects constraints upstream.
    pub ebnf: Option<String>,
    /// Structured-output structural tag. SMG rejects constraints upstream.
    pub structural_tag: Option<String>,
    /// Ignore the EOS token and keep generating until another stop condition
    /// (default `false`).
    pub ignore_eos: bool,
    /// Whether detokenization skips special tokens (default `true`).
    pub skip_special_tokens: bool,
    /// Whether detokenization inserts spaces between special tokens
    /// (default `true`).
    pub spaces_between_special_tokens: bool,
    /// Whether stop sequences are kept in the output text (default `false`).
    pub no_stop_trim: bool,
    /// Streaming flush interval override. Not set by SMG.
    pub stream_interval: Option<u32>,
    /// Per-token logit bias, keyed by stringified token id. SMG rejects
    /// logit_bias upstream (no support on this backend).
    pub logit_bias: Option<HashMap<String, f64>>,
    /// Random seed. `None` lets the engine derive one so all ranks agree.
    pub sampling_seed: Option<u64>,
    /// Free-form engine extension parameters. Not set by SMG.
    pub custom_params: Option<OpaqueValue>,
    /// Normalized stop strings, resolved by the engine. Not set by SMG.
    pub stop_strs: Option<Vec<String>>,
    /// Normalized stop regexes, resolved by the engine. Not set by SMG.
    pub stop_regex_strs: Option<Vec<String>>,
    /// Longest stop string in tokens, resolved by the engine (default `0`).
    pub stop_str_max_len: u32,
    /// Longest stop regex in tokens, resolved by the engine (default `0`).
    pub stop_regex_max_len: u32,
    /// True once the tokenizer-derived fields (`stop_strs`, `stop_str_max_len`,
    /// …) are resolved. Defaults to `false` to match SGLang's class default;
    /// producers that resolve those fields themselves set it `true`.
    pub is_normalized: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        // Mirror the SGLang class defaults so an untouched instance encodes to
        // the same positional array SGLang itself would emit.
        Self {
            max_new_tokens: Some(128),
            stop: None,
            stop_token_ids: None,
            stop_regex: None,
            temperature: 1.0,
            top_p: 1.0,
            top_k: TOP_K_ALL,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            min_new_tokens: 0,
            n: 1,
            json_schema: None,
            regex: None,
            ebnf: None,
            structural_tag: None,
            ignore_eos: false,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            no_stop_trim: false,
            stream_interval: None,
            logit_bias: None,
            sampling_seed: None,
            custom_params: None,
            stop_strs: None,
            stop_regex_strs: None,
            stop_str_max_len: 0,
            stop_regex_max_len: 0,
            is_normalized: false,
        }
    }
}

impl Serialize for SamplingParams {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(FIELD_COUNT)?;
        tuple.serialize_element(&self.max_new_tokens)?;
        tuple.serialize_element(&self.stop)?;
        tuple.serialize_element(&self.stop_token_ids)?;
        tuple.serialize_element(&self.stop_regex)?;
        tuple.serialize_element(&self.temperature)?;
        tuple.serialize_element(&self.top_p)?;
        tuple.serialize_element(&self.top_k)?;
        tuple.serialize_element(&self.min_p)?;
        tuple.serialize_element(&self.frequency_penalty)?;
        tuple.serialize_element(&self.presence_penalty)?;
        tuple.serialize_element(&self.repetition_penalty)?;
        tuple.serialize_element(&self.min_new_tokens)?;
        tuple.serialize_element(&self.n)?;
        tuple.serialize_element(&self.json_schema)?;
        tuple.serialize_element(&self.regex)?;
        tuple.serialize_element(&self.ebnf)?;
        tuple.serialize_element(&self.structural_tag)?;
        tuple.serialize_element(&self.ignore_eos)?;
        tuple.serialize_element(&self.skip_special_tokens)?;
        tuple.serialize_element(&self.spaces_between_special_tokens)?;
        tuple.serialize_element(&self.no_stop_trim)?;
        tuple.serialize_element(&self.stream_interval)?;
        tuple.serialize_element(&self.logit_bias)?;
        tuple.serialize_element(&self.sampling_seed)?;
        tuple.serialize_element(&self.custom_params)?;
        tuple.serialize_element(&self.stop_strs)?;
        tuple.serialize_element(&self.stop_regex_strs)?;
        tuple.serialize_element(&self.stop_str_max_len)?;
        tuple.serialize_element(&self.stop_regex_max_len)?;
        tuple.serialize_element(&self.is_normalized)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for SamplingParams {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ParamsVisitor;

        impl<'de> Visitor<'de> for ParamsVisitor {
            type Value = SamplingParams;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a SamplingParams positional array")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let default = SamplingParams::default();
                // Each position falls back to the SGLang default when the array
                // is shorter than the full field list (msgspec omits nothing
                // today, but tolerate short arrays for forward compatibility).
                macro_rules! field {
                    ($name:ident) => {
                        seq.next_element()?.unwrap_or(default.$name)
                    };
                }
                let params = SamplingParams {
                    max_new_tokens: seq.next_element()?.unwrap_or(default.max_new_tokens),
                    stop: seq.next_element()?.unwrap_or(default.stop),
                    stop_token_ids: seq.next_element()?.unwrap_or(default.stop_token_ids),
                    stop_regex: seq.next_element()?.unwrap_or(default.stop_regex),
                    temperature: field!(temperature),
                    top_p: field!(top_p),
                    top_k: field!(top_k),
                    min_p: field!(min_p),
                    frequency_penalty: field!(frequency_penalty),
                    presence_penalty: field!(presence_penalty),
                    repetition_penalty: field!(repetition_penalty),
                    min_new_tokens: field!(min_new_tokens),
                    n: field!(n),
                    json_schema: seq.next_element()?.unwrap_or(default.json_schema),
                    regex: seq.next_element()?.unwrap_or(default.regex),
                    ebnf: seq.next_element()?.unwrap_or(default.ebnf),
                    structural_tag: seq.next_element()?.unwrap_or(default.structural_tag),
                    ignore_eos: field!(ignore_eos),
                    skip_special_tokens: field!(skip_special_tokens),
                    spaces_between_special_tokens: field!(spaces_between_special_tokens),
                    no_stop_trim: field!(no_stop_trim),
                    stream_interval: seq.next_element()?.unwrap_or(default.stream_interval),
                    logit_bias: seq.next_element()?.unwrap_or(default.logit_bias),
                    sampling_seed: seq.next_element()?.unwrap_or(default.sampling_seed),
                    custom_params: seq.next_element()?.unwrap_or(default.custom_params),
                    stop_strs: seq.next_element()?.unwrap_or(default.stop_strs),
                    stop_regex_strs: seq.next_element()?.unwrap_or(default.stop_regex_strs),
                    stop_str_max_len: field!(stop_str_max_len),
                    stop_regex_max_len: field!(stop_regex_max_len),
                    is_normalized: field!(is_normalized),
                };
                // SGLang appends fields over time; skip everything past the
                // modeled prefix.
                while seq.next_element::<serde::de::IgnoredAny>()?.is_some() {}
                Ok(params)
            }
        }

        deserializer.deserialize_seq(ParamsVisitor)
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;
    use crate::codec::{decode_msgpack, decode_value, encode_msgpack};

    /// The pinned SamplingParams array captured from the installed SGLang
    /// encoder (`msgspec.msgpack`, `SGLANG_USE_PICKLE_IPC=0`) for
    /// `SamplingParams(max_new_tokens=64, temperature=0.7, top_p=0.9, top_k=50)`
    /// — a 30-element positional array.
    const PYTHON_SAMPLING_VECTOR: &str =
        "dc001e40c0c0c0cb3fe6666666666666cb3feccccccccccccd32cb0000000000000000cb00000000\
         00000000cb0000000000000000cb3ff00000000000000001c0c0c0c0c2c3c3c2c0c0c0c0c0c00000c2";

    fn python_sampling_bytes() -> Vec<u8> {
        let hex: String = PYTHON_SAMPLING_VECTOR
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }

    fn vector_sampling() -> SamplingParams {
        SamplingParams {
            max_new_tokens: Some(64),
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            ..SamplingParams::default()
        }
    }

    #[test]
    fn python_sampling_vector_decodes() {
        let decoded: SamplingParams = decode_msgpack(&python_sampling_bytes()).unwrap();
        assert_eq!(decoded, vector_sampling());
        // Fields SMG never set carry the SGLang defaults.
        assert_eq!(decoded.min_p, 0.0);
        assert_eq!(decoded.n, 1);
        assert!(!decoded.is_normalized);
    }

    #[test]
    fn encoder_matches_python_bytes() {
        let encoded = encode_msgpack(&vector_sampling()).unwrap();
        assert_eq!(encoded, python_sampling_bytes());
    }

    #[test]
    fn encodes_as_a_positional_array() {
        let encoded = encode_msgpack(&vector_sampling()).unwrap();
        let Value::Array(elements) = decode_value(&encoded).unwrap() else {
            panic!("expected a msgpack array (array_like=True struct)");
        };
        assert_eq!(elements.len(), FIELD_COUNT);
        assert_eq!(elements[0], Value::from(64u32)); // max_new_tokens
        assert_eq!(elements[6], Value::from(50)); // top_k
        assert_eq!(elements[12], Value::from(1)); // n
        assert_eq!(elements[29], Value::from(false)); // is_normalized
    }

    #[test]
    fn roundtrips_through_the_array_wire_form() {
        let params = SamplingParams {
            max_new_tokens: Some(128),
            stop_token_ids: Some(vec![2, 3]),
            temperature: 0.5,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            frequency_penalty: 0.1,
            sampling_seed: Some(42),
            ignore_eos: true,
            ..SamplingParams::default()
        };
        let encoded = encode_msgpack(&params).unwrap();
        assert_eq!(decode_msgpack::<SamplingParams>(&encoded).unwrap(), params);
    }

    #[test]
    fn default_encodes_all_fields() {
        let encoded = encode_msgpack(&SamplingParams::default()).unwrap();
        let Value::Array(elements) = decode_value(&encoded).unwrap() else {
            panic!("expected a msgpack array");
        };
        assert_eq!(elements.len(), FIELD_COUNT);
        assert_eq!(elements[6], Value::from(TOP_K_ALL)); // default top_k sentinel
    }
}
