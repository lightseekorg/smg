// SGLang per-step batched output — the native `BatchTokenIDOutput` from
// `io_struct.py`, a tagged `msgspec.Struct(array_like=True)`: on the wire it is
// a positional msgpack array with the class-name tag string as element 0.
// **Field order is the wire contract** — do not reorder. SGLang carries far
// more columns than SMG consumes; the decoder reads the modeled prefix (through
// the output logprob columns), consuming the intervening columns positionally,
// and skips everything past it.

use serde::{
    de::{IgnoredAny, SeqAccess, Visitor},
    Deserialize, Deserializer,
};

use crate::{
    error::{Error, Result},
    protocol::{expect_tag, next_field, sglang::token_ids::TokenIdArray, EngineOutput},
};

/// The msgspec tag for [`BatchTokenIDOutput`] (element 0 on the wire).
pub const BATCH_TOKEN_ID_OUTPUT_TAG: &str = "BatchTokenIDOutput";

/// A finish reason entry — SGLang encodes each as a dict such as
/// `{"type": "stop", "matched": 2}` (or `nil` while the request is still
/// generating). Only the `type` discriminant matters to SMG; the extra keys
/// (`matched` / `length` / ...) are ignored on decode.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FinishReason {
    /// The finish-reason kind (`"stop"`, `"length"`, `"abort"`).
    #[serde(rename = "type")]
    pub kind: String,
}

/// A batch of per-request token outputs from one scheduler step. Every field is
/// a column indexed in parallel by request: `rids[i]` owns `output_ids[i]`,
/// `finished_reasons[i]`, and the token-count columns.
///
/// The logprob columns are the SGLang nesting `List[Optional[List[Optional[T]]]]`
/// — outer per request, inner per token — and are absent (`None`) unless the
/// request asked for logprobs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BatchTokenIDOutput {
    /// Request ids, one per column.
    pub rids: Vec<String>,
    /// Finish reason per request; `None` while still generating.
    pub finished_reasons: Vec<Option<FinishReason>>,
    /// Newly generated token ids per request this step.
    pub output_ids: Vec<TokenIdArray>,
    /// Prompt token count per request.
    pub prompt_tokens: Vec<u32>,
    /// Reasoning-phase token count per request.
    pub reasoning_tokens: Vec<u32>,
    /// Completion token count so far, per request.
    pub completion_tokens: Vec<u32>,
    /// Prefix-cache-hit token count per request.
    pub cached_tokens: Vec<u32>,
    /// Sampled-token logprob values per request (per-token inner list).
    pub output_token_logprobs_val: Option<Vec<Option<Vec<Option<f64>>>>>,
    /// Token id each logprob belongs to, per request (per-token inner list).
    pub output_token_logprobs_idx: Option<Vec<Option<Vec<Option<u32>>>>>,
}

impl<'de> Deserialize<'de> for BatchTokenIDOutput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct BatchVisitor;

        impl<'de> Visitor<'de> for BatchVisitor {
            type Value = BatchTokenIDOutput;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a tagged BatchTokenIDOutput positional array")
            }

            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                expect_tag(&mut seq, BATCH_TOKEN_ID_OUTPUT_TAG)?;
                // rids / http_worker_ipcs are the BaseBatchReq prefix.
                let rids =
                    next_field::<_, Option<Vec<String>>>(&mut seq, "rids")?.unwrap_or_default();
                next_field::<_, IgnoredAny>(&mut seq, "http_worker_ipcs")?;
                let finished_reasons = next_field(&mut seq, "finished_reasons")?;
                next_field::<_, IgnoredAny>(&mut seq, "decoded_texts")?;
                next_field::<_, IgnoredAny>(&mut seq, "decode_ids")?;
                next_field::<_, IgnoredAny>(&mut seq, "read_offsets")?;
                let output_ids =
                    next_field::<_, Option<Vec<TokenIdArray>>>(&mut seq, "output_ids")?
                        .unwrap_or_default();
                next_field::<_, IgnoredAny>(&mut seq, "skip_special_tokens")?;
                next_field::<_, IgnoredAny>(&mut seq, "spaces_between_special_tokens")?;
                next_field::<_, IgnoredAny>(&mut seq, "no_stop_trim")?;
                let prompt_tokens = next_field(&mut seq, "prompt_tokens")?;
                let reasoning_tokens = next_field(&mut seq, "reasoning_tokens")?;
                let completion_tokens = next_field(&mut seq, "completion_tokens")?;
                let cached_tokens = next_field(&mut seq, "cached_tokens")?;
                next_field::<_, IgnoredAny>(&mut seq, "input_token_logprobs_val")?;
                next_field::<_, IgnoredAny>(&mut seq, "input_token_logprobs_idx")?;
                let output_token_logprobs_val = next_field(&mut seq, "output_token_logprobs_val")?;
                let output_token_logprobs_idx = next_field(&mut seq, "output_token_logprobs_idx")?;
                // SGLang appends many more columns; skip everything past here.
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(BatchTokenIDOutput {
                    rids,
                    finished_reasons,
                    output_ids,
                    prompt_tokens,
                    reasoning_tokens,
                    completion_tokens,
                    cached_tokens,
                    output_token_logprobs_val,
                    output_token_logprobs_idx,
                })
            }
        }

        deserializer.deserialize_seq(BatchVisitor)
    }
}

/// One request's slice of a [`BatchTokenIDOutput`], in the engine-neutral shape
/// the connector routes to per-request streams.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SglangOutput {
    /// The request id this output belongs to.
    pub request_id: String,
    /// Newly generated token ids this step.
    pub output_ids: Vec<u32>,
    /// Finish reason; `None` while the request is still generating.
    pub finish_reason: Option<String>,
    /// Prompt token count.
    pub prompt_tokens: u32,
    /// Reasoning-phase token count.
    pub reasoning_tokens: u32,
    /// Completion token count so far.
    pub completion_tokens: u32,
    /// Prefix-cache-hit token count.
    pub cached_tokens: u32,
    /// Sampled-token logprob value per decoded token this step. Empty when
    /// logprobs were not requested.
    pub output_logprobs_val: Vec<f64>,
    /// Token id each logprob in `output_logprobs_val` belongs to. Empty when
    /// logprobs were not requested.
    pub output_logprobs_idx: Vec<u32>,
}

impl EngineOutput for SglangOutput {
    fn request_id(&self) -> &str {
        &self.request_id
    }

    fn finished(&self) -> bool {
        self.finish_reason.is_some()
    }
}

impl BatchTokenIDOutput {
    /// Split the parallel columns into one [`SglangOutput`] per request. Errors
    /// if the required columns are ragged (a length mismatch is a protocol bug).
    pub fn into_outputs(self) -> Result<Vec<SglangOutput>> {
        let n = self.rids.len();

        // `output_ids` may arrive as a whole-field `nil` (no new tokens this
        // step); treat that as an empty column so it stays aligned.
        let output_ids = if self.output_ids.is_empty() {
            vec![TokenIdArray::default(); n]
        } else {
            self.output_ids
        };

        let ragged = self.finished_reasons.len() != n
            || output_ids.len() != n
            || self.prompt_tokens.len() != n
            || self.reasoning_tokens.len() != n
            || self.completion_tokens.len() != n
            || self.cached_tokens.len() != n;
        if ragged {
            return Err(Error::Decode {
                target_type: "BatchTokenIDOutput",
                message: format!(
                    "ragged columns: rids={n}, finished_reasons={}, output_ids={}, \
                     prompt_tokens={}, reasoning_tokens={}, completion_tokens={}, \
                     cached_tokens={}",
                    self.finished_reasons.len(),
                    output_ids.len(),
                    self.prompt_tokens.len(),
                    self.reasoning_tokens.len(),
                    self.completion_tokens.len(),
                    self.cached_tokens.len(),
                ),
            });
        }

        let BatchTokenIDOutput {
            rids,
            finished_reasons,
            prompt_tokens,
            reasoning_tokens,
            completion_tokens,
            cached_tokens,
            output_token_logprobs_val,
            output_token_logprobs_idx,
            ..
        } = self;

        Ok(rids
            .into_iter()
            .zip(finished_reasons)
            .zip(output_ids)
            .zip(prompt_tokens)
            .zip(reasoning_tokens)
            .zip(completion_tokens)
            .zip(cached_tokens)
            .enumerate()
            .map(
                |(index, ((((((rid, reason), ids), prompt), reasoning), completion), cached))| {
                    SglangOutput {
                        request_id: rid,
                        output_ids: ids.0,
                        finish_reason: reason.map(|reason| reason.kind),
                        prompt_tokens: prompt,
                        reasoning_tokens: reasoning,
                        completion_tokens: completion,
                        cached_tokens: cached,
                        output_logprobs_val: per_request_logprobs(
                            output_token_logprobs_val.as_ref(),
                            index,
                        ),
                        output_logprobs_idx: per_request_logprobs(
                            output_token_logprobs_idx.as_ref(),
                            index,
                        ),
                    }
                },
            )
            .collect())
    }
}

/// Flatten request `index`'s logprob column into a dense vec, dropping the
/// per-token `Option` nesting (`None` inner values are omitted). Returns an
/// empty vec when the column is absent or the request had no logprobs.
fn per_request_logprobs<T: Copy>(
    column: Option<&Vec<Option<Vec<Option<T>>>>>,
    index: usize,
) -> Vec<T> {
    column
        .and_then(|rows| rows.get(index))
        .and_then(Option::as_ref)
        .map(|values| values.iter().filter_map(|value| *value).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;
    use crate::codec::{decode_msgpack, decode_value};

    fn from_hex(hex: &str) -> Vec<u8> {
        let hex: String = hex.chars().filter(|c| !c.is_whitespace()).collect();
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }

    /// The pinned output vector (still-generating variant): one request
    /// "req-00000001", output_ids [[15496]], finish nil, prompt 3 / completion 1.
    const PYTHON_OUTPUT_VECTOR: &str =
        "dc0032b24261746368546f6b656e49444f757470757491ac7265712d3030303030303031c091\
         c091a09192a171c408883c00000000000091009192a171c408883c00000000000091c391c391\
         c29103910091019100c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0\
         c0c0c0c0c0c0";

    /// The finished variant: finished_reasons carries `{"type":"stop","matched":2}`.
    const PYTHON_OUTPUT_FINISHED_VECTOR: &str =
        "dc0032b24261746368546f6b656e49444f757470757491ac7265712d3030303030303031c091\
         82a474797065a473746f70a76d6174636865640291a09192a171c408883c0000000000009100\
         9192a171c408883c00000000000091c391c391c29103910091019100c0c0c0c0c0c0c0c0c0c0\
         c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0";

    #[test]
    fn python_output_vector_decodes_still_generating() {
        let decoded: BatchTokenIDOutput = decode_msgpack(&from_hex(PYTHON_OUTPUT_VECTOR)).unwrap();
        assert_eq!(decoded.rids, vec!["req-00000001".to_string()]);
        assert_eq!(decoded.finished_reasons, vec![None]);
        assert_eq!(decoded.output_ids, vec![TokenIdArray(vec![15496])]);
        assert_eq!(decoded.prompt_tokens, vec![3]);
        assert_eq!(decoded.completion_tokens, vec![1]);

        let outputs = decoded.into_outputs().unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].request_id, "req-00000001");
        assert_eq!(outputs[0].output_ids, vec![15496]);
        assert_eq!(outputs[0].finish_reason, None);
        assert!(!outputs[0].finished());
        assert_eq!(outputs[0].prompt_tokens, 3);
        assert_eq!(outputs[0].completion_tokens, 1);
    }

    #[test]
    fn python_output_vector_decodes_finished() {
        let decoded: BatchTokenIDOutput =
            decode_msgpack(&from_hex(PYTHON_OUTPUT_FINISHED_VECTOR)).unwrap();
        assert_eq!(
            decoded.finished_reasons,
            vec![Some(FinishReason {
                kind: "stop".to_string()
            })]
        );

        let outputs = decoded.into_outputs().unwrap();
        assert_eq!(outputs[0].finish_reason.as_deref(), Some("stop"));
        assert!(outputs[0].finished());
    }

    #[test]
    fn decode_rejects_wrong_tag() {
        let mut bytes = from_hex(PYTHON_OUTPUT_VECTOR);
        // Corrupt a char inside the tag: "BatchTokenIDOutput" -> "BXtch...".
        bytes[5] = b'X';
        let error = decode_msgpack::<BatchTokenIDOutput>(&bytes).unwrap_err();
        assert!(error.to_string().contains("wrong msgspec tag"), "{error}");
    }

    #[test]
    fn into_outputs_rejects_ragged_columns() {
        let batch = BatchTokenIDOutput {
            rids: vec!["a".into(), "b".into()],
            finished_reasons: vec![None, None],
            output_ids: vec![TokenIdArray(vec![10])],
            prompt_tokens: vec![3, 4],
            reasoning_tokens: vec![0, 0],
            completion_tokens: vec![1, 1],
            cached_tokens: vec![0, 0],
            output_token_logprobs_val: None,
            output_token_logprobs_idx: None,
        };
        assert!(batch.into_outputs().is_err());
    }

    #[test]
    fn into_outputs_flattens_optional_logprobs() {
        let batch = BatchTokenIDOutput {
            rids: vec!["a".into()],
            finished_reasons: vec![None],
            output_ids: vec![TokenIdArray(vec![10, 11])],
            prompt_tokens: vec![3],
            reasoning_tokens: vec![0],
            completion_tokens: vec![2],
            cached_tokens: vec![0],
            output_token_logprobs_val: Some(vec![Some(vec![Some(-0.5), Some(-0.25)])]),
            output_token_logprobs_idx: Some(vec![Some(vec![Some(10), Some(11)])]),
        };
        let outputs = batch.into_outputs().unwrap();
        assert_eq!(outputs[0].output_logprobs_val, vec![-0.5, -0.25]);
        assert_eq!(outputs[0].output_logprobs_idx, vec![10, 11]);
    }

    #[test]
    fn output_vector_is_a_positional_array() {
        let Value::Array(array) = decode_value(&from_hex(PYTHON_OUTPUT_VECTOR)).unwrap() else {
            panic!("expected positional array");
        };
        assert_eq!(array[0], Value::from(BATCH_TOKEN_ID_OUTPUT_TAG));
    }
}
