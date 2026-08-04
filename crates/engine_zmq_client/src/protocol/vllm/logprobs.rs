// Ported from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/logprobs.rs + protocol/logprobs/wire.rs.
// The numpy-array decoders live in `crate::codec::tensor`.

use serde::{Deserialize, Deserializer, Serialize};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use crate::{
    codec::tensor::{
        decode_array1_u32, decode_array2_f32, decode_array2_u32, WireArrayData, WireNdArray,
    },
    error::{Error, Result},
};

/// One token candidate and its logprob metadata for a single sequence position.
///
/// The first entry in a [`PositionLogprobs`] is always the sampled/selected
/// token; remaining entries follow the engine's returned top-k candidate order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    pub token_id: u32,
    pub logprob: f32,
    /// The sampled/selected token uses its actual vocab rank; remaining entries
    /// use 1-based top-k ranks matching the engine's candidate order.
    pub rank: u32,
}

/// Logprob payload for one sequence position (semantic, post-decode form).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositionLogprobs {
    pub entries: Vec<TokenLogprob>,
}

impl PositionLogprobs {
    /// Group one decoded logprobs row into per-position form, attaching the
    /// sampled/selected token's actual vocab rank to entry 0.
    fn from_decoded_row(token_ids: &[u32], logprobs: &[f32], sampled_rank: u32) -> Result<Self> {
        if token_ids.len() != logprobs.len() {
            return Err(Error::ExtValueDecode {
                message: format!(
                    "logprobs row length mismatch: token_ids={}, logprobs={}",
                    token_ids.len(),
                    logprobs.len()
                ),
            });
        }
        if sampled_rank == 0 {
            return Err(Error::ExtValueDecode {
                message: "token_ranks must be >= 1 for decoded engine-core logprobs".to_string(),
            });
        }

        let mut entries = Vec::with_capacity(token_ids.len());
        for (index, (&token_id, &logprob)) in token_ids.iter().zip(logprobs.iter()).enumerate() {
            let rank = if index == 0 {
                sampled_rank
            } else {
                index as u32
            };
            entries.push(TokenLogprob {
                token_id,
                logprob,
                rank,
            });
        }
        Ok(Self { entries })
    }
}

/// Decoded per-request logprobs payload for one engine-core output: one
/// [`PositionLogprobs`] per scored position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Logprobs {
    pub positions: Vec<PositionLogprobs>,
}

impl Logprobs {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// Python wire representation of `LogprobsLists`/`LogprobsTensors` before
/// aux-frame references and raw-view payloads are resolved. Mirrors Python
/// `vllm/v1/outputs.py`.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct WireLogprobs {
    /// Wire array with shape `[num_positions, max_num_logprobs + 1]`.
    pub logprob_token_ids: WireNdArray,
    /// Wire array with shape `[num_positions, max_num_logprobs + 1]`.
    pub logprobs: WireNdArray,
    /// Wire array with shape `[num_positions]`. Python names it
    /// `sampled_token_ranks` (sample logprobs) / `selected_token_ranks` (prompt
    /// logprobs); one neutral field here since both share the wire shape.
    pub token_ranks: WireNdArray,
    /// Preserved only for wire compatibility with batch-level Python tensors.
    /// Scheduler-sliced per-request outputs emit `None`; any other value is
    /// rejected by the semantic decoder.
    #[serde(default)]
    pub cu_num_generated_tokens: Option<Vec<usize>>,
}

impl WireLogprobs {
    /// Convert semantic per-position logprobs into the Python wire tuple shape.
    /// Exists mainly so tests can inject semantic logprobs without hand-building
    /// ndarray raw-view tuples.
    fn from_direct(value: &Logprobs) -> std::result::Result<Self, String> {
        let rows = value.positions.len();
        let cols = value
            .positions
            .first()
            .map(|position| position.entries.len())
            .unwrap_or(0);

        let mut token_ids = Vec::with_capacity(rows.saturating_mul(cols).saturating_mul(8));
        let mut logprobs = Vec::with_capacity(rows.saturating_mul(cols).saturating_mul(4));
        let mut token_ranks = Vec::with_capacity(rows.saturating_mul(8));

        for (row_index, position) in value.positions.iter().enumerate() {
            if position.entries.len() != cols {
                return Err(format!(
                    "logprobs row {row_index} length mismatch: expected {cols}, got {}",
                    position.entries.len()
                ));
            }
            let Some((sampled, _)) = position.entries.split_first() else {
                return Err(format!("logprobs row {row_index} is empty"));
            };

            token_ranks.extend_from_slice(&(sampled.rank as i64).to_le_bytes());
            for entry in &position.entries {
                token_ids.extend_from_slice(&(entry.token_id as i64).to_le_bytes());
                logprobs.extend_from_slice(&entry.logprob.to_le_bytes());
            }
        }

        Ok(Self {
            logprob_token_ids: WireNdArray {
                dtype: "<i8".to_string(),
                shape: vec![rows, cols],
                data: WireArrayData::RawView(token_ids.into()),
            },
            logprobs: WireNdArray {
                dtype: "<f4".to_string(),
                shape: vec![rows, cols],
                data: WireArrayData::RawView(logprobs.into()),
            },
            token_ranks: WireNdArray {
                dtype: "<i8".to_string(),
                shape: vec![rows],
                data: WireArrayData::RawView(token_ranks.into()),
            },
            cu_num_generated_tokens: None,
        })
    }

    /// Resolve wire-format logprobs into semantic [`Logprobs`] by decoding the
    /// three arrays (via aux frames as needed) and grouping each row.
    fn resolve<Frame>(self, frames: &[Frame], field_prefix: &str) -> Result<Logprobs>
    where
        Frame: AsRef<[u8]>,
    {
        if let Some(indices) = self.cu_num_generated_tokens {
            return Err(Error::ExtValueDecode {
                message: format!(
                    "{field_prefix}.cu_num_generated_tokens: expected None for per-request \
                     engine-core logprobs payload, got {indices:?}"
                ),
            });
        }

        let token_ids = decode_array2_u32(
            self.logprob_token_ids,
            &format!("{field_prefix}.logprob_token_ids"),
            frames,
        )?;
        let logprobs =
            decode_array2_f32(self.logprobs, &format!("{field_prefix}.logprobs"), frames)?;
        let token_ranks = decode_array1_u32(
            self.token_ranks,
            &format!("{field_prefix}.token_ranks"),
            frames,
        )?;

        if token_ids.rows != logprobs.rows || token_ids.cols != logprobs.cols {
            return Err(Error::ExtValueDecode {
                message: format!(
                    "{field_prefix}: row shape mismatch between token ids ({}, {}) and logprobs ({}, {})",
                    token_ids.rows, token_ids.cols, logprobs.rows, logprobs.cols
                ),
            });
        }
        if token_ids.rows != token_ranks.len() {
            return Err(Error::ExtValueDecode {
                message: format!(
                    "{field_prefix}: token_ranks length {} does not match row count {}",
                    token_ranks.len(),
                    token_ids.rows
                ),
            });
        }

        // Empty position lists may be encoded as either [0, 0] or [0, k + 1].
        if token_ids.rows == 0 {
            return Ok(Logprobs {
                positions: Vec::new(),
            });
        }
        if token_ids.cols == 0 {
            return Err(Error::ExtValueDecode {
                message: format!(
                    "{field_prefix}: zero-column logprobs payload with {} rows",
                    token_ids.rows
                ),
            });
        }

        let mut positions = Vec::with_capacity(token_ids.rows);
        for ((token_ids_row, logprobs_row), sampled_rank) in token_ids
            .data
            .chunks(token_ids.cols)
            .zip(logprobs.data.chunks(logprobs.cols))
            .zip(token_ranks)
        {
            positions.push(PositionLogprobs::from_decoded_row(
                token_ids_row,
                logprobs_row,
                sampled_rank,
            )?);
        }

        Ok(Logprobs { positions })
    }
}

/// Output field wrapper deserialized from the Python wire shape, then resolved
/// into [`Logprobs`] before the decoded message is returned to callers.
#[derive(Clone, PartialEq, Debug)]
pub enum MaybeWireLogprobs {
    /// Still in wire format; needs [`MaybeWireLogprobs::resolve`]. Internal use
    /// only during deserialization.
    Wire(Box<WireLogprobs>),
    /// The decoded logprobs value.
    Direct(Logprobs),
}

impl MaybeWireLogprobs {
    /// The decoded logprobs, if already resolved.
    pub fn as_direct(&self) -> Option<&Logprobs> {
        match self {
            Self::Direct(value) => Some(value),
            Self::Wire(_) => None,
        }
    }

    /// Resolve the wire representation into decoded logprobs, looking up aux
    /// frames and decoding raw views as needed.
    pub(crate) fn resolve<Frame>(self, frames: &[Frame], field_prefix: &str) -> Result<Self>
    where
        Frame: AsRef<[u8]>,
    {
        match self {
            Self::Direct(value) => Ok(Self::Direct(value)),
            Self::Wire(value) => value.resolve(frames, field_prefix).map(Self::Direct),
        }
    }
}

impl<'de> Deserialize<'de> for MaybeWireLogprobs {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // On the wire it is always the wire form.
        WireLogprobs::deserialize(deserializer).map(|v| Self::Wire(Box::new(v)))
    }
}

impl Serialize for MaybeWireLogprobs {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Test-only: we never actually serialize into aux frames on the send path.
        match self {
            Self::Wire(value) => value.serialize(serializer),
            Self::Direct(value) => WireLogprobs::from_direct(value)
                .map_err(serde::ser::Error::custom)?
                .serialize(serializer),
        }
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;

    fn direct(positions: Vec<Vec<(u32, f32, u32)>>) -> Logprobs {
        Logprobs {
            positions: positions
                .into_iter()
                .map(|entries| PositionLogprobs {
                    entries: entries
                        .into_iter()
                        .map(|(token_id, logprob, rank)| TokenLogprob {
                            token_id,
                            logprob,
                            rank,
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    #[test]
    fn logprobs_roundtrip_from_direct_through_wire_resolve() {
        // Two positions, top-2 each: entry 0 is the sampled token with its true
        // vocab rank; entry 1 is a top-k alternative (synthetic rank 1).
        let original = direct(vec![
            vec![(10, -0.1, 5), (11, -1.2, 1)],
            vec![(20, -0.3, 2), (21, -2.5, 1)],
        ]);
        let wire = WireLogprobs::from_direct(&original).unwrap();
        // Inline raw views, so no aux frames are needed to resolve.
        let resolved = wire.resolve(&[Bytes::new()], "lp").unwrap();
        assert_eq!(resolved, original);
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved.positions[0].entries[0].rank, 5);
        assert_eq!(resolved.positions[0].entries[1].rank, 1);
    }

    #[test]
    fn resolve_rejects_nonnull_cu_num_generated_tokens() {
        let mut wire = WireLogprobs::from_direct(&direct(vec![vec![(1, -0.5, 1)]])).unwrap();
        wire.cu_num_generated_tokens = Some(vec![1]);
        assert!(wire.resolve(&[Bytes::new()], "lp").is_err());
    }

    #[test]
    fn resolve_rejects_zero_sampled_rank() {
        let wire = WireLogprobs::from_direct(&direct(vec![vec![(1, -0.5, 0)]])).unwrap();
        assert!(wire.resolve(&[Bytes::new()], "lp").is_err());
    }

    #[test]
    fn maybe_wire_logprobs_deserializes_as_wire_then_resolves() {
        let bytes = rmp_serde::to_vec_named(
            &WireLogprobs::from_direct(&direct(vec![vec![(7, -0.2, 3)]])).unwrap(),
        )
        .unwrap();
        let maybe: MaybeWireLogprobs = rmp_serde::from_slice(&bytes).unwrap();
        assert!(maybe.as_direct().is_none()); // still wire
        let resolved = maybe.resolve(&[Bytes::new()], "lp").unwrap();
        assert_eq!(
            resolved.as_direct().unwrap().positions[0].entries[0].token_id,
            7
        );
    }
}
