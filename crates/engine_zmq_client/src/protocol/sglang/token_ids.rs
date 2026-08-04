// SGLang encodes token-id sequences as Python `array.array('q', ...)`, which
// msgspec serializes as the 2-tuple `[typecode, raw_bytes]` — the typecode
// string `"q"` (signed 64-bit) followed by a msgpack `bin` of little-endian
// int64s, not a plain msgpack integer list. This module carries that wire form
// as a newtype over the engine-neutral `Vec<u32>` token ids SMG uses.

use rmpv::Value;
use serde::{
    de::Error as _, ser::SerializeTuple, Deserialize, Deserializer, Serialize, Serializer,
};

/// The msgspec typecode string for a Python `array.array('q', ...)`: a signed
/// 64-bit little-endian integer array. Token ids ride the wire under this code.
const TYPECODE_INT64: &str = "q";

/// Number of bytes per `int64` element in the raw-bytes half of the tuple.
const INT64_BYTES: usize = 8;

/// A token-id sequence in SGLang's `array.array('q', ...)` wire form.
///
/// On the wire it is the 2-element msgpack array `["q", <int64-LE bytes>]`.
/// In memory it is the engine-neutral `Vec<u32>` token ids: SMG tokenizes
/// upstream and the engine speaks token ids on the skip-tokenizer path.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenIdArray(pub Vec<u32>);

/// Serialize `&[u8]` as a msgpack `bin` (serde's default treats it as a `u8`
/// sequence, which would emit an integer array instead of the raw buffer).
struct RawBytes<'a>(&'a [u8]);

impl Serialize for RawBytes<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(self.0)
    }
}

impl Serialize for TokenIdArray {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut raw = Vec::with_capacity(self.0.len() * INT64_BYTES);
        for &id in &self.0 {
            raw.extend_from_slice(&i64::from(id).to_le_bytes());
        }
        let mut tuple = serializer.serialize_tuple(2)?;
        tuple.serialize_element(TYPECODE_INT64)?;
        tuple.serialize_element(&RawBytes(&raw))?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for TokenIdArray {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Decode through a dynamic value: the second element is a msgpack `bin`,
        // which serde cannot map onto a typed field without `serde_bytes`.
        let value = Value::deserialize(deserializer)?;
        let Value::Array(items) = value else {
            return Err(D::Error::custom(format!(
                "expected a 2-element [typecode, bytes] token-id array, got {value:?}"
            )));
        };
        let [code, bytes] = items.as_slice() else {
            return Err(D::Error::custom(format!(
                "expected exactly 2 elements in a token-id array, got {}",
                items.len()
            )));
        };
        let code = code.as_str().ok_or_else(|| {
            D::Error::custom(format!("token-id array typecode is not a string: {code:?}"))
        })?;
        if code != TYPECODE_INT64 {
            return Err(D::Error::custom(format!(
                "unsupported token-id array typecode `{code}`, expected `{TYPECODE_INT64}`"
            )));
        }
        let Value::Binary(raw) = bytes else {
            return Err(D::Error::custom(format!(
                "token-id array payload is not a msgpack bin: {bytes:?}"
            )));
        };
        if raw.len() % INT64_BYTES != 0 {
            return Err(D::Error::custom(format!(
                "token-id array byte length {} is not a multiple of {INT64_BYTES}",
                raw.len()
            )));
        }
        raw.chunks_exact(INT64_BYTES)
            .map(|chunk| {
                let mut buf = [0u8; INT64_BYTES];
                buf.copy_from_slice(chunk);
                let id = i64::from_le_bytes(buf);
                u32::try_from(id)
                    .map_err(|_| D::Error::custom(format!("token id {id} is out of range for u32")))
            })
            .collect::<Result<Vec<u32>, D::Error>>()
            .map(TokenIdArray)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{decode_msgpack, decode_value, encode_msgpack};

    #[test]
    fn token_id_array_encodes_as_typed_bin_tuple() {
        // The golden wire bytes for [9906, 11, 1917] captured from the Python
        // msgspec encoder: 2-array ["q", bin24 of three little-endian int64s].
        let encoded = encode_msgpack(&TokenIdArray(vec![9906, 11, 1917])).unwrap();
        let expected = "92a171c418b2260000000000000b000000000000007d07000000000000";
        assert_eq!(hex(&encoded), expected);

        let Value::Array(items) = decode_value(&encoded).unwrap() else {
            panic!("expected a 2-element array");
        };
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], Value::from("q"));
    }

    #[test]
    fn token_id_array_roundtrips() {
        for ids in [
            vec![],
            vec![0],
            vec![1, 2, 3],
            vec![9906, 11, 1917, u32::MAX],
        ] {
            let encoded = encode_msgpack(&TokenIdArray(ids.clone())).unwrap();
            assert_eq!(
                decode_msgpack::<TokenIdArray>(&encoded).unwrap(),
                TokenIdArray(ids)
            );
        }
    }

    #[test]
    fn decode_rejects_wrong_typecode() {
        // A float32 array ("f") is not a valid token-id sequence.
        let value = Value::Array(vec![Value::from("f"), Value::Binary(vec![0; 8])]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &value).unwrap();
        let error = decode_msgpack::<TokenIdArray>(&bytes).unwrap_err();
        assert!(error
            .to_string()
            .contains("unsupported token-id array typecode"));
    }

    #[test]
    fn decode_rejects_ragged_byte_length() {
        // 7 bytes is not a whole number of int64 elements.
        let value = Value::Array(vec![Value::from("q"), Value::Binary(vec![0; 7])]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &value).unwrap();
        let error = decode_msgpack::<TokenIdArray>(&bytes).unwrap_err();
        assert!(error.to_string().contains("not a multiple of"));
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}
