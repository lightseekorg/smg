// SPDX-License-Identifier: Apache-2.0
//
// Adapted from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/mod.rs.

//! Engine-agnostic wire primitives: msgpack (de)serialization, numpy dtype
//! handling, and the zero-copy tensor aux-frame codec.

use std::{any::type_name, io::Cursor};

use rmpv::Value;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

pub mod dtype;
pub mod tensor;

/// Dynamic msgpack value used for schema positions that are preserved but not
/// yet strongly typed.
pub type OpaqueValue = Value;

/// Encode a Rust value into msgpack using named fields (map encoding for
/// structs; positional arrays come from `serde_tuple`-derived types).
pub fn encode_msgpack<T>(value: &T) -> Result<Vec<u8>>
where
    T: Serialize + std::fmt::Debug,
{
    rmp_serde::to_vec_named(value).map_err(|error| Error::Encode {
        target_type: type_name::<T>(),
        message: format!("failed to encode value `{value:?}`: {error}"),
    })
}

/// Decode a msgpack payload into a strongly typed value, attaching a dynamic
/// value preview to decode failures for diagnostics.
pub fn decode_msgpack<T>(bytes: &[u8]) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    fn decode_value_preview(bytes: &[u8]) -> String {
        match decode_value(bytes) {
            Ok(value) => format!("{value}"),
            Err(error) => format!("<value decode failed: {error}>"),
        }
    }

    rmp_serde::from_slice(bytes).map_err(|error| Error::Decode {
        target_type: type_name::<T>(),
        message: format!("{error}; value fallback: {}", decode_value_preview(bytes)),
    })
}

/// Decode a msgpack payload into a dynamic value for diagnostics and tests.
pub fn decode_value(bytes: &[u8]) -> Result<Value> {
    Ok(rmpv::decode::read_value(&mut Cursor::new(bytes))?)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn decode_msgpack_includes_type_name_and_value_fallback() {
        let payload = rmp_serde::to_vec_named(&BTreeMap::from([("status", "READY")])).unwrap();
        let error = decode_msgpack::<u64>(&payload).unwrap_err();
        let rendered = error.to_string();
        assert!(rendered.contains("u64"), "missing type name: {rendered}");
        assert!(
            rendered.contains("READY"),
            "missing value fallback: {rendered}"
        );
    }

    #[test]
    fn roundtrip_named_map_struct() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Cfg {
            status: String,
            headless: bool,
        }
        let cfg = Cfg {
            status: "HELLO".into(),
            headless: true,
        };
        let bytes = encode_msgpack(&cfg).unwrap();
        assert_eq!(decode_msgpack::<Cfg>(&bytes).unwrap(), cfg);
    }
}
