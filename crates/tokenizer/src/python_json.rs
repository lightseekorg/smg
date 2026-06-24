//! Serialize a [`serde_json::Value`] exactly like Python's
//! `json.dumps(value, ensure_ascii=False)`: default separators `", "` / `": "`
//! and raw UTF-8 (no `\uXXXX` escaping).
//!
//! The DeepSeek V3.2/V4 prompt encoders embed tool schemas and argument values
//! as JSON in the prompt. vLLM's reference encoder uses `json.dumps`, so plain
//! `serde_json::to_string` (compact, no spaces) produces a byte-different prompt
//! and shifts the model off its training distribution. Use this instead.

use std::io;

use serde::Serialize;
use serde_json::{
    ser::{Formatter, Serializer},
    Value,
};

/// `serde_json` formatter matching Python's default `json.dumps` separators.
struct PythonDefaultFormatter;

impl Formatter for PythonDefaultFormatter {
    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(b": ")
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
}

/// Serialize `value` like `json.dumps(value, ensure_ascii=False)`.
pub(crate) fn to_python_json_string(value: &Value) -> String {
    let mut buf = Vec::new();
    let mut ser = Serializer::with_formatter(&mut buf, PythonDefaultFormatter);
    if value.serialize(&mut ser).is_err() {
        return "null".to_string();
    }
    String::from_utf8(buf).unwrap_or_else(|_| "null".to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn matches_python_json_dumps_separators() {
        let v = json!({"name": "get_weather", "args": [1, 2, {"a": true}]});
        // Python: json.dumps(v, ensure_ascii=False)
        assert_eq!(
            to_python_json_string(&v),
            r#"{"name": "get_weather", "args": [1, 2, {"a": true}]}"#
        );
    }

    #[test]
    fn keeps_unicode_raw() {
        let v = json!({"city": "广州"});
        assert_eq!(to_python_json_string(&v), r#"{"city": "广州"}"#);
    }

    #[test]
    fn empty_containers() {
        assert_eq!(to_python_json_string(&json!({})), "{}");
        assert_eq!(to_python_json_string(&json!([])), "[]");
    }
}
