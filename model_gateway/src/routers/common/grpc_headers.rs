//! Convert `tonic::metadata::MetadataMap` to `axum::http::HeaderMap`.
//!
//! gRPC clients map metadata to HTTP/2 headers; this utility makes them
//! consumable by `BeforeModelCtx::headers` (which expects an axum HeaderMap).
//!
//! Binary metadata (keys ending in `-bin`) is skipped since `HeaderValue`
//! cannot represent arbitrary bytes safely.

use std::str::FromStr;

use axum::http::{HeaderMap, HeaderName, HeaderValue};
use tonic::metadata::{KeyAndValueRef, MetadataMap};

pub fn tonic_metadata_to_headermap(metadata: &MetadataMap) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for entry in metadata.iter() {
        if let KeyAndValueRef::Ascii(key, val) = entry {
            let name = HeaderName::from_str(key.as_str());
            let value = val
                .to_str()
                .ok()
                .and_then(|s| HeaderValue::from_str(s).ok());
            if let (Ok(name), Some(value)) = (name, value) {
                headers.append(name, value);
            }
        }
    }
    headers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_metadata_yields_empty_headers() {
        let md = MetadataMap::new();
        let h = tonic_metadata_to_headermap(&md);
        assert!(h.is_empty());
    }

    #[test]
    fn ascii_metadata_round_trips() {
        let mut md = MetadataMap::new();
        md.insert("x-conversation-memory-config", "test".parse().unwrap());
        md.insert("x-test-key", "value1".parse().unwrap());

        let h = tonic_metadata_to_headermap(&md);
        assert_eq!(
            h.get("x-conversation-memory-config")
                .and_then(|v| v.to_str().ok()),
            Some("test")
        );
        assert_eq!(h.get("x-test-key").and_then(|v| v.to_str().ok()), Some("value1"));
    }

    #[test]
    fn binary_metadata_is_skipped() {
        let mut md = MetadataMap::new();
        md.insert_bin(
            "x-binary-bin",
            tonic::metadata::MetadataValue::from_bytes(&[0xFF, 0x00]),
        );
        let h = tonic_metadata_to_headermap(&md);
        assert!(h.get("x-binary-bin").is_none());
    }
}
