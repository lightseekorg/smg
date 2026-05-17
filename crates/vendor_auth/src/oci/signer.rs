// Copyright (c) 2023, Oracle and/or its affiliates.
// Licensed under the Universal Permissive License (UPL), Version 1.0.
// Source: https://github.com/oracle/oci-rust-sdk
// Origin commit: 0590d5dcebabc68d9115520e2be5e42f9dbf1ffb
// Copy provenance: copied verbatim from
//   oci-rust-sdk/crates/common/src/signer.rs.
// Per HeaderMap-refactor decision documented in src/oci/NOTICE.md, this file
// keeps the upstream reqwest::HeaderMap / reqwest::Method types verbatim — the
// http::Request<Bytes> ↔ reqwest conversion happens at the adapter boundary
// (crates/vendor_auth/src/signer_adapter.rs).

use std::collections::HashMap;

use crate::oci::authentication_provider::AuthenticationProvider;
use base64ct::{Base64, Encoding};
use chrono::prelude::*;
use openssl::pkey::Private;
use openssl::rsa::Rsa;
use openssl::sha::Sha256;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Method;
use url::Url;

use super::http_signature;

static DATE_HEADER: &str = "date";
static REQUEST_TARGET_HEADER: &str = "(request-target)";
static HOST_HEADER: &str = "host";
static CONTENT_TYPE_HEADER: &str = "content-type";
static CONTENT_LENGTH_HEADER: &str = "content-length";
static X_CONTENT_SHA256: &str = "x-content-sha256";
static AUTHORIZATION_HEADER: &str = "authorization";

fn get_formatted_date_time(utc: DateTime<Utc>) -> String {
    let date_unformatted_string: String = utc.to_rfc2822();
    let mut date_formatted = String::new();
    for i in 0..date_unformatted_string.len() {
        let ch = date_unformatted_string.chars().nth(i).unwrap();
        if ch == '+' {
            break;
        }
        date_formatted.push(ch);
    }
    date_formatted = date_formatted + &*utc.timezone().to_string();
    date_formatted
}

pub fn get_required_headers_from_key_and_id(
    method: Method,
    payload: &str,
    original_headers: HeaderMap,
    url_data: Url,
    private_key: Rsa<Private>,
    key_id: String,
    query_params: HashMap<String, String>,
    exclude_body: bool,
) -> HeaderMap {
    let methods_with_extra_required_headers = vec![Method::PUT, Method::POST, Method::PATCH];
    let mut headers_to_sign = vec![DATE_HEADER, REQUEST_TARGET_HEADER, HOST_HEADER];
    let mut new_headers = original_headers.clone();
    if !new_headers.contains_key(DATE_HEADER) {
        let date_str = get_formatted_date_time(Utc::now());
        new_headers.insert(DATE_HEADER, date_str.parse().unwrap());
    }

    if !new_headers.contains_key(HOST_HEADER) {
        let url_host = url_data.host().unwrap().to_string();
        new_headers.insert(HOST_HEADER, url_host.parse().unwrap());
    }
    if !exclude_body {
        for method_with_extra_header in methods_with_extra_required_headers {
            if method == method_with_extra_header {
                if !new_headers.contains_key(CONTENT_TYPE_HEADER) {
                    new_headers.insert(CONTENT_TYPE_HEADER, "application/json".parse().unwrap());
                }
                if !new_headers.contains_key(CONTENT_LENGTH_HEADER) {
                    new_headers.insert(CONTENT_LENGTH_HEADER, HeaderValue::from(payload.len()));
                }
                if !new_headers.contains_key(X_CONTENT_SHA256) {
                    let body_hash = calculate_body_sha256_hash(payload);
                    new_headers.insert(X_CONTENT_SHA256, body_hash.parse().unwrap());
                }
                headers_to_sign.push(CONTENT_TYPE_HEADER);
                headers_to_sign.push(CONTENT_LENGTH_HEADER);
                headers_to_sign.push(X_CONTENT_SHA256);
            }
        }
    }
    let mut url_with_query_params = url_data.clone();
    if !query_params.is_empty() {
        let _url = Url::parse_with_params(url_data.as_str(), query_params);
        if _url.is_ok() {
            url_with_query_params = _url.unwrap();
        }
    }
    let string_to_sign = calculate_string_to_sign(
        url_with_query_params,
        method.clone(),
        &new_headers,
        headers_to_sign.clone(),
    );
    let signature = sign(string_to_sign, private_key);
    let joined_signed_headers = headers_to_sign.join(" ");
    let authorization_header = format!(
        r###"Signature version="1",keyId="{}",algorithm="rsa-sha256",headers="{}",signature="{}""###,
        key_id, joined_signed_headers, signature
    );

    new_headers.insert(AUTHORIZATION_HEADER, authorization_header.parse().unwrap());

    new_headers
}

pub fn get_required_headers(
    method: Method,
    payload: &str,
    original_headers: HeaderMap,
    url_data: Url,
    authentication_provider: &Box<dyn AuthenticationProvider>,
    query_params: HashMap<String, String>,
    exclude_body: bool,
) -> HeaderMap {
    return get_required_headers_from_key_and_id(
        method,
        payload,
        original_headers,
        url_data,
        authentication_provider.private_key().unwrap(),
        authentication_provider.key_id().expect("Key Id missing"),
        query_params,
        exclude_body,
    );
}

fn calculate_body_sha256_hash(body: &str) -> String {
    let mut sha256 = Sha256::new();
    sha256.update(body.as_bytes());
    let body_hash = sha256.finish();
    Base64::encode_string(&body_hash)
}

fn sign(data_to_sign: String, private_key: Rsa<Private>) -> String {
    let signature = http_signature::sign(private_key, data_to_sign.as_bytes());
    Base64::encode_string(&signature)
}

fn calculate_string_to_sign(
    url: Url,
    method: Method,
    headers: &HeaderMap,
    headers_to_sign: Vec<&str>,
) -> String {
    let mut result_parts = Vec::new();
    for header in headers_to_sign {
        let header_val;
        if header == REQUEST_TARGET_HEADER {
            let mut path_string: String = url.path().to_string();
            match url.query() {
                None => {}
                Some(_) => {
                    path_string = format!("{}?{}", path_string, url.query().unwrap());
                }
            }
            header_val = format!("{} {}", method.to_string().to_lowercase(), path_string);
        } else {
            header_val = (*headers)[header].to_str().unwrap().to_string();
        }
        result_parts.push(format!("{}: {}", header, header_val));
    }
    let result_string = result_parts.join("\n");
    result_string
}
