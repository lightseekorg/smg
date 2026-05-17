// Copyright (c) 2023, Oracle and/or its affiliates.
// Licensed under the Universal Permissive License (UPL), Version 1.0.
// Source: https://github.com/oracle/oci-rust-sdk
// Origin commit: 0590d5dcebabc68d9115520e2be5e42f9dbf1ffb
// Copy provenance: copied verbatim from
//   oci-rust-sdk/crates/common/src/auth_utils.rs.
// Note: this file is NOT in the design-doc §5 13-file list, but is required as
// a transitive compile dependency of jwt_claim_set.rs and x509_federation_client.rs.

use base64ct::{Base64Unpadded, Encoding};
use std::error::Error;

/// Decodes a base64 encoded string
///
/// # Arguments
///
/// * `base64_str` : The string to be decoded
///
/// # Returns
///
/// The decoded string in form a Vec<U8>
///
pub fn base64_decode(base64_str: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    return Ok(Base64Unpadded::decode_vec(base64_str)?);
}

/// Sanitize a cert or key file for OCI Requests
///
/// # Arguments
///
/// * `cert_string` : The string to be sanitized for request
///
/// # Returns
///
/// The sanitized string
///
pub fn sanitize_certificate_string(cert_string: String) -> String {
    return cert_string
        .replace("-----BEGIN CERTIFICATE-----", "")
        .replace("-----END CERTIFICATE-----", "")
        .replace("-----BEGIN PUBLIC KEY-----", "")
        .replace("-----END PUBLIC KEY-----", "")
        .replace("\n", "");
}
