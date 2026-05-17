// Copyright (c) 2023, Oracle and/or its affiliates.
// Licensed under the Universal Permissive License (UPL), Version 1.0.
// Source: https://github.com/oracle/oci-rust-sdk
// Origin commit: 0590d5dcebabc68d9115520e2be5e42f9dbf1ffb
// Copy provenance: copied verbatim from
//   oci-rust-sdk/crates/common/src/constants.rs.

pub const DEFAULT_CONFIG_FILE_PATH: &str = "~/.oci/config";
pub const TENANCY: &str = "tenancy";
pub const USER: &str = "user";
pub const PASS_PHRASE: &str = "pass_phrase";
pub const KEY_FILE: &str = "key_file";
pub const FINGERPRINT: &str = "fingerprint";
pub const REGION: &str = "region";
pub const SESSION_TOKEN_FILE: &str = "security_token_file";

// X509 constants
pub const METADATA_URL_BASE: &str = "http://169.254.169.254/opc/v2";
pub const LEAF_CERTIFICATE_URL_PATH: &str = "/identity/cert.pem";
pub const LEAF_CERTIFICATE_PRIVATE_KEY_URL_PATH: &str = "/identity/key.pem";
pub const REGION_URL_PATH: &str = "/instance/canonicalRegionName";
pub const REALM_URL_PATH: &str = "/instance/regionInfo/realmDomainComponent";
pub const INTERMEDIATE_CERTIFICATE_URL_PATH: &str = "/identity/intermediate.pem";
pub const EMPTY_STRING: &str = "";
pub const DEFAULT_FINGERPRINT_ALGORITHM: &str = "SHA256";
pub const DEFAULT_PURPOSE: &str = "DEFAULT";
