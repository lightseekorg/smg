// Copyright (c) 2023, Oracle and/or its affiliates.
// Licensed under the Universal Permissive License (UPL), Version 1.0.
// Source: https://github.com/oracle/oci-rust-sdk
// Origin commit: 0590d5dcebabc68d9115520e2be5e42f9dbf1ffb
// Copy provenance: copied verbatim from
//   oci-rust-sdk/crates/common/src/jwt_claim_set.rs.

use chrono::prelude::*;
use serde_json::Value;
use std::collections::HashMap;

use crate::oci::auth_utils::base64_decode;

/// Struct defining a JwtClaimsSet to keep information about a X509 token
#[derive(Debug, Clone)]
pub struct JwtClaimsSet {
    headers: HashMap<String, Value>,
    payload: HashMap<String, Value>,
    signature: String,
}

impl JwtClaimsSet {
    /// Creates a new JwtClaimsSet
    /// # Arguments
    ///
    /// * `token`: The X509 token to use
    ///
    /// # Returns
    ///
    /// A struct of type JwtClaimsSet or raise an error
    pub fn new(token: String) -> Result<Self, Box<dyn std::error::Error>> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err("Invalid JWT format".into());
        }

        let header_bytes = base64_decode(parts[0])?;
        let header_str = String::from_utf8(header_bytes)?;
        let headers: HashMap<String, Value> = serde_json::from_str(&header_str)?;

        let payload = base64_decode(parts[1])?;
        let payload_str = String::from_utf8(payload)?;
        let payload: HashMap<String, Value> = serde_json::from_str(&payload_str)?;

        let signature = parts[2].to_string();
        if signature.is_empty() {
            return Err("The token doesn't have a signature".into());
        }
        Ok(JwtClaimsSet {
            headers,
            payload,
            signature,
        })
    }

    /// Get signature of the X509 token
    ///
    /// # Returns
    ///
    /// The signature of the X509 token as String
    pub fn signature(&self) -> String {
        return String::from(&self.signature);
    }

    /// Get headers of the X509 token
    ///
    /// # Returns
    ///
    /// The headers of the X509 token as Hasmap of String to Serde Values
    pub fn headers(&self) -> &HashMap<String, Value> {
        return &self.headers;
    }

    /// Get expiration time of the X509 token
    ///
    /// # Returns
    ///
    /// The expiration time of the X509 token as DateTime<Utc>
    pub fn expiration_time(&self) -> Result<DateTime<Utc>, Box<dyn std::error::Error>> {
        let exp_claim = self.get_claim("exp")?;
        if let Some(timestamp) = exp_claim.as_i64() {
            let datetime: DateTime<Utc> = Utc.timestamp_opt(timestamp, 0).single().unwrap();
            return Ok(datetime);
        } else {
            return Err("Session Token is invalid, no exp claim found".into());
        }
    }

    /// Get issue time of the X509 token
    ///
    /// # Returns
    ///
    /// The issue time of the X509 token as DateTime<Utc>
    pub fn issue_time(&self) -> Result<DateTime<Utc>, Box<dyn std::error::Error>> {
        let iat_claim = self.get_claim("iat")?;
        if let Some(timestamp) = iat_claim.as_i64() {
            let datetime: DateTime<Utc> = Utc.timestamp_opt(timestamp, 0).single().unwrap();
            return Ok(datetime);
        } else {
            return Err("Session Token is invalid, no iat claim found".into());
        }
    }

    /// Get claim value with given name from the X509 token
    /// # Arguments
    ///
    /// * `name`: The name of the claim to query
    ///
    /// # Returns
    ///
    /// The value of claim if found else raise an Error
    pub fn get_claim(&self, name: &str) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        match self.payload.get(name) {
            Some(value) => Ok(value.clone()),
            None => {
                return Err(format!("{} not presesnt in claim", name).into());
            }
        }
    }

    /// Check if X509 token is valid or not
    ///
    /// # Returns
    ///
    /// The true if x509 is valid, false otherwise
    pub fn is_valid(&self) -> Result<bool, Box<dyn std::error::Error>> {
        return Ok(self.expiration_time()? > Utc::now());
    }
}
