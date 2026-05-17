// Copyright (c) 2023, Oracle and/or its affiliates.
// Licensed under the Universal Permissive License (UPL), Version 1.0.
// Source: https://github.com/oracle/oci-rust-sdk
// Origin commit: 0590d5dcebabc68d9115520e2be5e42f9dbf1ffb
// Copy provenance: copied verbatim from
//   oci-rust-sdk/crates/common/src/security_token_container.rs.

use crate::oci::jwt_claim_set::JwtClaimsSet;
use chrono::prelude::{DateTime, Utc};
use std::error::Error;

/// Struct defining a SecurityTokenContainer to contain a security token and its claim set
#[derive(Debug, Clone)]
pub struct SecurityTokenContainer {
    security_token: String,
    jwt: JwtClaimsSet,
}

impl SecurityTokenContainer {
    /// Get claim value with given name from the X509 token
    /// # Arguments
    ///
    /// * `get_claim`: The name of the claim to query
    ///
    /// # Returns
    ///
    /// The value of claim if found else raise an Error
    pub fn get_claim(&self, claim_name: &str) -> Result<serde_json::Value, Box<dyn Error>> {
        return self.jwt.get_claim(claim_name);
    }

    /// Get expiration time of the X509 token
    ///
    /// # Returns
    ///
    /// The expiration time of the X509 token as DateTime<Utc>
    pub fn get_expiry_time(&self) -> Result<DateTime<Utc>, Box<dyn Error>> {
        return self.jwt.expiration_time();
    }

    /// Check if X509 token is valid or not
    ///
    /// # Returns
    ///
    /// The true if x509 is valid, false otherwise
    pub fn is_valid(&self) -> Result<bool, Box<dyn Error>> {
        return self.jwt.is_valid();
    }

    /// Get the security token associated with this container
    ///
    /// # Returns
    ///
    /// The x509 security token
    pub fn session_token(&self) -> String {
        String::from(&self.security_token)
    }

    /// Creates a new Security token container
    /// # Arguments
    ///
    /// * `token`: The x509 token to use for the container
    ///
    /// # Returns
    ///
    /// A new Security Token Container
    ///
    pub fn new(token: String) -> Result<SecurityTokenContainer, Box<(dyn Error + 'static)>> {
        Ok(SecurityTokenContainer {
            security_token: String::from(&token),
            jwt: JwtClaimsSet::new(String::from(&token))?,
        })
    }
}
