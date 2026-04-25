// Copyright (c) 2023, Oracle and/or its affiliates.
// Licensed under the Universal Permissive License (UPL), Version 1.0.
// Source: https://github.com/oracle/oci-rust-sdk
// Origin commit: 0590d5dcebabc68d9115520e2be5e42f9dbf1ffb
// Copy provenance: copied verbatim from
//   oci-rust-sdk/crates/common/src/federation_client.rs.

use async_trait::async_trait;
use openssl::pkey::Private;
use openssl::rsa::Rsa;
use std::error::Error;
use std::fmt::Debug;

/// Trait defining an FederationClient
#[async_trait]
pub trait FederationClient: Send + Sync + Debug + FederationClientClone {
    /// Get Security token from the FederationClient
    fn get_security_token(&self) -> Result<String, Box<dyn Error>>;
    /// Refresh and get Security token from the FederationClient
    async fn refresh_and_get_security_token(&self) -> Result<String, Box<dyn Error>>;
    /// Get tenancy id from FederationClient
    fn get_tenancy_id(&self) -> Result<String, Box<dyn Error>>;
    /// Get fingerprint from FederationClient
    fn get_fingerprint(&self) -> Result<String, Box<dyn Error>>;
    /// Get region from FederationClient
    fn get_region(&self) -> Result<Option<String>, Box<dyn Error>>;
    /// Get private key from FederationClient
    fn get_private_key(&self) -> Result<Rsa<Private>, Box<dyn Error>>;
    /// Refresh FederationClient
    async fn refresh(&self) -> Result<(), Box<dyn Error>>;
}

// This allows users of this library to clone a Box<dyn FederationClient>
pub trait FederationClientClone {
    fn clone_box(&self) -> Box<dyn FederationClient>;
}

impl<T> FederationClientClone for T
where
    T: 'static + FederationClient + Clone,
{
    fn clone_box(&self) -> Box<dyn FederationClient> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn FederationClient> {
    fn clone(&self) -> Box<dyn FederationClient> {
        self.clone_box()
    }
}
