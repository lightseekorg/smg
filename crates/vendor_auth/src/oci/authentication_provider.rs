// Copyright (c) 2023, Oracle and/or its affiliates.
// Licensed under the Universal Permissive License (UPL), Version 1.0.
// Source: https://github.com/oracle/oci-rust-sdk
// Origin commit: 0590d5dcebabc68d9115520e2be5e42f9dbf1ffb
// Copy provenance: copied verbatim from
//   oci-rust-sdk/crates/common/src/authentication_provider.rs.

use openssl::pkey::Private;
use openssl::rsa::Rsa;
use std::error::Error;
use std::fmt::Debug;

/// Trait defining an Authentication Provider
pub trait AuthenticationProvider: Send + Sync + Debug + AuthenticationProviderClone {
    /// Returns the Tenancy OCID associated with this AuthenticationProvider
    fn tenancy_id(&self) -> Result<String, Box<dyn Error>>;
    /// Returns the User OCID associated with this AuthenticationProvider
    fn user_id(&self) -> Result<String, Box<dyn Error>>;
    /// Returns the Fingerprint associated with the Private Key of this AuthenticationProvider
    fn fingerprint(&self) -> Result<String, Box<dyn Error>>;
    /// Returns the Private Key associated with this AuthenticationProvider wrapped in a Result
    fn private_key(&self) -> Result<Rsa<Private>, Box<dyn Error>>;
    /// Returns the key id associated with this AuthenticationProvider to be used for signing requests
    fn key_id(&self) -> Result<String, Box<dyn Error>> {
        let key_id = format!(
            "{}/{}/{}",
            self.tenancy_id()?,
            self.user_id()?,
            self.fingerprint()?
        );
        Ok(key_id)
    }
    /// Returns the region-id associated with this AuthenticationProvider
    fn region_id(&self) -> Result<Option<String>, Box<dyn Error>>;
}

// This allows users of this library to clone a Box<dyn AuthenticationProvider>
pub trait AuthenticationProviderClone {
    fn clone_box(&self) -> Box<dyn AuthenticationProvider>;
}

impl<T> AuthenticationProviderClone for T
where
    T: 'static + AuthenticationProvider + Clone,
{
    fn clone_box(&self) -> Box<dyn AuthenticationProvider> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn AuthenticationProvider> {
    fn clone(&self) -> Box<dyn AuthenticationProvider> {
        self.clone_box()
    }
}
