// Copyright (c) 2023, Oracle and/or its affiliates.
// Licensed under the Universal Permissive License (UPL), Version 1.0.
// Source: https://github.com/oracle/oci-rust-sdk
// Origin commit: 0590d5dcebabc68d9115520e2be5e42f9dbf1ffb
// Copy provenance: copied verbatim from
//   oci-rust-sdk/crates/common/src/session_key_supplier.rs.

use openssl::bn::BigNum;
use openssl::{pkey::Private, rsa::Rsa};
use std::error::Error;
use std::sync::{Arc, RwLock};

static DEFAULT_KEY_SIZE: u32 = 2048;
static DEFAULT_EXPONENT: u32 = 65537;

/// Struct defining a SessionKeySupplier to contain a RSA key pair
#[derive(Debug, Clone)]
pub struct SessionKeySupplier {
    key_size: u32,
    rsa: Arc<RwLock<Rsa<Private>>>,
}

impl SessionKeySupplier {
    /// Creates a new SessionKeySupplier
    /// # Returns
    ///
    /// A new SessionKeySupplier
    ///
    pub fn new() -> Result<SessionKeySupplier, Box<dyn Error>> {
        return SessionKeySupplier::new_with_key_size(DEFAULT_KEY_SIZE);
    }

    /// Creates a new SessionKeySupplier with give key size
    /// # Arguments
    ///
    /// * `key_size`: key size of the Private key pair to use for generation
    ///
    /// # Returns
    ///
    /// A new SessionKeySupplier
    ///
    pub fn new_with_key_size(key_size: u32) -> Result<SessionKeySupplier, Box<dyn Error>> {
        let rsa = Rsa::generate_with_e(
            key_size,
            &BigNum::from_u32(DEFAULT_EXPONENT).expect("Error generating BigNumRef"),
        )?;
        Ok(SessionKeySupplier {
            key_size,
            rsa: Arc::new(RwLock::new(rsa)),
        })
    }

    /// Refresh the SessionKeySupplier
    pub async fn refresh(&self) {
        tracing::debug!("Refreshing Session key supplier");
        let mut _rsa = self.rsa.write().unwrap();
        *_rsa = Rsa::generate_with_e(
            self.key_size,
            &BigNum::from_u32(DEFAULT_EXPONENT).expect("Error generating BigNumRef"),
        )
        .unwrap();
        tracing::debug!("Refreshing Session key supplier completed");
    }

    /// Gets the Private key
    ///
    /// # Returns
    ///
    /// The private key associated with the SessionKeySupplier
    ///
    pub fn get_key(&self) -> Result<Rsa<Private>, Box<dyn Error>> {
        let rsa = self
            .rsa
            .read()
            .expect("Error getting RSA Key from SessionKeySupplier");
        return Ok(rsa.clone());
    }

    /// Gets the Public and Private key pair
    ///
    /// # Returns
    ///
    /// The Public and Private key pair associated with the SessionKeySupplier
    ///
    pub fn get_key_pair(&self) -> Result<(String, String), Box<dyn Error>> {
        let rsa = self.get_key()?;
        let session_public_key = String::from_utf8(rsa.public_key_to_pem()?)?;
        let session_private_key = String::from_utf8(rsa.private_key_to_pem()?)?;
        return Ok((session_public_key, session_private_key));
    }
}
