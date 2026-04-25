// Copyright (c) 2023, Oracle and/or its affiliates.
// Licensed under the Universal Permissive License (UPL), Version 1.0.
// Source: https://github.com/oracle/oci-rust-sdk
// Origin commit: 0590d5dcebabc68d9115520e2be5e42f9dbf1ffb
// Copy provenance: copied verbatim from
//   oci-rust-sdk/crates/common/src/private_key_supplier.rs.

use openssl::pkey::Private;
use openssl::rsa::Rsa;
use std::error::Error;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use crate::oci::file_utils::expand_user_home;

pub trait Supplier: Send + Sync + Debug + SupplierClone {
    fn get_key(&self) -> Result<Rsa<Private>, Box<dyn std::error::Error>>;
    fn refresh(&self) -> Result<(), Box<dyn std::error::Error>>;
}

pub trait SupplierClone {
    fn clone_box(&self) -> Box<dyn Supplier>;
}

impl<T> SupplierClone for T
where
    T: 'static + Supplier + Clone,
{
    fn clone_box(&self) -> Box<dyn Supplier> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Supplier> {
    fn clone(&self) -> Box<dyn Supplier> {
        self.clone_box()
    }
}

#[derive(Debug, Clone)]
pub struct PrivateKeySupplier {
    private_key: Arc<RwLock<Rsa<Private>>>,
}

impl PrivateKeySupplier {
    pub fn new(key_content: String) -> Result<Self, Box<dyn Error>> {
        let rsa_private_key = Rsa::private_key_from_pem(key_content.as_bytes())?;
        Ok(PrivateKeySupplier {
            private_key: Arc::new(RwLock::new(rsa_private_key)),
        })
    }

    pub fn new_with_passphrase(
        key_content: String,
        passphrase: Option<Vec<char>>,
    ) -> Result<Self, Box<dyn Error>> {
        match passphrase.as_ref() {
            Some(pass) => {
                let pass_bytes = pass.iter().map(|c| *c as u8).collect::<Vec<_>>();
                let rsa_private_key =
                    Rsa::private_key_from_pem_passphrase(key_content.as_bytes(), &pass_bytes)?;
                Ok(PrivateKeySupplier {
                    private_key: Arc::new(RwLock::new(rsa_private_key)),
                })
            }
            None => Err("Missing Passphrase for private key".into()),
        }
    }
}

impl Supplier for PrivateKeySupplier {
    fn get_key(&self) -> Result<Rsa<Private>, Box<dyn Error>> {
        Ok(self.private_key.read().unwrap().clone())
    }

    fn refresh(&self) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FilePrivateKeySupplier {
    private_key: Arc<RwLock<Rsa<Private>>>,
    key_path: String,
    passphrase: Option<Vec<char>>,
}

impl FilePrivateKeySupplier {
    pub fn new(key_path: String) -> Result<Self, Box<dyn Error>> {
        let key_content = std::fs::read_to_string(&expand_user_home(&key_path))
            .expect("Unable to read the Private key file");
        let rsa_private_key = Rsa::private_key_from_pem(key_content.as_bytes())?;
        Ok(FilePrivateKeySupplier {
            private_key: Arc::new(RwLock::new(rsa_private_key)),
            key_path,
            passphrase: None,
        })
    }

    pub fn new_with_passphrase(
        key_path: String,
        passphrase: Option<Vec<char>>,
    ) -> Result<Self, Box<dyn Error>> {
        match passphrase.as_ref() {
            Some(pass) => {
                let key_content =
                    std::fs::read_to_string(&key_path).expect("Unable to read Private key file");
                let pass_bytes = pass.iter().map(|c| *c as u8).collect::<Vec<_>>();
                let rsa_private_key =
                    Rsa::private_key_from_pem_passphrase(key_content.as_bytes(), &pass_bytes)?;
                Ok(FilePrivateKeySupplier {
                    private_key: Arc::new(RwLock::new(rsa_private_key)),
                    key_path,
                    passphrase: None,
                })
            }
            None => Err("Missing Passphrase for private key".into()),
        }
    }
}

impl Supplier for FilePrivateKeySupplier {
    fn get_key(&self) -> Result<Rsa<Private>, Box<dyn std::error::Error>> {
        Ok(self.private_key.read().unwrap().clone())
    }

    fn refresh(&self) -> Result<(), Box<dyn Error>> {
        // Get write lock for private key
        let mut _pk = self.private_key.write().unwrap();
        match self.passphrase.as_ref() {
            Some(pass) => {
                let key_content = std::fs::read_to_string(&self.key_path)
                    .expect("Unable to read Private key file");
                let pass_bytes = pass.iter().map(|c| *c as u8).collect::<Vec<_>>();
                let rsa_private_key =
                    Rsa::private_key_from_pem_passphrase(key_content.as_bytes(), &pass_bytes)?;
                *_pk = rsa_private_key;
            }
            None => {
                let key_content = std::fs::read_to_string(&expand_user_home(&self.key_path))
                    .expect("Unable to read the Private key file");
                let rsa_private_key = Rsa::private_key_from_pem(key_content.as_bytes())?;
                *_pk = rsa_private_key
            }
        }
        Ok(())
    }
}
