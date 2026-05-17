// Copyright (c) 2023, Oracle and/or its affiliates.
// Licensed under the Universal Permissive License (UPL), Version 1.0.
// Source: https://github.com/oracle/oci-rust-sdk
// Origin commit: 0590d5dcebabc68d9115520e2be5e42f9dbf1ffb
// Copy provenance: copied verbatim from
//   oci-rust-sdk/crates/common/src/certficate_retreiver.rs (file renamed to fix
//   typo per design doc D8; internal type names like `UrlBasedCertificateRetriever`
//   are kept verbatim).

use async_trait::async_trait;
use reqwest::header::HeaderMap;
use std::error::Error;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tracing::{debug, trace};

/// Trait defining an CertificateRetriever
#[async_trait]
pub trait CertificateRetriever: Send + Sync + Debug + CertificateRetrieverClone {
    /// Returns the CertificateAndPrivateKey associated with this CertificateRetriever
    fn get_certificate_and_private_key(&self) -> Result<CertificateAndPrivateKey, Box<dyn Error>>;
    /// Returns the Raw Cert as String associated with this CertificateRetriever
    fn get_certificate_raw(&self) -> Result<String, Box<dyn Error>>;
    /// Returns the Private Key associated with this CertificateRetriever
    fn get_private_key_pem(&self) -> Result<Option<String>, Box<dyn Error>>;
    /// Refreshes the CertificateRetriever
    async fn refresh(&self) -> Result<(), Box<dyn Error>>;
}

pub trait CertificateRetrieverClone {
    fn clone_box(&self) -> Box<dyn CertificateRetriever>;
}

impl<T> CertificateRetrieverClone for T
where
    T: 'static + CertificateRetriever + Clone,
{
    fn clone_box(&self) -> Box<dyn CertificateRetriever> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn CertificateRetriever> {
    fn clone(&self) -> Box<dyn CertificateRetriever> {
        self.clone_box()
    }
}

/// Struct defining a CertificateAndPrivateKey that maintains the cert and private key for the CertificateRetriever
#[derive(Debug, Clone)]
pub struct CertificateAndPrivateKey {
    pub certificate: String,
    pub private_key_pem: Option<String>,
}

/// Struct defining a UrlBasedCertificateRetriever that retrieves its cert and private key from a URL
#[derive(Debug, Clone)]
pub struct UrlBasedCertificateRetriever {
    /// Reqwest client to make calls to retrieve certs and private key
    client: Arc<reqwest::Client>,
    /// URL to use for retrieving the certificate
    certificate_url: String,
    /// Optional URL to use for retrieving the priavte key from
    private_key_url: Option<String>,
    /// Shared access CertificateAndPrivateKey struct that holds the certififcate and private key
    pub certificate_and_private_key: Arc<RwLock<CertificateAndPrivateKey>>,
}

impl UrlBasedCertificateRetriever {
    /// Creates a new UrlBasedCertificateRetriever
    ///
    /// # Arguments
    ///
    /// * `client` : Reqwest client to use
    /// * `certificate_url` : URL to use for retrieving the certificate
    /// * `private_key_url` : Optional URL to use for retrieving the priavte key from
    ///
    /// # Returns
    ///
    /// The UrlBasedCertificateRetriever
    ///
    pub async fn new(
        client: &reqwest::Client,
        certificate_url: &str,
        private_key_url: Option<&str>,
    ) -> Result<Self, Box<dyn Error>> {
        let cert_and_key =
            get_url_based_certificate_and_private_key(client, certificate_url, private_key_url)
                .await?;
        let me: UrlBasedCertificateRetriever;
        match private_key_url {
            Some(pk_url) => {
                me = UrlBasedCertificateRetriever {
                    client: Arc::new(client.clone()),
                    certificate_url: certificate_url.to_string(),
                    private_key_url: Some(pk_url.to_string()),
                    certificate_and_private_key: Arc::new(RwLock::new(cert_and_key)),
                };
            }
            None => {
                me = UrlBasedCertificateRetriever {
                    client: Arc::new(client.clone()),
                    certificate_url: certificate_url.to_string(),
                    private_key_url: None,
                    certificate_and_private_key: Arc::new(RwLock::new(cert_and_key)),
                };
            }
        }
        Ok(me)
    }
}

#[async_trait]
impl CertificateRetriever for UrlBasedCertificateRetriever {
    fn get_certificate_and_private_key(&self) -> Result<CertificateAndPrivateKey, Box<dyn Error>> {
        if let Ok(certificate_and_private_key) = self.certificate_and_private_key.read() {
            return Ok(certificate_and_private_key.clone());
        } else {
            return Err("Error getting lock on certificate_and_private_key".into());
        }
    }
    fn get_certificate_raw(&self) -> Result<String, Box<dyn Error>> {
        if let Ok(certificate_and_private_key) = self.certificate_and_private_key.read() {
            return Ok(certificate_and_private_key.certificate.clone());
        } else {
            return Err("Error getting lock on certificate_and_private_key".into());
        }
    }
    fn get_private_key_pem(&self) -> Result<Option<String>, Box<dyn Error>> {
        if let Ok(certificate_and_private_key) = self.certificate_and_private_key.read() {
            return Ok(certificate_and_private_key.private_key_pem.clone());
        } else {
            return Err("Error getting lock on certificate_and_private_key".into());
        }
    }
    async fn refresh(&self) -> Result<(), Box<dyn Error>> {
        let new_cert_and_key: CertificateAndPrivateKey;
        match &self.private_key_url {
            Some(pk_url) => {
                new_cert_and_key = get_url_based_certificate_and_private_key(
                    &self.client,
                    &self.certificate_url,
                    Some(pk_url),
                )
                .await?;
            }
            None => {
                new_cert_and_key = get_url_based_certificate_and_private_key(
                    &self.client,
                    &self.certificate_url,
                    None,
                )
                .await?;
            }
        }
        // Get Write lock to original cert and key
        let mut original_cert_and_key = self.certificate_and_private_key.write().unwrap();
        *original_cert_and_key = new_cert_and_key;
        Ok(())
    }
}

async fn get_url_based_certificate_and_private_key(
    client: &reqwest::Client,
    certificate_url: &str,
    private_key_url: Option<&str>,
) -> Result<CertificateAndPrivateKey, Box<dyn Error>> {
    let mut auth_headers: HeaderMap = HeaderMap::new();
    auth_headers.insert("Authorization", "Bearer Oracle".parse()?);

    debug!("Getting certificate from {:?}", certificate_url);
    let certificate = client
        .get(certificate_url)
        .headers(auth_headers.clone())
        .timeout(Duration::new(60, 0))
        .send()
        .await?
        .text()
        .await?;
    trace!("Got certificate: {:?}", certificate);

    let cert_and_key: CertificateAndPrivateKey;
    match private_key_url {
        Some(url) => {
            let private_key = client
                .get(url)
                .headers(auth_headers.clone())
                .timeout(Duration::new(5, 0))
                .send()
                .await?
                .text()
                .await?;
            trace!("Got private_key: {:?}", private_key);
            cert_and_key = CertificateAndPrivateKey {
                certificate,
                private_key_pem: Some(private_key),
            };
        }
        None => {
            cert_and_key = CertificateAndPrivateKey {
                certificate,
                private_key_pem: None,
            };
        }
    }
    return Ok(cert_and_key);
}
