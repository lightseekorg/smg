// Copyright (c) 2023, Oracle and/or its affiliates.
// Licensed under the Universal Permissive License (UPL), Version 1.0.
// Source: https://github.com/oracle/oci-rust-sdk
// Origin commit: 0590d5dcebabc68d9115520e2be5e42f9dbf1ffb
// Copy provenance: copied verbatim from
//   oci-rust-sdk/crates/common/src/instance_principals_provider.rs.
// Only modifications:
//   1. `crate::certficate_retreiver` import rewritten to
//      `crate::oci::certificate_retriever` (file rename per design doc D8).
//   2. Other `crate::*` paths rewritten to `crate::oci::*` to live under the
//      `oci` submodule.

use crate::oci::authentication_provider::AuthenticationProvider;
use crate::oci::certificate_retriever::UrlBasedCertificateRetriever;
use crate::oci::constants::*;
use crate::oci::federation_client::FederationClient;
use crate::oci::x509_federation_client::X509FederationClient;
use openssl::pkey::Private;
use openssl::rsa::Rsa;
use std::error::Error;
use std::sync::Arc;
use tokio::time::Duration as TDuration;

const DEFAULT_REFRESH_INTERVAL: u64 = 50;

/// Struct defining a instance principals based Authentication Provider that reads all authentication information from IMDS
#[derive(Debug, Clone)]
pub struct InstancePrincipalAuthProvider {
    /// The federation client associated with this provider
    federation_client: Arc<Box<dyn FederationClient>>,
}

impl AuthenticationProvider for InstancePrincipalAuthProvider {
    fn tenancy_id(&self) -> Result<String, Box<dyn Error>> {
        return self.federation_client.get_tenancy_id();
    }
    fn fingerprint(&self) -> Result<String, Box<dyn Error>> {
        return self.federation_client.get_fingerprint();
    }
    fn user_id(&self) -> Result<String, Box<dyn Error>> {
        Ok(EMPTY_STRING.to_string())
    }
    fn private_key(&self) -> Result<Rsa<Private>, Box<dyn Error>> {
        return self.federation_client.get_private_key();
    }
    fn region_id(&self) -> Result<Option<String>, Box<dyn Error>> {
        return self.federation_client.get_region();
    }
    fn key_id(&self) -> Result<String, Box<dyn Error>> {
        return Ok(format!(
            "ST${}",
            &self.federation_client.get_security_token()?
        ));
    }
}

impl InstancePrincipalAuthProvider {
    /// Creates a new Instance Principals Provider
    ///
    /// A new instance principals authentication provider
    ///
    pub async fn new() -> Result<InstancePrincipalAuthProvider, Box<dyn Error>> {
        InstancePrincipalAuthProvider::new_with_options(
            &reqwest::Client::builder().build()?,
            None,
            DEFAULT_REFRESH_INTERVAL,
        )
        .await
    }

    /// Creates a new Instance Prinicpals Authentication Provider
    /// # Arguments
    ///
    /// * `client`: Reqwest client to use
    /// * `federation_endpoint`: Federation endpoint to use to make x509 calls
    /// * `refresh_interval`: The refresh interval for refreshing the instance principals provider
    ///
    /// # Returns
    ///
    /// A new instance principals authentication provider
    ///
    pub async fn new_with_options(
        client: &reqwest::Client,
        federation_endpoint: Option<String>,
        refresh_interval: u64,
    ) -> Result<InstancePrincipalAuthProvider, Box<dyn Error>> {
        // Leaf cert
        let leaf_certificate_retreiver = UrlBasedCertificateRetriever::new(
            client,
            &format!("{}{}", METADATA_URL_BASE, LEAF_CERTIFICATE_URL_PATH),
            Some(&format!(
                "{}{}",
                METADATA_URL_BASE, LEAF_CERTIFICATE_PRIVATE_KEY_URL_PATH
            )),
        )
        .await?;

        // Intermediate cert
        let intermediate_certificate_retriever = UrlBasedCertificateRetriever::new(
            client,
            &format!("{}{}", METADATA_URL_BASE, INTERMEDIATE_CERTIFICATE_URL_PATH),
            None,
        )
        .await?;
        let federation_client = X509FederationClient::new_with_options(
            None,
            federation_endpoint,
            DEFAULT_PURPOSE.to_string(),
            &client,
            Box::new(leaf_certificate_retreiver),
            vec![Box::new(intermediate_certificate_retriever)],
        )
        .await?;
        let me = InstancePrincipalAuthProvider {
            federation_client: Arc::new(Box::new(federation_client)),
        };
        let shared_federation_client = me.federation_client.clone();
        tokio::spawn(async move {
            tracing::info!("Starting refresh loop");
            loop {
                tracing::info!(
                    "Sleeping for {} minutes, before refreshing Instance Principals token",
                    refresh_interval
                );
                tokio::time::sleep(TDuration::from_secs(refresh_interval * 60)).await;
                tracing::info!("Refreshing Federation client");
                shared_federation_client
                    .refresh()
                    .await
                    .expect("Error Refressing federation client");
                tracing::info!("Refreshing Federation client completed successfully");
            }
        });

        Ok(me)
    }
}
