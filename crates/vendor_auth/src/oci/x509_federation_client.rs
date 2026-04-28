// Copyright (c) 2023, Oracle and/or its affiliates.
// Licensed under the Universal Permissive License (UPL), Version 1.0.
// Source: https://github.com/oracle/oci-rust-sdk
// Origin commit: 0590d5dcebabc68d9115520e2be5e42f9dbf1ffb
// Copy provenance: copied verbatim from
//   oci-rust-sdk/crates/common/src/x509_federation_client.rs.
// Only modification: `crate::*` paths rewritten to `crate::oci::*` to live
// under the `oci` submodule.

use async_trait::async_trait;
use itertools::Itertools;
use openssl::rsa::Rsa;
use openssl::x509::X509;
use reqwest::header::HeaderMap;
use reqwest::Method;
use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, RwLock};
use tracing::{instrument, trace};
use url::Url;

use crate::oci::auth_utils::sanitize_certificate_string;
use crate::oci::certificate_retriever::CertificateRetriever;
use crate::oci::constants::*;
use crate::oci::federation_client::FederationClient;
use crate::oci::session_key_supplier::SessionKeySupplier;
use crate::oci::signer;

#[derive(Debug, Clone)]
pub struct X509FederationClient {
    client: Arc<reqwest::Client>,
    token: Arc<RwLock<String>>,
    region_id: String,
    tenancy_id: String,
    federation_endpoint: String,
    leaf_certificate_supplier: Box<dyn CertificateRetriever>,
    session_key_supplier: SessionKeySupplier,
    intermediate_certificate_suppliers: Vec<Box<dyn CertificateRetriever>>,
    purpose: String,
}

impl X509FederationClient {
    /// Creates a new X509FederationClient using the values passed in the arguments.
    ///
    /// # Arguments
    ///
    /// * `http_client`: The Reqwest client to use for making X509 calls
    /// * `leaf_certificate_supplier`: The leaf certificate supplier that provides the leaf certififcate and leaf key
    /// * `intermediate_certificate_suppliers`: The intermediate certificate suppliers to use to provide the intermediate certs and optional keys
    ///
    /// # Returns
    ///
    /// An instance of X509FederationClient
    ///
    pub async fn new(
        http_client: &reqwest::Client,
        leaf_certificate_supplier: Box<dyn CertificateRetriever>,
        intermediate_certificate_suppliers: Vec<Box<dyn CertificateRetriever>>,
    ) -> Result<X509FederationClient, Box<dyn Error>> {
        return X509FederationClient::new_with_options(
            None,
            None,
            DEFAULT_PURPOSE.to_string(),
            http_client,
            leaf_certificate_supplier,
            intermediate_certificate_suppliers,
        )
        .await;
    }

    /// Creates a new X509FederationClient using the values passed in the arguments.
    ///
    /// # Arguments
    ///
    /// * `tenancy_id`: Optional tenancy ID to use for the x509 calls
    /// * `federation_endpoint`: Optional federation endpoint to use to make x509 calls
    /// * `purpose`: Optional purpose for the x509 request
    /// * `http_client`: The Reqwest client to use for making X509 calls
    /// * `leaf_certificate_supplier`: The leaf certificate supplier that provides the leaf certififcate and leaf key
    /// * `intermediate_certificate_suppliers`: The intermediate certificate suppliers to use to provide the intermediate certs and optional keys
    ///
    /// # Returns
    ///
    /// An instance of X509FederationClient
    ///
    pub async fn new_with_options(
        tenancy_id: Option<String>,
        federation_endpoint: Option<String>,
        purpose: String,
        http_client: &reqwest::Client,
        leaf_certificate_supplier: Box<dyn CertificateRetriever>,
        intermediate_certificate_suppliers: Vec<Box<dyn CertificateRetriever>>,
    ) -> Result<X509FederationClient, Box<dyn Error>> {
        // Get Region information
        let mut auth_headers: HeaderMap = HeaderMap::new();
        auth_headers.insert("Authorization", "Bearer Oracle".parse()?);

        let get_region_url: &str = &format!("{}{}", METADATA_URL_BASE, REGION_URL_PATH);
        let region = get_instance_metadata(
            http_client,
            get_region_url.to_string(),
            auth_headers.clone(),
        )
        .await?;
        // Get Realm info
        let get_domain_url: &str = &format!("{}{}", METADATA_URL_BASE, REALM_URL_PATH);
        let domain =
            get_instance_metadata(http_client, get_domain_url.to_string(), auth_headers).await?;

        // Use federation endpoint if provided, otherwsie use the auth endpoint derived from IMDS region
        let host: String;
        match federation_endpoint {
            Some(endpoint) => host = endpoint,
            None => {
                host = format!("https://auth.{}.{}/v1/x509", &region, &domain);
            }
        }

        // Get Tenancy ID from leaf cert
        let cert = leaf_certificate_supplier.get_certificate_raw()?;
        let tenancy: String;
        match tenancy_id {
            Some(tenant) => tenancy = tenant,
            None => {
                tenancy = get_tenancy_id_from_certificate(&cert)?;
            }
        }
        // Build session key supplier
        let session_key_supplier = SessionKeySupplier::new()?;
        let fingerprint = x509_fingerprint(&cert)?;

        let token = get_security_token_from_auth_service(
            http_client,
            host.clone(),
            tenancy.clone(),
            purpose,
            fingerprint.clone(),
            leaf_certificate_supplier.clone_box(),
            intermediate_certificate_suppliers.clone(),
            session_key_supplier.clone(),
        )
        .await?;

        Ok(X509FederationClient {
            client: Arc::new(http_client.clone()),
            token: Arc::new(RwLock::new(token)),
            region_id: region,
            tenancy_id: tenancy,
            federation_endpoint: host,
            leaf_certificate_supplier,
            session_key_supplier,
            intermediate_certificate_suppliers,
            purpose: DEFAULT_PURPOSE.to_string(),
        })
    }
}

#[async_trait]
impl FederationClient for X509FederationClient {
    fn get_security_token(&self) -> Result<String, Box<dyn Error>> {
        if let Ok(token_value) = self.token.read() {
            return Ok(token_value.clone());
        } else {
            return Err("Error getting lock on security token".into());
        }
    }

    async fn refresh_and_get_security_token(&self) -> Result<String, Box<dyn Error>> {
        self.refresh().await?;
        if let Ok(token_value) = self.token.read() {
            return Ok(token_value.clone());
        } else {
            return Err("Error getting lock on security token".into());
        }
    }

    fn get_tenancy_id(&self) -> Result<String, Box<dyn Error>> {
        return Ok(self.tenancy_id.to_string());
    }

    fn get_fingerprint(&self) -> Result<String, Box<dyn Error>> {
        let cert = self.leaf_certificate_supplier.get_certificate_raw()?;
        return Ok(x509_fingerprint(&cert)?);
    }

    fn get_region(&self) -> Result<Option<String>, Box<dyn Error>> {
        return Ok(Some(self.region_id.to_string()));
    }

    fn get_private_key(&self) -> Result<Rsa<openssl::pkey::Private>, Box<dyn Error>> {
        return self.session_key_supplier.clone().get_key();
    }

    async fn refresh(&self) -> Result<(), Box<dyn Error>> {
        tracing::debug!("Refresh federation client started");
        // Refresh Session key supplier
        self.session_key_supplier.refresh().await;
        // Refresh leaf certificate supplier
        tracing::debug!("Refresh leaf certificate supplier");
        self.leaf_certificate_supplier.refresh().await?;
        tracing::debug!("Refresh leaf certificate supplier completed");

        // Refresh all intermediate certificate suppliers
        tracing::debug!("Refresh intermediate certificate supplier");
        for cert_supplier in &self.intermediate_certificate_suppliers {
            cert_supplier.refresh().await?;
        }
        tracing::debug!("Refresh intermediate certificate supplier completed");

        let mut intermediate_certificate_suppliers: Vec<Box<dyn CertificateRetriever>> = vec![];
        for certficate_retreiver in &self.intermediate_certificate_suppliers {
            intermediate_certificate_suppliers.push(certficate_retreiver.clone_box());
        }

        let token = get_security_token_from_auth_service(
            self.client.as_ref(),
            self.federation_endpoint.clone(),
            self.tenancy_id.clone(),
            self.purpose.clone(),
            self.get_fingerprint()
                .expect("Error getting fingerprint from client"),
            self.leaf_certificate_supplier.clone_box(),
            intermediate_certificate_suppliers,
            self.session_key_supplier.clone(),
        )
        .await
        .expect("Error refreshing token from IDDP");

        let mut new_token = self.token.write().unwrap();
        *new_token = token;
        tracing::debug!("Refresh federation client completed");
        Ok(())
    }
}

#[instrument]
async fn get_security_token_from_auth_service(
    client: &reqwest::Client,
    host: String,
    tenancy_id: String,
    purpose: String,
    fingerprint: String,
    leaf_certificate_supplier: Box<dyn CertificateRetriever>,
    intermediate_certificate_suppliers: Vec<Box<dyn CertificateRetriever>>,
    session_key_supplier: SessionKeySupplier,
) -> Result<String, Box<dyn Error>> {
    // Extract Private Key and Tenancy id from Leaf Cert
    let leaf_certificate = leaf_certificate_supplier.get_certificate_raw()?;
    let leaf_key = leaf_certificate_supplier
        .get_private_key_pem()
        .expect("Leaf cert private key missing")
        .ok_or("Private key missing from leaf certitificate supplier")?;

    // Get Optional intermediate certs
    let mut intermediate_certificates: Vec<String> = Vec::new();
    for cert_supplier in &intermediate_certificate_suppliers {
        intermediate_certificates.push(sanitize_certificate_string(
            cert_supplier.get_certificate_raw()?.clone(),
        ));
    }

    let public_key = session_key_supplier.clone().get_key_pair()?.0;

    // Get Token from Auth Service
    // Build signing information and sign via leaf private key
    let key_id = format!("{}/fed-x509-sha256/{}", &tenancy_id, &fingerprint);

    // Build Request
    let request = TokenRequest {
        certificate: sanitize_certificate_string(leaf_certificate.clone()),
        public_key: sanitize_certificate_string(public_key),
        intermediate_certificates,
        fingerprint_algorithm: DEFAULT_FINGERPRINT_ALGORITHM.to_string(),
        purpose,
    };

    // Serialize request bdy
    let jwt_request_body = serde_json::to_string(&request)?;

    let url = Url::parse(&host)?;
    let required_headers = signer::get_required_headers_from_key_and_id(
        Method::POST,
        &jwt_request_body,
        HeaderMap::new(),
        url.clone(),
        Rsa::private_key_from_pem(leaf_key.as_bytes())?,
        key_id.to_string(),
        HashMap::new(),
        false,
    );
    trace!(
        "Sending http post request to {} \n with headers : {:?}",
        url,
        required_headers
    );
    let response = client
        .post(url)
        .body(jwt_request_body)
        .headers(required_headers)
        .send()
        .await?;
    trace!("Response received from the service : {:?}", response);
    if response.status().is_success() {
        let token_response: TokenResponse = response.json().await?;
        tracing::info!("Refreshed the Session Token successfully");
        return Ok(token_response.token);
    }
    return Err(format!(
        "Failed to get token from auth service status {}",
        response.status().as_str()
    )
    .into());
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TokenResponse {
    pub token: String,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenRequest {
    pub certificate: String,
    pub intermediate_certificates: Vec<String>,
    pub public_key: String,
    pub fingerprint_algorithm: String,
    pub purpose: String,
}

async fn get_instance_metadata(
    client: &reqwest::Client,
    get_region_url: String,
    auth_headers: HeaderMap,
) -> Result<String, Box<dyn Error>> {
    let response = client
        .get(get_region_url)
        .headers(auth_headers)
        .send()
        .await?
        .text()
        .await?
        .trim()
        .to_lowercase();
    Ok(response)
}

fn get_tenancy_id_from_certificate(cert: &str) -> Result<String, Box<dyn Error>> {
    let cert = cert.as_bytes();
    let cert = X509::from_pem(cert)?;
    let subject = cert.subject_name();
    for entry in subject.entries() {
        if let Ok(value) = entry.data().as_utf8() {
            let value_str = value.to_string();
            if value_str.starts_with("opc-tenant:") {
                return Ok(value_str["opc-tenant:".len()..].to_string());
            } else if value_str.starts_with("opc-identity:") {
                return Ok(value_str["opc-identity:".len()..].to_string());
            }
        }
    }
    return Err(format!("Tenancy Id not found in cert").into());
}

fn x509_fingerprint(cert: &String) -> Result<String, Box<dyn Error>> {
    let cert = cert.as_bytes();
    let cert = X509::from_pem(cert)?;
    let cert = cert.digest(openssl::hash::MessageDigest::sha256())?;
    let fingerprint = format!("{:02X}", cert.iter().format(":"));
    Ok(fingerprint)
}
