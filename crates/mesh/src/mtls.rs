//! mTLS (mutual TLS) support for mesh cluster communication
//!
//! Provides optional mTLS encryption for gRPC mesh connections using rustls.
//! Supports certificate rotation without restart.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, Result};
use openssl::x509::X509;
use rustls::{
    client::{
        danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
        WebPkiServerVerifier,
    },
    pki_types::{CertificateDer, PrivateKeyDer, ServerName, UnixTime},
    server::WebPkiClientVerifier,
    CertificateError, ClientConfig, DigitallySignedStruct, Error as RustlsError, RootCertStore,
    ServerConfig, SignatureScheme,
};
use rustls_pemfile::{certs, pkcs8_private_keys};
use tokio::{fs, sync::RwLock};
use tonic::transport::{Certificate, ClientTlsConfig, Identity, ServerTlsConfig};
use tracing::{info, warn};

const SPIFFE_PREFIX: &str = "spiffe://oraclecorp.com/oci/";

/// SPIFFE service name used by SMG Region Agents.
pub const REGION_AGENT_SPIFFE_SERVICE: &str = "smg-region-agent";

/// Parsed SMG SPIFFE identity carried in a certificate URI SAN.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpiffeIdentity {
    uri: String,
    realm: String,
    environment: String,
    region_id: String,
    service: String,
}

impl SpiffeIdentity {
    /// Parse a SPIFFE URI using the SMG identity shape.
    pub fn parse(uri: impl Into<String>) -> Result<Self> {
        let uri = uri.into();
        if uri.trim() != uri {
            return Err(anyhow::anyhow!(
                "SPIFFE URI SAN must not contain surrounding whitespace"
            ));
        }

        let remainder = uri
            .strip_prefix(SPIFFE_PREFIX)
            .ok_or_else(|| anyhow::anyhow!("SPIFFE URI SAN must start with '{SPIFFE_PREFIX}'"))?;
        let segments = remainder.split('/').collect::<Vec<_>>();
        if segments.len() != 6 {
            return Err(anyhow::anyhow!(
                "SPIFFE URI SAN must use '<realm>/<environment>/region/<region_id>/service/<service>'"
            ));
        }

        if segments[2] != "region" || segments[4] != "service" {
            return Err(anyhow::anyhow!(
                "SPIFFE URI SAN must include region and service path markers"
            ));
        }

        for segment in &segments {
            if segment.trim().is_empty() {
                return Err(anyhow::anyhow!(
                    "SPIFFE URI SAN contains an empty realm, environment, region, or service segment"
                ));
            }
            if segment.contains(['?', '#']) {
                return Err(anyhow::anyhow!(
                    "SPIFFE URI SAN segments must not contain query or fragment markers"
                ));
            }
        }

        let realm = segments[0].to_string();
        let environment = segments[1].to_string();
        let region_id = segments[3].to_string();
        let service = segments[5].to_string();

        Ok(Self {
            uri,
            realm,
            environment,
            region_id,
            service,
        })
    }

    /// Parse a SPIFFE URI and require the SMG Region Agent service identity.
    pub fn parse_region_agent(uri: impl Into<String>) -> Result<Self> {
        let identity = Self::parse(uri)?;
        if identity.service != REGION_AGENT_SPIFFE_SERVICE {
            return Err(anyhow::anyhow!(
                "SPIFFE service must be '{REGION_AGENT_SPIFFE_SERVICE}', got '{}'",
                identity.service
            ));
        }
        Ok(identity)
    }

    /// Extract the single SMG Region Agent SPIFFE URI SAN from a DER certificate.
    pub fn from_certificate_der(cert_der: &[u8]) -> Result<Self> {
        let cert = X509::from_der(cert_der).context("failed to parse peer certificate")?;
        let subject_alt_names = cert
            .subject_alt_names()
            .ok_or_else(|| anyhow::anyhow!("peer certificate has no subject alternative names"))?;

        let mut identities = Vec::new();
        for san in subject_alt_names {
            let Some(uri) = san.uri() else {
                continue;
            };
            if uri.starts_with(SPIFFE_PREFIX) {
                identities.push(Self::parse_region_agent(uri)?);
            }
        }

        match identities.len() {
            1 => Ok(identities.remove(0)),
            0 => Err(anyhow::anyhow!(
                "peer certificate does not contain an SMG Region Agent SPIFFE URI SAN"
            )),
            _ => Err(anyhow::anyhow!(
                "peer certificate contains multiple SMG Region Agent SPIFFE URI SANs"
            )),
        }
    }

    /// Return the original URI SAN.
    pub fn uri(&self) -> &str {
        &self.uri
    }

    /// Return the OCI realm segment.
    pub fn realm(&self) -> &str {
        &self.realm
    }

    /// Return the deployment environment segment.
    pub fn environment(&self) -> &str {
        &self.environment
    }

    /// Return the region id segment.
    pub fn region_id(&self) -> &str {
        &self.region_id
    }

    /// Return the service segment.
    pub fn service(&self) -> &str {
        &self.service
    }
}

/// Expected server identity metadata for one outbound mesh authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedPeerTlsIdentity {
    tls_server_name: String,
    spiffe_identity: SpiffeIdentity,
}

impl ExpectedPeerTlsIdentity {
    /// Build outbound TLS identity metadata for a configured mesh peer.
    pub fn new(
        tls_server_name: impl Into<String>,
        spiffe_identity: SpiffeIdentity,
    ) -> Result<Self> {
        let tls_server_name = tls_server_name.into();
        if tls_server_name.trim().is_empty() {
            return Err(anyhow::anyhow!(
                "expected peer TLS server name must not be empty"
            ));
        }
        Ok(Self {
            tls_server_name,
            spiffe_identity,
        })
    }

    /// Return the DNS server name used for SNI and DNS SAN validation.
    pub fn tls_server_name(&self) -> &str {
        &self.tls_server_name
    }

    /// Return the exact SPIFFE identity required in the server URI SAN.
    pub fn spiffe_identity(&self) -> &SpiffeIdentity {
        &self.spiffe_identity
    }
}

/// Server verifier that adds exact Region Agent SPIFFE URI SAN validation.
#[derive(Debug)]
struct SpiffeServerVerifier {
    inner: Arc<WebPkiServerVerifier>,
    expected_identity: SpiffeIdentity,
}

impl SpiffeServerVerifier {
    /// Build a server verifier from trusted roots and the expected peer identity.
    fn new(root_store: RootCertStore, expected_identity: SpiffeIdentity) -> Result<Self> {
        let inner = WebPkiServerVerifier::builder(Arc::new(root_store))
            .build()
            .context("failed to build server certificate verifier")?;
        Ok(Self {
            inner,
            expected_identity,
        })
    }

    /// Convert SPIFFE identity validation errors into TLS verification failures.
    fn invalid_spiffe_identity() -> RustlsError {
        RustlsError::InvalidCertificate(CertificateError::ApplicationVerificationFailure)
    }
}

impl ServerCertVerifier for SpiffeServerVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        ocsp_response: &[u8],
        now: UnixTime,
    ) -> std::result::Result<ServerCertVerified, RustlsError> {
        let verified = self.inner.verify_server_cert(
            end_entity,
            intermediates,
            server_name,
            ocsp_response,
            now,
        )?;
        let actual_identity = SpiffeIdentity::from_certificate_der(end_entity.as_ref())
            .map_err(|_| Self::invalid_spiffe_identity())?;
        if actual_identity != self.expected_identity {
            return Err(Self::invalid_spiffe_identity());
        }
        Ok(verified)
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, RustlsError> {
        self.inner.verify_tls12_signature(message, cert, dss)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, RustlsError> {
        self.inner.verify_tls13_signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.inner.supported_verify_schemes()
    }
}

/// mTLS configuration
#[derive(Debug, Clone)]
pub struct MTLSConfig {
    /// Path to CA certificate file
    pub ca_cert_path: PathBuf,
    /// Path to server certificate file
    pub server_cert_path: PathBuf,
    /// Path to server private key file
    pub server_key_path: PathBuf,
    /// Path to client certificate file
    pub client_cert_path: PathBuf,
    /// Path to client private key file
    pub client_key_path: PathBuf,
    /// SPIFFE identities allowed to connect to this mesh server.
    pub allowed_peer_identities: Vec<SpiffeIdentity>,
    /// Expected server TLS metadata for each outbound peer authority.
    pub expected_peer_tls_by_authority: BTreeMap<String, ExpectedPeerTlsIdentity>,
    /// Whether to require client certificates
    pub require_client_cert: bool,
    /// Certificate rotation check interval
    pub rotation_check_interval: Duration,
}

impl Default for MTLSConfig {
    fn default() -> Self {
        Self {
            ca_cert_path: PathBuf::from("/etc/ssl/certs/ca-certificates.crt"),
            server_cert_path: PathBuf::from("/etc/ssl/certs/server.crt"),
            server_key_path: PathBuf::from("/etc/ssl/private/server.key"),
            client_cert_path: PathBuf::from("/etc/ssl/certs/client.crt"),
            client_key_path: PathBuf::from("/etc/ssl/private/client.key"),
            allowed_peer_identities: Vec::new(),
            expected_peer_tls_by_authority: BTreeMap::new(),
            require_client_cert: true,
            rotation_check_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

impl MTLSConfig {
    /// Validate that a peer SPIFFE identity is in the configured allowlist.
    pub fn validate_allowed_peer_identity(&self, identity: &SpiffeIdentity) -> Result<()> {
        if self.allowed_peer_identities.is_empty()
            || self.allowed_peer_identities.contains(identity)
        {
            return Ok(());
        }

        Err(anyhow::anyhow!(
            "peer SPIFFE identity '{}' is not configured",
            identity.uri()
        ))
    }

    /// Return the expected peer TLS metadata for an outbound authority.
    pub fn expected_peer_tls_for_authority(
        &self,
        authority: &str,
    ) -> Option<&ExpectedPeerTlsIdentity> {
        self.expected_peer_tls_by_authority.get(authority)
    }

    /// Return true when outbound peers must have configured SPIFFE identities.
    pub fn requires_expected_peer_identity(&self) -> bool {
        !self.expected_peer_tls_by_authority.is_empty()
    }
}

/// mTLS certificate manager
#[derive(Debug)]
pub struct MTLSManager {
    config: MTLSConfig,
    server_config: Arc<RwLock<Option<Arc<ServerConfig>>>>,
    client_config: Arc<RwLock<Option<Arc<ClientConfig>>>>,
}

impl MTLSManager {
    /// Create a new mTLS manager
    pub fn new(config: MTLSConfig) -> Self {
        Self {
            config,
            server_config: Arc::new(RwLock::new(None)),
            client_config: Arc::new(RwLock::new(None)),
        }
    }

    /// Load server TLS configuration
    pub async fn load_server_config(&self) -> Result<Arc<ServerConfig>> {
        let certs = self.load_certs(&self.config.server_cert_path).await?;
        let key = self.load_private_key(&self.config.server_key_path).await?;
        let root_store = self.load_root_store().await?;

        let builder = ServerConfig::builder();
        let mut server_config = if self.config.require_client_cert {
            let verifier = WebPkiClientVerifier::builder(Arc::new(root_store))
                .build()
                .context("failed to build client certificate verifier")?;
            builder
                .with_client_cert_verifier(verifier)
                .with_single_cert(certs, key)?
        } else {
            builder.with_no_client_auth().with_single_cert(certs, key)?
        };

        // Enable ALPN for HTTP/2
        server_config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

        let config = Arc::new(server_config);
        *self.server_config.write().await = Some(config.clone());
        Ok(config)
    }

    /// Load client TLS configuration
    pub async fn load_client_config(&self) -> Result<Arc<ClientConfig>> {
        let root_store = self.load_root_store().await?;
        let certs = self.load_certs(&self.config.client_cert_path).await?;
        let key = self.load_private_key(&self.config.client_key_path).await?;

        let mut client_config = ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_client_auth_cert(certs, key)?;

        // Enable ALPN for HTTP/2
        client_config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

        let config = Arc::new(client_config);
        *self.client_config.write().await = Some(config.clone());
        Ok(config)
    }

    /// Load client TLS configuration that also verifies the expected server SPIFFE identity.
    pub async fn load_client_config_for_server_identity(
        &self,
        expected_identity: &SpiffeIdentity,
    ) -> Result<Arc<ClientConfig>> {
        let root_store = self.load_root_store().await?;
        let certs = self.load_certs(&self.config.client_cert_path).await?;
        let key = self.load_private_key(&self.config.client_key_path).await?;
        let verifier = SpiffeServerVerifier::new(root_store, expected_identity.clone())?;

        let mut client_config = ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(verifier))
            .with_client_auth_cert(certs, key)?;

        client_config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

        let config = Arc::new(client_config);
        *self.client_config.write().await = Some(config.clone());
        Ok(config)
    }

    /// Load CA certificate for tonic client TLS configuration
    pub async fn load_ca_certificate(&self) -> Result<Certificate> {
        let ca_cert = fs::read(&self.config.ca_cert_path).await?;
        Ok(Certificate::from_pem(ca_cert))
    }

    /// Load a tonic server TLS config that requires client certificates when enabled.
    pub async fn load_tonic_server_tls_config(&self) -> Result<ServerTlsConfig> {
        let identity = self
            .load_tonic_identity(&self.config.server_cert_path, &self.config.server_key_path)
            .await?;
        let mut config = ServerTlsConfig::new().identity(identity);
        if self.config.require_client_cert {
            config = config
                .client_ca_root(self.load_ca_certificate().await?)
                .client_auth_optional(false);
        }
        Ok(config)
    }

    /// Load a tonic client TLS config that presents the client certificate.
    pub async fn load_tonic_client_tls_config(
        &self,
        domain_name: impl Into<String>,
    ) -> Result<ClientTlsConfig> {
        Ok(ClientTlsConfig::new()
            .domain_name(domain_name)
            .ca_certificate(self.load_ca_certificate().await?)
            .identity(
                self.load_tonic_identity(
                    &self.config.client_cert_path,
                    &self.config.client_key_path,
                )
                .await?,
            ))
    }

    /// Load a tonic client TLS config for use with a custom server verifier.
    pub async fn load_tonic_client_identity_tls_config(
        &self,
        domain_name: impl Into<String>,
    ) -> Result<ClientTlsConfig> {
        Ok(ClientTlsConfig::new().domain_name(domain_name).identity(
            self.load_tonic_identity(&self.config.client_cert_path, &self.config.client_key_path)
                .await?,
        ))
    }

    /// Load a rustls server verifier that enforces the expected SPIFFE URI SAN.
    pub async fn load_server_verifier_for_identity(
        &self,
        expected_identity: &SpiffeIdentity,
    ) -> Result<Arc<dyn ServerCertVerifier>> {
        let root_store = self.load_root_store().await?;
        Ok(Arc::new(SpiffeServerVerifier::new(
            root_store,
            expected_identity.clone(),
        )?))
    }

    /// Validate that a peer SPIFFE identity is allowed by this manager.
    pub fn validate_allowed_peer_identity(&self, identity: &SpiffeIdentity) -> Result<()> {
        self.config.validate_allowed_peer_identity(identity)
    }

    /// Return configured expected TLS metadata for an outbound authority.
    pub fn expected_peer_tls_for_authority(
        &self,
        authority: &str,
    ) -> Option<ExpectedPeerTlsIdentity> {
        self.config
            .expected_peer_tls_for_authority(authority)
            .cloned()
    }

    /// Return true when outbound authorities require explicit expected identities.
    pub fn requires_expected_peer_identity(&self) -> bool {
        self.config.requires_expected_peer_identity()
    }

    /// Load a root store from the configured CA bundle.
    async fn load_root_store(&self) -> Result<RootCertStore> {
        let mut root_store = RootCertStore::empty();
        for cert in self.load_certs(&self.config.ca_cert_path).await? {
            root_store.add(cert)?;
        }
        Ok(root_store)
    }

    /// Load a tonic PEM identity from certificate and private key paths.
    async fn load_tonic_identity(&self, cert_path: &Path, key_path: &Path) -> Result<Identity> {
        let cert = fs::read(cert_path).await?;
        let key = fs::read(key_path).await?;
        Ok(Identity::from_pem(cert, key))
    }

    /// Load certificates from file
    async fn load_certs(&self, path: &Path) -> Result<Vec<CertificateDer<'static>>> {
        let cert_data = fs::read(path).await?;
        let certs = certs(&mut cert_data.as_slice()).collect::<Result<Vec<_>, _>>()?;
        Ok(certs)
    }

    /// Load private key from file
    async fn load_private_key(&self, path: &Path) -> Result<PrivateKeyDer<'static>> {
        let key_data = fs::read(path).await?;
        let mut keys =
            pkcs8_private_keys(&mut key_data.as_slice()).collect::<Result<Vec<_>, _>>()?;

        if keys.is_empty() {
            return Err(anyhow::anyhow!("No private key found in file"));
        }

        Ok(PrivateKeyDer::Pkcs8(keys.remove(0)))
    }

    /// Start certificate rotation monitoring
    #[expect(
        clippy::disallowed_methods,
        reason = "fire-and-forget background monitor; rotation runs for the process lifetime and does not need explicit join"
    )]
    pub fn start_rotation_monitor(&self) {
        let config = self.config.clone();
        let server_config = self.server_config.clone();
        let client_config = self.client_config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.rotation_check_interval);
            loop {
                interval.tick().await;

                // Check if certificates have changed
                if let Err(e) =
                    Self::check_and_reload_certs(&config, &server_config, &client_config).await
                {
                    warn!("Error checking certificate rotation: {}", e);
                }
            }
        });
    }

    /// Check and reload certificates if they have changed
    async fn check_and_reload_certs(
        config: &MTLSConfig,
        _server_config: &Arc<RwLock<Option<Arc<ServerConfig>>>>,
        _client_config: &Arc<RwLock<Option<Arc<ClientConfig>>>>,
    ) -> Result<()> {
        // Get file modification times
        let server_cert_mtime = fs::metadata(&config.server_cert_path).await?.modified()?;
        let server_key_mtime = fs::metadata(&config.server_key_path).await?.modified()?;
        let ca_cert_mtime = fs::metadata(&config.ca_cert_path).await?.modified()?;

        // TODO: Compare with cached modification times
        // For now, we'll just log that rotation monitoring is active
        info!(
            "Certificate rotation check: server_cert={:?}, server_key={:?}, ca_cert={:?}",
            server_cert_mtime, server_key_mtime, ca_cert_mtime
        );

        // Reload if certificates have changed
        // This is a simplified version - in production, you'd compare mtimes
        Ok(())
    }

    /// Get current server config (for use with tonic)
    pub async fn get_server_config(&self) -> Option<Arc<ServerConfig>> {
        self.server_config.read().await.clone()
    }

    /// Get current client config (for use with tonic)
    pub async fn get_client_config(&self) -> Option<Arc<ClientConfig>> {
        self.client_config.read().await.clone()
    }
}

/// Optional mTLS manager
pub type OptionalMTLSManager = Option<Arc<MTLSManager>>;

/// Stub for `tonic::transport::Endpoint::tls_config_with_verifier`, which is
/// referenced by `service.rs::configure_mtls_endpoint_for_peer` but does not
/// exist in the workspace-pinned tonic 0.14.2.
///
/// **TODO(cross-region mTLS)**: Properly install `verifier` into the rustls
/// `ClientConfig` and feed that into tonic via a custom `Connector` (or
/// upgrade tonic to a version that exposes the verifier hook directly).
/// Today this stub silently drops the verifier and only applies the base
/// `ClientTlsConfig`, which means SPIFFE-identity enforcement is **off** on
/// outbound mesh connections that went through this path.
///
/// Returning `Result` keeps the call-site shape identical to what the
/// missing tonic API would have provided, so the eventual fix only touches
/// this function's body.
pub fn endpoint_tls_config_with_verifier(
    endpoint: tonic::transport::Endpoint,
    tls_config: ClientTlsConfig,
    _verifier: Arc<dyn ServerCertVerifier>,
) -> std::result::Result<tonic::transport::Endpoint, tonic::transport::Error> {
    warn!(
        "endpoint_tls_config_with_verifier: SPIFFE server-cert verifier not enforced \
         (tonic 0.14.2 stub); see mtls.rs TODO"
    );
    endpoint.tls_config(tls_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a self-signed certificate with the provided URI SANs.
    fn certificate_der_with_uri_sans(uris: &[&str]) -> Vec<u8> {
        use openssl::{
            asn1::Asn1Time,
            bn::{BigNum, MsbOption},
            hash::MessageDigest,
            nid::Nid,
            pkey::PKey,
            rsa::Rsa,
            x509::{extension::SubjectAlternativeName, X509NameBuilder},
        };

        let rsa = Rsa::generate(2048).expect("RSA key should generate");
        let pkey = PKey::from_rsa(rsa).expect("PKey should build");
        let mut builder = X509::builder().expect("certificate builder should build");
        builder.set_version(2).expect("version should set");

        let mut serial = BigNum::new().expect("serial should allocate");
        serial
            .rand(64, MsbOption::MAYBE_ZERO, false)
            .expect("serial should generate");
        let serial = serial.to_asn1_integer().expect("serial should encode");
        builder
            .set_serial_number(&serial)
            .expect("serial should set");
        builder
            .set_not_before(
                Asn1Time::days_from_now(0)
                    .expect("not_before should build")
                    .as_ref(),
            )
            .expect("not_before should set");
        builder
            .set_not_after(
                Asn1Time::days_from_now(1)
                    .expect("not_after should build")
                    .as_ref(),
            )
            .expect("not_after should set");

        let mut name = X509NameBuilder::new().expect("name builder should build");
        name.append_entry_by_nid(Nid::COMMONNAME, "smg-region-agent")
            .expect("CN should set");
        let name = name.build();
        builder.set_subject_name(&name).expect("subject should set");
        builder.set_issuer_name(&name).expect("issuer should set");
        builder.set_pubkey(&pkey).expect("public key should set");

        let mut san = SubjectAlternativeName::new();
        for uri in uris {
            san.uri(uri);
        }
        let san_extension = san
            .build(&builder.x509v3_context(None, None))
            .expect("SAN extension should build");
        builder
            .append_extension(san_extension)
            .expect("SAN extension should set");
        builder
            .sign(&pkey, MessageDigest::sha256())
            .expect("certificate should sign");
        builder.build().to_der().expect("certificate should encode")
    }

    #[test]
    fn spiffe_identity_parses_region_agent_uri() {
        let identity = SpiffeIdentity::parse_region_agent(
            "spiffe://oraclecorp.com/oci/oc1/prod/region/us-chicago-1/service/smg-region-agent",
        )
        .expect("valid region agent identity should parse");

        assert_eq!(identity.realm(), "oc1");
        assert_eq!(identity.environment(), "prod");
        assert_eq!(identity.region_id(), "us-chicago-1");
        assert_eq!(identity.service(), "smg-region-agent");
    }

    #[test]
    fn spiffe_identity_rejects_invalid_uri_san() {
        let error = SpiffeIdentity::parse_region_agent(
            "spiffe://oraclecorp.com/oci/oc1/prod/region/us-chicago-1/service/other",
        )
        .expect_err("wrong service should be rejected");

        assert!(error.to_string().contains("service"));
    }

    #[test]
    fn spiffe_identity_rejects_missing_region() {
        let error = SpiffeIdentity::parse_region_agent(
            "spiffe://oraclecorp.com/oci/oc1/prod/region//service/smg-region-agent",
        )
        .expect_err("missing region should be rejected");

        assert!(error.to_string().contains("region"));
    }

    #[test]
    fn spiffe_identity_extracts_uri_san_from_der_certificate() {
        let der = certificate_der_with_uri_sans(&[
            "spiffe://oraclecorp.com/oci/oc1/prod/region/us-chicago-1/service/smg-region-agent",
        ]);

        let identity =
            SpiffeIdentity::from_certificate_der(&der).expect("certificate identity should parse");

        assert_eq!(identity.region_id(), "us-chicago-1");
        assert_eq!(identity.realm(), "oc1");
        assert_eq!(identity.environment(), "prod");
    }

    #[test]
    fn spiffe_identity_rejects_certificate_without_region_agent_uri_san() {
        let der = certificate_der_with_uri_sans(&[
            "spiffe://oraclecorp.com/oci/oc1/prod/region/us-chicago-1/service/other",
        ]);

        let error = SpiffeIdentity::from_certificate_der(&der)
            .expect_err("certificate with wrong service should be rejected");

        assert!(error.to_string().contains("SPIFFE"));
    }

    #[test]
    fn mtls_config_accepts_only_allowed_peer_identities() {
        let allowed = SpiffeIdentity::parse_region_agent(
            "spiffe://oraclecorp.com/oci/oc1/prod/region/us-chicago-1/service/smg-region-agent",
        )
        .expect("allowed identity should parse");
        let rejected = SpiffeIdentity::parse_region_agent(
            "spiffe://oraclecorp.com/oci/oc1/prod/region/us-phoenix-1/service/smg-region-agent",
        )
        .expect("rejected identity should parse");
        let config = MTLSConfig {
            allowed_peer_identities: vec![allowed.clone()],
            ..MTLSConfig::default()
        };

        config
            .validate_allowed_peer_identity(&allowed)
            .expect("configured peer identity should be accepted");
        let error = config
            .validate_allowed_peer_identity(&rejected)
            .expect_err("unconfigured peer identity should be rejected");

        assert!(error.to_string().contains("not configured"));
    }

    #[test]
    fn mtls_config_resolves_expected_peer_tls_by_authority() {
        let expected = SpiffeIdentity::parse_region_agent(
            "spiffe://oraclecorp.com/oci/oc1/prod/region/us-chicago-1/service/smg-region-agent",
        )
        .expect("expected identity should parse");
        let expected_tls = ExpectedPeerTlsIdentity::new("sync.example", expected.clone())
            .expect("expected TLS metadata should build");
        let config = MTLSConfig {
            expected_peer_tls_by_authority: BTreeMap::from([(
                "sync.example:9443".to_string(),
                expected_tls.clone(),
            )]),
            ..MTLSConfig::default()
        };

        assert!(config.requires_expected_peer_identity());
        let actual = config
            .expected_peer_tls_for_authority("sync.example:9443")
            .expect("expected TLS metadata should resolve");
        assert_eq!(actual.tls_server_name(), "sync.example");
        assert_eq!(actual.spiffe_identity(), &expected);
        assert_eq!(actual, &expected_tls);
        assert!(config
            .expected_peer_tls_for_authority("other.example:9443")
            .is_none());
    }
}
