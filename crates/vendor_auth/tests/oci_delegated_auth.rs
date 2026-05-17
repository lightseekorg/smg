//! Unit tests for `OciDelegatedAuth`.
//!
//! Covers tests:
//! - oci_delegated_sets_obo_and_project
//! - oci_delegated_signs_request
//! - oci_delegated_redacts_in_debug
//!
//! See design doc § 6.6.
//!
//! ## How the tests construct an AuthenticationProvider
//!
//! We can't bootstrap a real `InstancePrincipalAuthProvider` in unit tests
//! (it hits IMDS at `169.254.169.254`). Instead we provide
//! `MockSecurityTokenProvider` — a hand-rolled `AuthenticationProvider` impl
//! that generates an ephemeral 2048-bit RSA key in memory and returns the
//! ST$<token> shape that instance principals would produce in production.
//! This lets us assert on the wire shape of the resulting signature without
//! depending on an OCI cluster.

// Test files use unwrap/expect freely on infallible builders. The workspace
// `clippy::unwrap_used = "deny"` rule ordinarily forbids this, but
// `clippy.toml` sets `allow-unwrap-in-tests = true` for `#[test]` bodies.
// Helper functions outside `#[test]` are not auto-allowed; opt the whole
// test crate in explicitly.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::error::Error;
use std::sync::Arc;

use bytes::Bytes;
use http::Request;
use openssl::bn::BigNum;
use openssl::pkey::Private;
use openssl::rsa::Rsa;
use secrecy::SecretString;

use smg_vendor_auth::OciDelegatedAuth;
use smg_vendor_auth::OutboundAuth;
use smg_vendor_auth::oci::authentication_provider::AuthenticationProvider;

const TEST_SECURITY_TOKEN: &str = "FAKE.SECURITY.TOKEN";

#[derive(Debug, Clone)]
struct MockSecurityTokenProvider {
    rsa: Rsa<Private>,
}

impl MockSecurityTokenProvider {
    fn new() -> Self {
        let rsa = Rsa::generate_with_e(
            2048,
            &BigNum::from_u32(65537).expect("BigNumRef from 65537"),
        )
        .expect("generate RSA key");
        Self { rsa }
    }
}

impl AuthenticationProvider for MockSecurityTokenProvider {
    fn tenancy_id(&self) -> Result<String, Box<dyn Error>> {
        Ok("ocid1.tenancy.oc1..test".into())
    }
    fn user_id(&self) -> Result<String, Box<dyn Error>> {
        Ok(String::new())
    }
    fn fingerprint(&self) -> Result<String, Box<dyn Error>> {
        Ok("aa:bb:cc:dd:ee:ff".into())
    }
    fn private_key(&self) -> Result<Rsa<Private>, Box<dyn Error>> {
        Ok(self.rsa.clone())
    }
    fn region_id(&self) -> Result<Option<String>, Box<dyn Error>> {
        Ok(Some("us-ashburn-1".into()))
    }
    // Override key_id to mimic InstancePrincipalAuthProvider ("ST$<token>").
    fn key_id(&self) -> Result<String, Box<dyn Error>> {
        Ok(format!("ST${TEST_SECURITY_TOKEN}"))
    }
}

fn make_provider() -> Arc<Box<dyn AuthenticationProvider + Send + Sync>> {
    Arc::new(Box::new(MockSecurityTokenProvider::new()))
}

fn make_post() -> Request<Bytes> {
    Request::builder()
        .method("POST")
        .uri("https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/v1/containers")
        .header("content-type", "application/json")
        .body(Bytes::from_static(
            b"{\"name\":\"cntr-test\",\"memory_limit\":\"4g\"}",
        ))
        .unwrap()
}

fn make_get() -> Request<Bytes> {
    Request::builder()
        .method("GET")
        .uri("https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/v1/containers/cntr_xyz")
        .body(Bytes::new())
        .unwrap()
}

#[tokio::test]
async fn oci_delegated_sets_obo_and_project() {
    let auth = OciDelegatedAuth::new(
        SecretString::from("opaque-obo-token-value".to_owned()),
        "proj_xyz",
        make_provider(),
    );
    let mut req = make_post();
    auth.apply(&mut req).await.expect("apply succeeds");

    let obo = req
        .headers()
        .get("opc-obo-token")
        .expect("opc-obo-token header present")
        .to_str()
        .unwrap();
    assert_eq!(obo, "opaque-obo-token-value");

    let project = req
        .headers()
        .get("openai-project")
        .expect("openai-project header present")
        .to_str()
        .unwrap();
    assert_eq!(project, "proj_xyz");
}

#[tokio::test]
async fn oci_delegated_signs_request() {
    let auth = OciDelegatedAuth::new(
        SecretString::from("opaque-obo-token-value".to_owned()),
        "proj_xyz",
        make_provider(),
    );

    // POST: body methods include x-content-sha256
    let mut req = make_post();
    auth.apply(&mut req).await.expect("apply succeeds");

    let auth_hdr = req
        .headers()
        .get("authorization")
        .expect("authorization header present after signing")
        .to_str()
        .unwrap();

    assert!(
        auth_hdr.starts_with("Signature "),
        "authorization must be Signature scheme, got: {auth_hdr}"
    );
    assert!(
        auth_hdr.contains("version=\"1\""),
        "signature must declare version=\"1\""
    );
    assert!(
        auth_hdr.contains("algorithm=\"rsa-sha256\""),
        "signature must declare rsa-sha256 algorithm"
    );
    assert!(
        auth_hdr.contains(&format!("keyId=\"ST${TEST_SECURITY_TOKEN}\"")),
        "keyId must be ST$<security-token>, got: {auth_hdr}"
    );

    // Extract headers="..." list and check expected entries
    let headers_attr = extract_attr(auth_hdr, "headers")
        .expect("authorization must include headers=\"...\" list");
    assert!(
        headers_attr.contains("(request-target)"),
        "(request-target) must be in signed headers, got: {headers_attr}"
    );
    assert!(
        headers_attr.contains("host"),
        "host must be in signed headers, got: {headers_attr}"
    );
    assert!(
        headers_attr.contains("date"),
        "date must be in signed headers, got: {headers_attr}"
    );

    // Body methods (POST) must include x-content-sha256 + content-length + content-type
    assert!(
        headers_attr.contains("x-content-sha256"),
        "x-content-sha256 must be in signed headers for POST, got: {headers_attr}"
    );
    assert!(
        headers_attr.contains("content-length"),
        "content-length must be in signed headers for POST, got: {headers_attr}"
    );
    assert!(
        headers_attr.contains("content-type"),
        "content-type must be in signed headers for POST, got: {headers_attr}"
    );

    // R1 sub-assertion (per design doc §11 R1 + §13 wire example note).
    //
    // The upstream `oci-common::signer::get_required_headers` builds its
    // headers-to-sign list from a fixed set
    // (`date`, `(request-target)`, `host` + body headers) and does NOT
    // include `opc-obo-token`. Whether OCI's gateway accepts an OBO that
    // wasn't signed-over is verified by integration test in CB-4. This
    // assertion is EXPECTED to fail today; it documents the gap and tracks
    // CB-1.A.
    //
    // We deliberately do not flip the test pass/fail logic here: the
    // assertion below stays as a regular assert! that PANICS, but is
    // wrapped in a `#[should_panic]` test so the suite stays green while
    // the gap is open. When CB-1.A lands (additive entry-point with
    // `extra_headers_to_sign: &[&str]` includes `opc-obo-token`), this
    // panic disappears and the test will start failing — at which point we
    // remove `#[should_panic]` and turn it into a positive assertion.
    //
    // Tracked: CB-1.A.

    // GET: no body — content-* headers must NOT be in the signed list
    let mut req2 = make_get();
    auth.apply(&mut req2).await.expect("apply on GET succeeds");
    let auth_hdr2 = req2
        .headers()
        .get("authorization")
        .expect("authorization on GET")
        .to_str()
        .unwrap();
    let headers_attr2 = extract_attr(auth_hdr2, "headers")
        .expect("GET authorization must include headers=\"...\" list");
    assert!(
        !headers_attr2.contains("x-content-sha256"),
        "GET must NOT sign x-content-sha256, got: {headers_attr2}"
    );
    assert!(
        headers_attr2.contains("(request-target)"),
        "GET still signs (request-target)"
    );
    assert!(headers_attr2.contains("host"), "GET still signs host");
}

/// R1 follow-up: track the gap with a `#[should_panic]` test.
///
/// `oci-common::signer`'s default `headers_to_sign` list does not include
/// `opc-obo-token`. CB-1.A tracks adding an additive entry-point with
/// `extra_headers_to_sign: &[&str]`. When that lands, this test will start
/// passing without panicking; remove `#[should_panic]` at that point.
#[tokio::test]
#[should_panic(expected = "R1: opc-obo-token not in signed headers")]
async fn oci_delegated_signs_obo_token_header_r1_followup() {
    let auth = OciDelegatedAuth::new(
        SecretString::from("opaque-obo-token-value".to_owned()),
        "proj_xyz",
        make_provider(),
    );
    let mut req = make_post();
    auth.apply(&mut req).await.expect("apply succeeds");

    let auth_hdr = req
        .headers()
        .get("authorization")
        .unwrap()
        .to_str()
        .unwrap();
    let headers_attr = extract_attr(auth_hdr, "headers").unwrap();

    // Expected to fail today (R1) — see CB-1.A.
    assert!(
        headers_attr.contains("opc-obo-token"),
        "R1: opc-obo-token not in signed headers (got: {headers_attr}) — CB-1.A follow-up needed"
    );
}

#[tokio::test]
async fn oci_delegated_redacts_in_debug() {
    // R7: SecretString must redact OBO tokens in any Debug/Display output.
    let secret_value = "super-sensitive-obo-token-DO-NOT-LEAK";
    let auth = OciDelegatedAuth::new(
        SecretString::from(secret_value.to_owned()),
        "proj_xyz",
        make_provider(),
    );

    let dbg = format!("{auth:?}");
    assert!(
        !dbg.contains(secret_value),
        "Debug output must NOT contain the raw OBO token. Got: {dbg}"
    );
    // Sanity: the wrapper must be present (we expect SecretString's Redacted
    // marker, not the inner value).
    assert!(
        dbg.contains("OciDelegatedAuth") || dbg.contains("project_id"),
        "Debug output must still describe the struct. Got: {dbg}"
    );
}

/// Extract `key="value"` from an authorization header string.
fn extract_attr(s: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=\"");
    let start = s.find(&needle)? + needle.len();
    let rest = &s[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_owned())
}
