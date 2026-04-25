//! OCI auth subtree — copied from `oci-rust-sdk/crates/common/src/`.
//!
//! See `NOTICE.md` for license, provenance, and drift-management policy.
//!
//! # Why this module is "forked-and-frozen"
//!
//! Per design doc D8 (`.claude/plans/container-backend-design.md` § 2), SMG
//! does not take a path-dep on `oci-rust-sdk`. We copy the minimal subset of
//! the upstream auth code into this submodule and treat it as forked-and-frozen.
//! Drift management (R8): quarterly diff against upstream, manual re-pull on
//! CVE.
//!
//! # File layout
//!
//! - `signer`, `http_signature` — request signing primitives.
//! - `authentication_provider`, `federation_client`, `certificate_retriever` — traits.
//! - `instance_principals_provider`, `x509_federation_client` — IMDS-based
//!   federation flow (the only v1 provider per D10).
//! - `session_key_supplier`, `private_key_supplier` — RSA key suppliers.
//! - `auth_utils`, `constants`, `file_utils` — utilities.
//! - `jwt_claim_set`, `security_token_container`, `region_definitions` — extra
//!   data types kept for forward-compat with non-IP providers.

// SAFETY: copied OCI source uses `unwrap()` and `expect()` heavily. Workspace
// lints deny these in SMG-authored code; in the `oci` subtree we accept them
// verbatim from upstream UPL-1.0 source. No SMG code outside this module may
// follow this pattern.
//
// We disable the workspace's `unwrap_used`, `expect_used`, `panic`, etc.
// lints inside this subtree to keep the upstream source byte-for-byte
// identical. We also suppress `unused_qualifications` and `unused_parens`
// for the same reason — these are upstream stylistic choices we don't
// modify.
// Disable all clippy lints in the verbatim upstream subtree. SMG-authored
// code outside this module remains under the workspace lint policy.
#![allow(clippy::all, clippy::pedantic, clippy::nursery, clippy::restriction)]
#![allow(
    unused_qualifications,
    unused_parens,
    unused_imports,
    dead_code,
    unused_variables,
    unused_assignments,
    unused_mut
)]

pub mod auth_utils;
pub mod authentication_provider;
pub mod certificate_retriever;
pub mod constants;
pub mod federation_client;
pub mod file_utils;
pub mod http_signature;
pub mod instance_principals_provider;
pub mod jwt_claim_set;
pub mod private_key_supplier;
pub mod region_definitions;
pub mod security_token_container;
pub mod session_key_supplier;
pub mod signer;
pub mod x509_federation_client;

// Public re-exports for the API consumed by signer_adapter.rs and downstream
// SMG callers (e.g. oci_provider_factory.rs).
pub use authentication_provider::AuthenticationProvider;
pub use instance_principals_provider::InstancePrincipalAuthProvider;
