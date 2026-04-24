//! Canonical tenant identity and per-request metadata for serving paths.

use std::{net::IpAddr, sync::Arc};

use axum::http::Extensions;
use uuid::Uuid;

pub const DEFAULT_TENANT_HEADER_NAME: &str = "x-smg-tenant-id";
const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum TenantIdentity {
    Authenticated(Arc<str>),
    Header(Arc<str>),
    IpAddress(IpAddr),
    Anonymous,
    Explicit(Arc<str>),
}

impl TenantIdentity {
    #[must_use]
    pub fn into_key(self) -> TenantKey {
        let key = match self {
            Self::Authenticated(id) => format!("auth:{id}"),
            Self::Header(id) => format!("header:{id}"),
            Self::IpAddress(addr) => format!("ip:{addr}"),
            Self::Anonymous => "anonymous".to_string(),
            Self::Explicit(key) => key.to_string(),
        };
        TenantKey::from(key)
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct TenantKey(Arc<str>);

impl TenantKey {
    #[must_use]
    pub fn new(key: impl AsRef<str>) -> Self {
        Self(Arc::from(key.as_ref()))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TenantKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for TenantKey {
    fn from(value: String) -> Self {
        Self(Arc::from(value))
    }
}

impl From<&str> for TenantKey {
    fn from(value: &str) -> Self {
        Self(Arc::from(value))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataPlaneCaller {
    tenant_key: TenantKey,
}

impl DataPlaneCaller {
    #[must_use]
    pub fn new(tenant_key: TenantKey) -> Self {
        Self { tenant_key }
    }

    #[must_use]
    pub fn tenant_key(&self) -> &TenantKey {
        &self.tenant_key
    }

    #[inline]
    #[must_use]
    pub fn authenticated_from_sha256(hash: [u8; 32]) -> Self {
        Self::new(authenticated_tenant_key_from_sha256(hash))
    }
}

#[derive(Debug, Clone)]
pub struct RouteRequestMeta {
    pub tenant_key: TenantKey,
    pub request_charge_id: Uuid,
    extensions: Extensions,
}

impl RouteRequestMeta {
    #[must_use]
    pub fn new(tenant_key: TenantKey) -> Self {
        Self {
            tenant_key,
            request_charge_id: Uuid::now_v7(),
            extensions: Extensions::new(),
        }
    }

    #[must_use]
    pub fn tenant_key(&self) -> &TenantKey {
        &self.tenant_key
    }

    #[must_use]
    pub fn request_charge_id(&self) -> Uuid {
        self.request_charge_id
    }

    #[must_use]
    pub fn with_extension<T>(mut self, value: T) -> Self
    where
        T: Clone + Send + Sync + 'static,
    {
        self.extensions.insert(value);
        self
    }

    #[must_use]
    pub fn extension<T>(&self) -> Option<&T>
    where
        T: Send + Sync + 'static,
    {
        self.extensions.get::<T>()
    }
}

impl PartialEq for RouteRequestMeta {
    fn eq(&self, other: &Self) -> bool {
        self.tenant_key == other.tenant_key && self.request_charge_id == other.request_charge_id
    }
}

impl Eq for RouteRequestMeta {}

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum TenantResolutionError {
    #[error("admin routes require an explicit target tenant id")]
    MissingTargetTenant,
}

#[must_use]
pub fn canonical_tenant_key(identity: TenantIdentity) -> TenantKey {
    identity.into_key()
}

#[inline]
#[must_use]
pub fn authenticated_tenant_key_from_sha256(hash: [u8; 32]) -> TenantKey {
    let mut key = String::with_capacity(5 + hash.len() * 2);
    key.push_str("auth:");
    for byte in hash {
        key.push(HEX_DIGITS[(byte >> 4) as usize] as char);
        key.push(HEX_DIGITS[(byte & 0x0f) as usize] as char);
    }
    TenantKey::from(key)
}

pub fn resolve_admin_target_tenant_key(
    tenant_key: &str,
) -> Result<TenantKey, TenantResolutionError> {
    let tenant_key = tenant_key.trim();
    if tenant_key.is_empty() {
        return Err(TenantResolutionError::MissingTargetTenant);
    }

    Ok(canonical_tenant_key(TenantIdentity::Explicit(Arc::from(
        tenant_key,
    ))))
}

pub fn resolve_admin_target_tenant_id(
    tenant_key: &str,
) -> Result<RouteRequestMeta, TenantResolutionError> {
    resolve_admin_target_tenant_key(tenant_key).map(RouteRequestMeta::new)
}
