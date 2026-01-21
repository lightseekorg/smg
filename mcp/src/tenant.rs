//! Tenant context for multi-tenant MCP operations.

use serde::{Deserialize, Serialize};

/// Unique identifier for a tenant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(String);

impl TenantId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for TenantId {
    fn default() -> Self {
        Self("default".to_string())
    }
}

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for TenantId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for TenantId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Unique identifier for a session within a tenant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(String);

impl SessionId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Rate limits for a tenant.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateLimits {
    pub calls_per_minute: Option<u32>,
    pub calls_per_hour: Option<u32>,
    pub max_concurrent: Option<u32>,
}

impl RateLimits {
    pub fn unlimited() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_per_minute(mut self, limit: u32) -> Self {
        self.calls_per_minute = Some(limit);
        self
    }

    #[must_use]
    pub fn with_per_hour(mut self, limit: u32) -> Self {
        self.calls_per_hour = Some(limit);
        self
    }

    #[must_use]
    pub fn with_max_concurrent(mut self, limit: u32) -> Self {
        self.max_concurrent = Some(limit);
        self
    }
}

/// Per-tenant configuration and state.
#[derive(Debug, Clone, Default)]
pub struct TenantContext {
    pub tenant_id: TenantId,
    pub session_id: SessionId,
    pub rate_limits: RateLimits,
}

impl TenantContext {
    pub fn new(tenant_id: impl Into<TenantId>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            ..Default::default()
        }
    }

    #[must_use]
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = SessionId::new(session_id);
        self
    }

    #[must_use]
    pub fn with_rate_limits(mut self, limits: RateLimits) -> Self {
        self.rate_limits = limits;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_id() {
        let id = TenantId::new("tenant-123");
        assert_eq!(id.as_str(), "tenant-123");
        assert_eq!(id.to_string(), "tenant-123");
    }

    #[test]
    fn test_session_id_default() {
        let id1 = SessionId::default();
        let id2 = SessionId::default();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_rate_limits() {
        let limits = RateLimits::unlimited()
            .with_per_minute(60)
            .with_max_concurrent(5);
        assert_eq!(limits.calls_per_minute, Some(60));
        assert_eq!(limits.max_concurrent, Some(5));
        assert_eq!(limits.calls_per_hour, None);
    }
}
