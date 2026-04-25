//! Wire types for the OpenAI Containers REST surface.
//!
//! These types serialize/deserialize to the exact JSON shape used by:
//!
//! - The public OpenAI `/v1/containers` endpoint, and
//! - OCI's OpenAI-compat `/v1/containers` endpoint (D5).
//!
//! Field names are pinned by the Java reference at
//! `RemoteMcpConstants.java:158-168` (CONTAINER_KEY, CONTAINER_ID_KEY,
//! MEMORY_LIMIT, EXPIRES_AFTER, FILE_IDS, CONTAINER_STATUS_RUNNING,
//! CONTAINER_NAME_PREFIX) and `:193-197` (NETWORK_POLICY, ALLOWED_DOMAINS,
//! DOMAIN_SECRETS, DOMAIN_KEY, VALUE_KEY).
//!
//! See design doc `.claude/plans/container-backend-design.md` §7.2 for the
//! field-by-field cross-ref to the Java side and §13 for wire-shape examples.

use serde::{Deserialize, Serialize};

/// A container as returned by the backend.
///
/// `_additionalProperties().get("skills")` from the Java SDK becomes the
/// `skills` field below, modelled as an opaque `serde_json::Value` because
/// it's a per-vendor extension that the backend treats as pass-through.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Container {
    pub id: String,
    /// Always `"container"` on the wire.
    pub object: String,
    /// Unix epoch seconds.
    pub created_at: i64,
    pub status: ContainerStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Unix epoch seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_active_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_after: Option<ExpiresAfter>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_limit: Option<MemoryLimit>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network_policy: Option<NetworkPolicy>,
    #[serde(default)]
    pub file_ids: Vec<String>,
    /// Container "skills" — opaque to the backend; surfaced from Java's
    /// `ContainerCreateResponse._additionalProperties().get("skills")`.
    /// See `AbstractContainerToolProcessor.java:193-233`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills: Option<serde_json::Value>,
}

/// Container lifecycle status.
///
/// `Unknown` is a forward-compat catch-all (R4): if the backend introduces a
/// new status string we still deserialize successfully and let the caller
/// decide whether to treat the container as usable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContainerStatus {
    Running,
    Expired,
    Deleted,
    /// Forward-compat catch-all — never panic on unknown statuses.
    #[serde(other)]
    Unknown,
}

/// Memory limit for a container. Wire format is the lowercase string
/// `"1g" | "4g" | "16g" | "64g"`.
///
/// Pins the four documented sizes per Java
/// `CodeInterpreterProcessor.java:365` (`ContainerCreateParams.MemoryLimit.of`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryLimit {
    #[serde(rename = "1g")]
    Mem1G,
    #[serde(rename = "4g")]
    Mem4G,
    #[serde(rename = "16g")]
    Mem16G,
    #[serde(rename = "64g")]
    Mem64G,
}

/// Network policy applied to the container.
///
/// Wire format is tagged on the `type` field — `"disabled"` or
/// `"allowlist"`. See Java `RemoteMcpConstants.java:193-197` for the field
/// names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NetworkPolicy {
    Disabled,
    Allowlist {
        #[serde(default)]
        allowed_domains: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        domain_secrets: Vec<DomainSecret>,
    },
}

/// A `(domain, value)` pair forwarded as part of an `Allowlist` network policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainSecret {
    pub domain: String,
    pub value: String,
}

/// Expiry policy for a container.
///
/// `anchor` is `"last_active_at"` for the OpenAI surface today. `minutes`
/// is the idle window before the container is reclaimed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpiresAfter {
    pub anchor: String,
    pub minutes: u32,
}

/// Parameters for [`ContainerBackend::create`](crate::ContainerBackend::create).
///
/// All fields are optional except that the wire format always carries at
/// least an empty `file_ids` array (see §13.1 wire example).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CreateContainerParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_limit: Option<MemoryLimit>,
    /// Always serialized — wire shape matches Java `populateBodyFromConfig`
    /// `bodyBuilder.fileIds(...)` (`CodeInterpreterProcessor.java:366`).
    #[serde(default)]
    pub file_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network_policy: Option<NetworkPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_after: Option<ExpiresAfter>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills: Option<serde_json::Value>,
}

/// Query parameters for [`ContainerBackend::list`](crate::ContainerBackend::list).
#[derive(Debug, Default, Clone)]
pub struct ListQuery {
    pub limit: Option<u32>,
    pub after: Option<String>,
    pub before: Option<String>,
    pub order: Option<ListOrder>,
}

/// Sort order for a list query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListOrder {
    Asc,
    Desc,
}

/// Standard OpenAI list-response envelope.
///
/// `{"object":"list","data":[...],"first_id":..,"last_id":..,"has_more":..}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page<T> {
    pub data: Vec<T>,
    /// Always `"list"` on the wire.
    pub object: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
    pub has_more: bool,
}
