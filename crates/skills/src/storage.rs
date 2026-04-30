use async_trait::async_trait;

use crate::types::{
    BundleTokenClaim, ContinuationCookieClaim, SkillRecord, SkillVersionRecord, TenantAliasRecord,
};

/// Shared error type for skills metadata/token persistence contracts.
#[derive(Debug, thiserror::Error)]
pub enum SkillsStoreError {
    #[error("storage error: {0}")]
    Storage(String),

    #[error("invalid data: {0}")]
    InvalidData(String),
}

/// Result alias for skills persistence operations.
pub type SkillsStoreResult<T> = Result<T, SkillsStoreError>;

/// Persistence contract for skill metadata rows and immutable version rows.
///
/// The combined write methods are part of the contract intentionally: CRUD
/// callers must not publish a skill projection separately from the version row
/// that makes the projection resolvable.
#[async_trait]
pub trait SkillMetadataStore: Send + Sync + 'static {
    async fn put_skill(&self, record: SkillRecord) -> SkillsStoreResult<()>;

    async fn put_skill_with_initial_version(
        &self,
        skill: SkillRecord,
        version: SkillVersionRecord,
    ) -> SkillsStoreResult<()>;

    async fn get_skill(
        &self,
        tenant_id: &str,
        skill_id: &str,
    ) -> SkillsStoreResult<Option<SkillRecord>>;

    async fn list_skills(&self, tenant_id: &str) -> SkillsStoreResult<Vec<SkillRecord>>;

    async fn delete_skill(&self, tenant_id: &str, skill_id: &str) -> SkillsStoreResult<bool>;

    async fn put_skill_version(&self, record: SkillVersionRecord) -> SkillsStoreResult<()>;

    async fn put_skill_version_and_update_skill(
        &self,
        version: SkillVersionRecord,
        skill: SkillRecord,
    ) -> SkillsStoreResult<()>;

    async fn get_skill_version(
        &self,
        skill_id: &str,
        version: &str,
    ) -> SkillsStoreResult<Option<SkillVersionRecord>>;

    async fn list_skill_versions(
        &self,
        skill_id: &str,
    ) -> SkillsStoreResult<Vec<SkillVersionRecord>>;

    async fn delete_skill_version(&self, skill_id: &str, version: &str) -> SkillsStoreResult<bool>;
}

/// Persistence contract for tenant-alias rows.
#[async_trait]
pub trait TenantAliasStore: Send + Sync + 'static {
    async fn put_tenant_alias(&self, record: TenantAliasRecord) -> SkillsStoreResult<()>;

    async fn get_tenant_alias(
        &self,
        alias_tenant_id: &str,
    ) -> SkillsStoreResult<Option<TenantAliasRecord>>;

    async fn delete_tenant_alias(&self, alias_tenant_id: &str) -> SkillsStoreResult<bool>;
}

/// Persistence contract for bundle-token rows keyed by a deterministic secret hash.
///
/// Callers are expected to hash the presented bearer secret before invoking the
/// lookup and revoke methods on this contract.
#[async_trait]
pub trait BundleTokenStore: Send + Sync + 'static {
    async fn put_bundle_token(&self, claim: BundleTokenClaim) -> SkillsStoreResult<()>;

    async fn get_bundle_token(
        &self,
        token_hash: &str,
    ) -> SkillsStoreResult<Option<BundleTokenClaim>>;

    async fn revoke_bundle_token(&self, token_hash: &str) -> SkillsStoreResult<bool>;

    async fn revoke_bundle_tokens_for_exec(&self, exec_id: &str) -> SkillsStoreResult<usize>;
}

/// Persistence contract for continuation-cookie rows keyed by a deterministic secret hash.
///
/// Callers are expected to hash the presented continuation secret before
/// invoking the lookup and revoke methods on this contract.
#[async_trait]
pub trait ContinuationCookieStore: Send + Sync + 'static {
    async fn put_continuation_cookie(
        &self,
        claim: ContinuationCookieClaim,
    ) -> SkillsStoreResult<()>;

    async fn get_continuation_cookie(
        &self,
        cookie_hash: &str,
    ) -> SkillsStoreResult<Option<ContinuationCookieClaim>>;

    async fn revoke_continuation_cookie(&self, cookie_hash: &str) -> SkillsStoreResult<bool>;

    async fn revoke_continuation_cookies_for_exec(&self, exec_id: &str)
        -> SkillsStoreResult<usize>;
}
