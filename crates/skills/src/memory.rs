use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Bound::Included,
};

use async_trait::async_trait;
use parking_lot::RwLock;

use crate::{
    storage::{
        BundleTokenStore, ContinuationCookieStore, SkillMetadataStore, SkillsStoreResult,
        TenantAliasStore,
    },
    types::{
        BundleTokenClaim, ContinuationCookieClaim, SkillRecord, SkillVersionRecord,
        TenantAliasRecord,
    },
};

#[derive(Debug, Default)]
struct InMemorySkillState {
    skills: BTreeMap<String, SkillRecord>,
    skill_ids_by_tenant: BTreeMap<String, BTreeSet<String>>,
    skill_versions: BTreeMap<(String, String), SkillVersionRecord>,
    skill_versions_by_skill: BTreeMap<String, BTreeSet<String>>,
    tenant_aliases: BTreeMap<String, TenantAliasRecord>,
    bundle_tokens: BTreeMap<String, BundleTokenClaim>,
    bundle_tokens_by_exec: BTreeMap<String, BTreeSet<String>>,
    continuation_cookies: BTreeMap<String, ContinuationCookieClaim>,
    continuation_cookies_by_exec: BTreeMap<String, BTreeSet<String>>,
}

/// Single-process in-memory skills persistence used for local development and
/// early end-to-end testing.
#[derive(Debug, Default)]
pub struct InMemorySkillStore {
    state: RwLock<InMemorySkillState>,
}

#[async_trait]
impl SkillMetadataStore for InMemorySkillStore {
    async fn put_skill(&self, record: SkillRecord) -> SkillsStoreResult<()> {
        let mut state = self.state.write();
        if let Some(existing) = state.skills.get(&record.skill_id) {
            if existing.tenant_id != record.tenant_id {
                return Err(crate::storage::SkillsStoreError::InvalidData(format!(
                    "skill_id '{}' already belongs to tenant '{}'",
                    record.skill_id, existing.tenant_id
                )));
            }
        }

        state
            .skill_ids_by_tenant
            .entry(record.tenant_id.clone())
            .or_default()
            .insert(record.skill_id.clone());
        state.skills.insert(record.skill_id.clone(), record);
        Ok(())
    }

    async fn get_skill(
        &self,
        tenant_id: &str,
        skill_id: &str,
    ) -> SkillsStoreResult<Option<SkillRecord>> {
        Ok(self
            .state
            .read()
            .skills
            .get(skill_id)
            .and_then(|record| (record.tenant_id == tenant_id).then(|| record.clone())))
    }

    async fn list_skills(&self, tenant_id: &str) -> SkillsStoreResult<Vec<SkillRecord>> {
        let state = self.state.read();
        let mut skills = state
            .skill_ids_by_tenant
            .get(tenant_id)
            .into_iter()
            .flat_map(|skill_ids| skill_ids.iter())
            .filter_map(|skill_id| state.skills.get(skill_id))
            .cloned()
            .collect::<Vec<_>>();
        skills.sort_by(|left, right| {
            left.name
                .cmp(&right.name)
                .then(left.skill_id.cmp(&right.skill_id))
        });
        Ok(skills)
    }

    async fn delete_skill(&self, tenant_id: &str, skill_id: &str) -> SkillsStoreResult<bool> {
        let mut state = self.state.write();
        let Some(record) = state.skills.get(skill_id) else {
            return Ok(false);
        };
        if record.tenant_id != tenant_id {
            return Ok(false);
        }

        state.skills.remove(skill_id);
        let tenant_has_no_skills =
            if let Some(skill_ids) = state.skill_ids_by_tenant.get_mut(tenant_id) {
                skill_ids.remove(skill_id);
                skill_ids.is_empty()
            } else {
                false
            };
        if tenant_has_no_skills {
            state.skill_ids_by_tenant.remove(tenant_id);
        }
        if let Some(versions) = state.skill_versions_by_skill.remove(skill_id) {
            for version in versions {
                state
                    .skill_versions
                    .remove(&(skill_id.to_string(), version));
            }
        }

        Ok(true)
    }

    async fn put_skill_version(&self, record: SkillVersionRecord) -> SkillsStoreResult<()> {
        let key = (record.skill_id.clone(), record.version.clone());
        let mut state = self.state.write();
        if !state.skills.contains_key(&record.skill_id) {
            return Err(crate::storage::SkillsStoreError::InvalidData(format!(
                "skill_id '{}' must exist before inserting a version",
                record.skill_id
            )));
        }
        state
            .skill_versions_by_skill
            .entry(record.skill_id.clone())
            .or_default()
            .insert(record.version.clone());
        state.skill_versions.insert(key, record);
        Ok(())
    }

    async fn get_skill_version(
        &self,
        skill_id: &str,
        version: &str,
    ) -> SkillsStoreResult<Option<SkillVersionRecord>> {
        Ok(self
            .state
            .read()
            .skill_versions
            .get(&(skill_id.to_string(), version.to_string()))
            .cloned())
    }

    async fn list_skill_versions(
        &self,
        skill_id: &str,
    ) -> SkillsStoreResult<Vec<SkillVersionRecord>> {
        let start = (skill_id.to_string(), String::new());
        let end = (skill_id.to_string(), max_sort_key());
        let mut versions = self
            .state
            .read()
            .skill_versions
            .range((Included(start), Included(end)))
            .map(|(_, record)| record.clone())
            .collect::<Vec<_>>();
        versions.sort_by(|left, right| {
            left.version_number
                .cmp(&right.version_number)
                .then(left.version.cmp(&right.version))
        });
        Ok(versions)
    }

    async fn delete_skill_version(&self, skill_id: &str, version: &str) -> SkillsStoreResult<bool> {
        let mut state = self.state.write();
        let removed = state
            .skill_versions
            .remove(&(skill_id.to_string(), version.to_string()))
            .is_some();
        if !removed {
            return Ok(false);
        }

        if let Some(versions) = state.skill_versions_by_skill.get_mut(skill_id) {
            versions.remove(version);
            if versions.is_empty() {
                state.skill_versions_by_skill.remove(skill_id);
            }
        }

        Ok(true)
    }
}

fn max_sort_key() -> String {
    char::MAX.to_string()
}

#[async_trait]
impl TenantAliasStore for InMemorySkillStore {
    async fn put_tenant_alias(&self, record: TenantAliasRecord) -> SkillsStoreResult<()> {
        self.state
            .write()
            .tenant_aliases
            .insert(record.alias_tenant_id.clone(), record);
        Ok(())
    }

    async fn get_tenant_alias(
        &self,
        alias_tenant_id: &str,
    ) -> SkillsStoreResult<Option<TenantAliasRecord>> {
        Ok(self
            .state
            .read()
            .tenant_aliases
            .get(alias_tenant_id)
            .cloned())
    }

    async fn delete_tenant_alias(&self, alias_tenant_id: &str) -> SkillsStoreResult<bool> {
        Ok(self
            .state
            .write()
            .tenant_aliases
            .remove(alias_tenant_id)
            .is_some())
    }
}

#[async_trait]
impl BundleTokenStore for InMemorySkillStore {
    async fn put_bundle_token(&self, claim: BundleTokenClaim) -> SkillsStoreResult<()> {
        let mut state = self.state.write();
        let token_hash = claim.token_hash.clone();
        let exec_id = claim.exec_id.clone();
        if let Some(previous) = state.bundle_tokens.insert(token_hash.clone(), claim) {
            if let Some(tokens) = state.bundle_tokens_by_exec.get_mut(&previous.exec_id) {
                tokens.remove(&token_hash);
                if tokens.is_empty() {
                    state.bundle_tokens_by_exec.remove(&previous.exec_id);
                }
            }
        }
        state
            .bundle_tokens_by_exec
            .entry(exec_id)
            .or_default()
            .insert(token_hash);
        Ok(())
    }

    async fn get_bundle_token(
        &self,
        token_hash: &str,
    ) -> SkillsStoreResult<Option<BundleTokenClaim>> {
        Ok(self.state.read().bundle_tokens.get(token_hash).cloned())
    }

    async fn revoke_bundle_token(&self, token_hash: &str) -> SkillsStoreResult<bool> {
        let mut state = self.state.write();
        let Some(claim) = state.bundle_tokens.remove(token_hash) else {
            return Ok(false);
        };
        if let Some(tokens) = state.bundle_tokens_by_exec.get_mut(&claim.exec_id) {
            tokens.remove(token_hash);
            if tokens.is_empty() {
                state.bundle_tokens_by_exec.remove(&claim.exec_id);
            }
        }
        Ok(true)
    }

    async fn revoke_bundle_tokens_for_exec(&self, exec_id: &str) -> SkillsStoreResult<usize> {
        let mut state = self.state.write();
        let Some(tokens) = state.bundle_tokens_by_exec.remove(exec_id) else {
            return Ok(0);
        };
        let removed = tokens.len();
        for token_hash in tokens {
            state.bundle_tokens.remove(&token_hash);
        }
        Ok(removed)
    }
}

#[async_trait]
impl ContinuationCookieStore for InMemorySkillStore {
    async fn put_continuation_cookie(
        &self,
        claim: ContinuationCookieClaim,
    ) -> SkillsStoreResult<()> {
        let mut state = self.state.write();
        let cookie_hash = claim.cookie_hash.clone();
        let exec_id = claim.exec_id.clone();
        if let Some(previous) = state
            .continuation_cookies
            .insert(cookie_hash.clone(), claim)
        {
            if let Some(cookies) = state
                .continuation_cookies_by_exec
                .get_mut(&previous.exec_id)
            {
                cookies.remove(&cookie_hash);
                if cookies.is_empty() {
                    state.continuation_cookies_by_exec.remove(&previous.exec_id);
                }
            }
        }
        state
            .continuation_cookies_by_exec
            .entry(exec_id)
            .or_default()
            .insert(cookie_hash);
        Ok(())
    }

    async fn get_continuation_cookie(
        &self,
        cookie_hash: &str,
    ) -> SkillsStoreResult<Option<ContinuationCookieClaim>> {
        Ok(self
            .state
            .read()
            .continuation_cookies
            .get(cookie_hash)
            .cloned())
    }

    async fn revoke_continuation_cookie(&self, cookie_hash: &str) -> SkillsStoreResult<bool> {
        let mut state = self.state.write();
        let Some(claim) = state.continuation_cookies.remove(cookie_hash) else {
            return Ok(false);
        };
        if let Some(cookies) = state.continuation_cookies_by_exec.get_mut(&claim.exec_id) {
            cookies.remove(cookie_hash);
            if cookies.is_empty() {
                state.continuation_cookies_by_exec.remove(&claim.exec_id);
            }
        }
        Ok(true)
    }

    async fn revoke_continuation_cookies_for_exec(
        &self,
        exec_id: &str,
    ) -> SkillsStoreResult<usize> {
        let mut state = self.state.write();
        let Some(cookies) = state.continuation_cookies_by_exec.remove(exec_id) else {
            return Ok(0);
        };
        let removed = cookies.len();
        for cookie_hash in cookies {
            state.continuation_cookies.remove(&cookie_hash);
        }
        Ok(removed)
    }
}
