use std::{fmt, sync::Arc};

use smg_blob_storage::BlobStore;

use crate::{
    memory::InMemorySkillStore,
    storage::{BundleTokenStore, ContinuationCookieStore, SkillMetadataStore, TenantAliasStore},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SkillServiceMode {
    #[default]
    Placeholder,
    SingleProcess,
}

struct SkillServiceInner {
    mode: SkillServiceMode,
    metadata_store: Option<Arc<dyn SkillMetadataStore>>,
    tenant_alias_store: Option<Arc<dyn TenantAliasStore>>,
    bundle_token_store: Option<Arc<dyn BundleTokenStore>>,
    continuation_cookie_store: Option<Arc<dyn ContinuationCookieStore>>,
    blob_store: Option<Arc<dyn BlobStore>>,
}

impl fmt::Debug for SkillServiceInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SkillServiceInner")
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

/// Skills service boundary used by the gateway. This PR introduces a concrete
/// single-process mode backed by in-memory metadata stores and a generic blob
/// store.
#[derive(Debug, Clone)]
pub struct SkillService {
    inner: Arc<SkillServiceInner>,
}

impl Default for SkillService {
    fn default() -> Self {
        Self::placeholder()
    }
}

impl SkillService {
    pub fn placeholder() -> Self {
        Self {
            inner: Arc::new(SkillServiceInner {
                mode: SkillServiceMode::Placeholder,
                metadata_store: None,
                tenant_alias_store: None,
                bundle_token_store: None,
                continuation_cookie_store: None,
                blob_store: None,
            }),
        }
    }

    pub fn single_process(
        metadata_store: Arc<dyn SkillMetadataStore>,
        tenant_alias_store: Arc<dyn TenantAliasStore>,
        bundle_token_store: Arc<dyn BundleTokenStore>,
        continuation_cookie_store: Arc<dyn ContinuationCookieStore>,
        blob_store: Arc<dyn BlobStore>,
    ) -> Self {
        Self {
            inner: Arc::new(SkillServiceInner {
                mode: SkillServiceMode::SingleProcess,
                metadata_store: Some(metadata_store),
                tenant_alias_store: Some(tenant_alias_store),
                bundle_token_store: Some(bundle_token_store),
                continuation_cookie_store: Some(continuation_cookie_store),
                blob_store: Some(blob_store),
            }),
        }
    }

    pub fn in_memory(blob_store: Arc<dyn BlobStore>) -> Self {
        let store = Arc::new(InMemorySkillStore::default());
        Self::single_process(
            store.clone(),
            store.clone(),
            store.clone(),
            store,
            blob_store,
        )
    }

    pub fn mode(&self) -> SkillServiceMode {
        self.inner.mode
    }

    pub fn metadata_store(&self) -> Option<Arc<dyn SkillMetadataStore>> {
        self.inner.metadata_store.clone()
    }

    pub fn tenant_alias_store(&self) -> Option<Arc<dyn TenantAliasStore>> {
        self.inner.tenant_alias_store.clone()
    }

    pub fn bundle_token_store(&self) -> Option<Arc<dyn BundleTokenStore>> {
        self.inner.bundle_token_store.clone()
    }

    pub fn continuation_cookie_store(&self) -> Option<Arc<dyn ContinuationCookieStore>> {
        self.inner.continuation_cookie_store.clone()
    }

    pub fn blob_store(&self) -> Option<Arc<dyn BlobStore>> {
        self.inner.blob_store.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::{anyhow, Result};
    use smg_blob_storage::FilesystemBlobStore;
    use tempfile::TempDir;

    use super::{SkillService, SkillServiceMode};

    #[test]
    fn placeholder_service_reports_placeholder_mode() {
        let service = SkillService::placeholder();
        assert_eq!(service.mode(), SkillServiceMode::Placeholder);
        assert!(service.metadata_store().is_none());
    }

    #[test]
    fn single_process_service_exposes_all_stores() -> Result<()> {
        let root = TempDir::new()?;
        let blob_store = Arc::new(FilesystemBlobStore::new(root.path())?);
        let service = SkillService::in_memory(blob_store.clone());

        assert_eq!(service.mode(), SkillServiceMode::SingleProcess);
        service
            .metadata_store()
            .ok_or_else(|| anyhow!("metadata store missing"))?;
        service
            .tenant_alias_store()
            .ok_or_else(|| anyhow!("tenant alias store missing"))?;
        service
            .bundle_token_store()
            .ok_or_else(|| anyhow!("bundle token store missing"))?;
        service
            .continuation_cookie_store()
            .ok_or_else(|| anyhow!("continuation cookie store missing"))?;
        service
            .blob_store()
            .ok_or_else(|| anyhow!("blob store missing"))?;
        Ok(())
    }
}
