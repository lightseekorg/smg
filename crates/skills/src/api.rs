use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    io::{Cursor, Write},
    sync::Arc,
};

use chrono::Utc;
use smg_blob_storage::{BlobKey, BlobStore, BlobStoreError, PutBlobRequest};
use ulid::Ulid;
use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};

use crate::{
    config::SkillUploadLimits,
    memory::InMemorySkillStore,
    storage::{
        BundleTokenStore, ContinuationCookieStore, SkillMetadataStore, SkillsStoreError,
        TenantAliasStore,
    },
    types::{
        NormalizedSkillBundle, ParsedSkillBundle, SkillFileRecord, SkillParseWarning, SkillRecord,
        SkillVersionRecord,
    },
    validation::{
        is_code_file_path, normalize_skill_bundle_zip_with_limits, parse_skill_bundle,
        SkillBundleArchiveError, SkillParseError,
    },
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
    upload_limits: SkillUploadLimits,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UploadedSkillFile {
    pub relative_path: String,
    pub contents: Vec<u8>,
    pub media_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillUpload {
    Zip(Vec<u8>),
    Files(Vec<UploadedSkillFile>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateSkillRequest {
    pub tenant_id: String,
    pub upload: SkillUpload,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateSkillVersionRequest {
    pub tenant_id: String,
    pub skill_id: String,
    pub upload: SkillUpload,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateSkillRequest {
    pub tenant_id: String,
    pub skill_id: String,
    pub default_version_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateSkillVersionRequest {
    pub tenant_id: String,
    pub skill_id: String,
    pub version_ref: String,
    pub deprecated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillCreateResult {
    pub skill: SkillRecord,
    pub version: SkillVersionRecord,
    pub warnings: Vec<SkillParseWarning>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeletedSkillVersionResult {
    pub deleted_skill: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum SkillServiceError {
    #[error("skills service is not available for {component}")]
    MissingComponent { component: &'static str },

    #[error("target tenant id is required")]
    MissingTenantId,

    #[error("skill id is required")]
    MissingSkillId,

    #[error("skill '{skill_id}' was not found for tenant '{tenant_id}'")]
    SkillNotFound { tenant_id: String, skill_id: String },

    #[error(
        "skill version '{version}' was not found for skill '{skill_id}' in tenant '{tenant_id}'"
    )]
    SkillVersionNotFound {
        tenant_id: String,
        skill_id: String,
        version: String,
    },

    #[error("multipart upload must contain either one zip archive or one or more files[] parts")]
    MissingUploadParts,

    #[error("multipart upload cannot mix zip archive and files[] parts")]
    MixedUploadModes,

    #[error("zip archive uploads must use a .zip filename or application/zip content type")]
    InvalidZipUpload,

    #[error("multipart file name is required for files[] uploads")]
    MissingFileName,

    #[error("SKILL.md must be valid UTF-8")]
    SkillMdNotUtf8,

    #[error(transparent)]
    BundleArchive(#[from] SkillBundleArchiveError),

    #[error(transparent)]
    BundleParse(#[from] SkillParseError),

    #[error(transparent)]
    BlobStore(#[from] BlobStoreError),

    #[error(transparent)]
    Store(#[from] SkillsStoreError),

    #[error("failed to build skill bundle: {0}")]
    BundleBuild(String),

    #[error("skill '{skill_id}' in tenant '{tenant_id}' has no default version")]
    MissingDefaultVersion { tenant_id: String, skill_id: String },

    #[error(
        "cannot delete default version '{version}' for skill '{skill_id}' in tenant '{tenant_id}' while other versions remain"
    )]
    CannotDeleteDefaultVersion {
        tenant_id: String,
        skill_id: String,
        version: String,
    },
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
                upload_limits: SkillUploadLimits::default(),
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
        Self::single_process_with_limits(
            metadata_store,
            tenant_alias_store,
            bundle_token_store,
            continuation_cookie_store,
            blob_store,
            SkillUploadLimits::default(),
        )
    }

    pub fn single_process_with_limits(
        metadata_store: Arc<dyn SkillMetadataStore>,
        tenant_alias_store: Arc<dyn TenantAliasStore>,
        bundle_token_store: Arc<dyn BundleTokenStore>,
        continuation_cookie_store: Arc<dyn ContinuationCookieStore>,
        blob_store: Arc<dyn BlobStore>,
        upload_limits: SkillUploadLimits,
    ) -> Self {
        Self {
            inner: Arc::new(SkillServiceInner {
                mode: SkillServiceMode::SingleProcess,
                metadata_store: Some(metadata_store),
                tenant_alias_store: Some(tenant_alias_store),
                bundle_token_store: Some(bundle_token_store),
                continuation_cookie_store: Some(continuation_cookie_store),
                blob_store: Some(blob_store),
                upload_limits,
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

    pub fn in_memory_with_limits(
        blob_store: Arc<dyn BlobStore>,
        upload_limits: SkillUploadLimits,
    ) -> Self {
        let store = Arc::new(InMemorySkillStore::default());
        Self::single_process_with_limits(
            store.clone(),
            store.clone(),
            store.clone(),
            store,
            blob_store,
            upload_limits,
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

    pub fn upload_limits(&self) -> SkillUploadLimits {
        self.inner.upload_limits
    }

    pub async fn get_skill(
        &self,
        tenant_id: &str,
        skill_id: &str,
    ) -> Result<SkillRecord, SkillServiceError> {
        let tenant_id =
            normalize_required_value(tenant_id.to_string(), SkillServiceError::MissingTenantId)?;
        let skill_id =
            normalize_required_value(skill_id.to_string(), SkillServiceError::MissingSkillId)?;
        let metadata_store = self
            .metadata_store()
            .ok_or(SkillServiceError::MissingComponent {
                component: "metadata_store",
            })?;

        metadata_store
            .get_skill(&tenant_id, &skill_id)
            .await?
            .ok_or(SkillServiceError::SkillNotFound {
                tenant_id,
                skill_id,
            })
    }

    pub async fn list_skills(
        &self,
        tenant_id: &str,
        source: Option<&str>,
        name: Option<&str>,
    ) -> Result<Vec<SkillRecord>, SkillServiceError> {
        let tenant_id =
            normalize_required_value(tenant_id.to_string(), SkillServiceError::MissingTenantId)?;
        let metadata_store = self
            .metadata_store()
            .ok_or(SkillServiceError::MissingComponent {
                component: "metadata_store",
            })?;

        let source = source.map(str::trim).filter(|value| !value.is_empty());
        let name = name.map(str::trim).filter(|value| !value.is_empty());

        let mut records = metadata_store.list_skills(&tenant_id).await?;
        records.retain(|record| {
            source.is_none_or(|value| record.source == value)
                && name.is_none_or(|value| record.name == value)
        });
        Ok(records)
    }

    pub async fn get_skill_version(
        &self,
        tenant_id: &str,
        skill_id: &str,
        version_ref: &str,
    ) -> Result<SkillVersionRecord, SkillServiceError> {
        let tenant_id =
            normalize_required_value(tenant_id.to_string(), SkillServiceError::MissingTenantId)?;
        let skill_id =
            normalize_required_value(skill_id.to_string(), SkillServiceError::MissingSkillId)?;
        let version_ref = normalize_required_value(
            version_ref.to_string(),
            SkillServiceError::SkillVersionNotFound {
                tenant_id: tenant_id.clone(),
                skill_id: skill_id.clone(),
                version: version_ref.trim().to_string(),
            },
        )?;
        let metadata_store = self
            .metadata_store()
            .ok_or(SkillServiceError::MissingComponent {
                component: "metadata_store",
            })?;
        resolve_skill_version_record(&*metadata_store, &tenant_id, &skill_id, &version_ref).await
    }

    pub async fn list_skill_versions(
        &self,
        tenant_id: &str,
        skill_id: &str,
        include_deprecated: bool,
    ) -> Result<Vec<SkillVersionRecord>, SkillServiceError> {
        let tenant_id =
            normalize_required_value(tenant_id.to_string(), SkillServiceError::MissingTenantId)?;
        let skill_id =
            normalize_required_value(skill_id.to_string(), SkillServiceError::MissingSkillId)?;
        let metadata_store = self
            .metadata_store()
            .ok_or(SkillServiceError::MissingComponent {
                component: "metadata_store",
            })?;

        metadata_store
            .get_skill(&tenant_id, &skill_id)
            .await?
            .ok_or_else(|| SkillServiceError::SkillNotFound {
                tenant_id: tenant_id.clone(),
                skill_id: skill_id.clone(),
            })?;

        let mut versions = metadata_store.list_skill_versions(&skill_id).await?;
        if !include_deprecated {
            versions.retain(|record| !record.deprecated);
        }
        Ok(versions)
    }

    pub async fn create_skill(
        &self,
        request: CreateSkillRequest,
    ) -> Result<SkillCreateResult, SkillServiceError> {
        let tenant_id =
            normalize_required_value(request.tenant_id, SkillServiceError::MissingTenantId)?;
        let metadata_store = self
            .metadata_store()
            .ok_or(SkillServiceError::MissingComponent {
                component: "metadata_store",
            })?;
        let blob_store = self
            .blob_store()
            .ok_or(SkillServiceError::MissingComponent {
                component: "blob_store",
            })?;

        let (bundle, media_types) = normalize_upload_bundle(request.upload, self.upload_limits())?;
        let parsed_bundle = parse_bundle_contents(&bundle)?;
        let skill_id = generate_skill_id();
        let version = generate_version_id(&[]);
        let now = Utc::now();
        let file_manifest = persist_bundle_files(
            &*blob_store,
            &tenant_id,
            &skill_id,
            &version,
            &bundle,
            &media_types,
        )
        .await?;

        let skill_record = SkillRecord {
            tenant_id: tenant_id.clone(),
            skill_id: skill_id.clone(),
            name: parsed_bundle.name.clone(),
            short_description: parsed_bundle.short_description.clone(),
            description: Some(parsed_bundle.description.clone()),
            source: "custom".to_string(),
            has_code_files: bundle.has_code_files,
            latest_version: Some(version.clone()),
            default_version: Some(version.clone()),
            created_at: now,
            updated_at: now,
        };
        let version_record = SkillVersionRecord {
            skill_id: skill_id.clone(),
            version: version.clone(),
            version_number: 1,
            name: parsed_bundle.name.clone(),
            short_description: parsed_bundle.short_description.clone(),
            description: parsed_bundle.description.clone(),
            interface: parsed_bundle.interface.clone(),
            dependencies: parsed_bundle.dependencies.clone(),
            policy: parsed_bundle.policy.clone(),
            deprecated: false,
            file_manifest,
            instruction_token_counts: BTreeMap::new(),
            created_at: now,
        };

        if let Err(error) = metadata_store
            .put_skill_with_initial_version(skill_record.clone(), version_record.clone())
            .await
        {
            cleanup_blobs(&*blob_store, &version_record.file_manifest).await;
            return Err(error.into());
        }

        Ok(SkillCreateResult {
            skill: skill_record,
            version: version_record,
            warnings: parsed_bundle.warnings,
        })
    }

    pub async fn create_skill_version(
        &self,
        request: CreateSkillVersionRequest,
    ) -> Result<SkillCreateResult, SkillServiceError> {
        let tenant_id =
            normalize_required_value(request.tenant_id, SkillServiceError::MissingTenantId)?;
        let skill_id =
            normalize_required_value(request.skill_id, SkillServiceError::MissingSkillId)?;
        let metadata_store = self
            .metadata_store()
            .ok_or(SkillServiceError::MissingComponent {
                component: "metadata_store",
            })?;
        let blob_store = self
            .blob_store()
            .ok_or(SkillServiceError::MissingComponent {
                component: "blob_store",
            })?;

        let existing_skill = metadata_store
            .get_skill(&tenant_id, &skill_id)
            .await?
            .ok_or_else(|| SkillServiceError::SkillNotFound {
                tenant_id: tenant_id.clone(),
                skill_id: skill_id.clone(),
            })?;
        let existing_versions = metadata_store.list_skill_versions(&skill_id).await?;
        let existing_default_projection = existing_skill
            .default_version
            .as_ref()
            .map(|default_version| {
                existing_versions
                    .iter()
                    .find(|record| record.version == *default_version)
                    .cloned()
                    .ok_or_else(|| SkillServiceError::SkillVersionNotFound {
                        tenant_id: tenant_id.clone(),
                        skill_id: skill_id.clone(),
                        version: default_version.clone(),
                    })
            })
            .transpose()?;

        let (bundle, media_types) = normalize_upload_bundle(request.upload, self.upload_limits())?;
        let parsed_bundle = parse_bundle_contents(&bundle)?;
        let version = generate_version_id(&existing_versions);
        let version_number = next_version_number(&existing_versions);
        let now = Utc::now();
        let file_manifest = persist_bundle_files(
            &*blob_store,
            &tenant_id,
            &skill_id,
            &version,
            &bundle,
            &media_types,
        )
        .await?;

        let version_record = SkillVersionRecord {
            skill_id: skill_id.clone(),
            version: version.clone(),
            version_number,
            name: parsed_bundle.name.clone(),
            short_description: parsed_bundle.short_description.clone(),
            description: parsed_bundle.description.clone(),
            interface: parsed_bundle.interface.clone(),
            dependencies: parsed_bundle.dependencies.clone(),
            policy: parsed_bundle.policy.clone(),
            deprecated: false,
            file_manifest,
            instruction_token_counts: BTreeMap::new(),
            created_at: now,
        };

        let mut updated_skill = existing_skill.clone();
        updated_skill.latest_version = Some(version.clone());
        if updated_skill.default_version.is_none() {
            updated_skill.default_version = Some(version.clone());
        }
        let default_projection =
            existing_default_projection.unwrap_or_else(|| version_record.clone());
        apply_skill_projection_from_version(&mut updated_skill, &default_projection);
        updated_skill.updated_at = now;

        if let Err(error) = metadata_store
            .put_skill_version_and_update_skill(version_record.clone(), updated_skill.clone())
            .await
        {
            cleanup_blobs(&*blob_store, &version_record.file_manifest).await;
            return Err(error.into());
        }

        Ok(SkillCreateResult {
            skill: updated_skill,
            version: version_record,
            warnings: parsed_bundle.warnings,
        })
    }

    pub async fn update_skill(
        &self,
        request: UpdateSkillRequest,
    ) -> Result<SkillRecord, SkillServiceError> {
        let tenant_id =
            normalize_required_value(request.tenant_id, SkillServiceError::MissingTenantId)?;
        let skill_id =
            normalize_required_value(request.skill_id, SkillServiceError::MissingSkillId)?;
        let default_version_ref = normalize_required_value(
            request.default_version_ref,
            SkillServiceError::SkillVersionNotFound {
                tenant_id: tenant_id.clone(),
                skill_id: skill_id.clone(),
                version: String::new(),
            },
        )?;
        let metadata_store = self
            .metadata_store()
            .ok_or(SkillServiceError::MissingComponent {
                component: "metadata_store",
            })?;

        let mut skill = metadata_store
            .get_skill(&tenant_id, &skill_id)
            .await?
            .ok_or_else(|| SkillServiceError::SkillNotFound {
                tenant_id: tenant_id.clone(),
                skill_id: skill_id.clone(),
            })?;
        let target_version = resolve_skill_version_record(
            &*metadata_store,
            &tenant_id,
            &skill_id,
            &default_version_ref,
        )
        .await?;

        skill.default_version = Some(target_version.version.clone());
        apply_skill_projection_from_version(&mut skill, &target_version);
        skill.updated_at = Utc::now();
        metadata_store.put_skill(skill.clone()).await?;
        Ok(skill)
    }

    pub async fn update_skill_version(
        &self,
        request: UpdateSkillVersionRequest,
    ) -> Result<SkillVersionRecord, SkillServiceError> {
        let tenant_id =
            normalize_required_value(request.tenant_id, SkillServiceError::MissingTenantId)?;
        let skill_id =
            normalize_required_value(request.skill_id, SkillServiceError::MissingSkillId)?;
        let version_ref = normalize_required_value(
            request.version_ref,
            SkillServiceError::SkillVersionNotFound {
                tenant_id: tenant_id.clone(),
                skill_id: skill_id.clone(),
                version: String::new(),
            },
        )?;
        let metadata_store = self
            .metadata_store()
            .ok_or(SkillServiceError::MissingComponent {
                component: "metadata_store",
            })?;

        let mut skill = metadata_store
            .get_skill(&tenant_id, &skill_id)
            .await?
            .ok_or_else(|| SkillServiceError::SkillNotFound {
                tenant_id: tenant_id.clone(),
                skill_id: skill_id.clone(),
            })?;
        let mut version =
            resolve_skill_version_record(&*metadata_store, &tenant_id, &skill_id, &version_ref)
                .await?;
        let original_version = version.clone();

        version.deprecated = request.deprecated;
        metadata_store.put_skill_version(version.clone()).await?;

        skill.updated_at = Utc::now();
        if let Err(error) = metadata_store.put_skill(skill).await {
            let _ = metadata_store.put_skill_version(original_version).await;
            return Err(error.into());
        }

        Ok(version)
    }

    pub async fn delete_skill(
        &self,
        tenant_id: &str,
        skill_id: &str,
    ) -> Result<(), SkillServiceError> {
        let tenant_id =
            normalize_required_value(tenant_id.to_string(), SkillServiceError::MissingTenantId)?;
        let skill_id =
            normalize_required_value(skill_id.to_string(), SkillServiceError::MissingSkillId)?;
        let metadata_store = self
            .metadata_store()
            .ok_or(SkillServiceError::MissingComponent {
                component: "metadata_store",
            })?;
        let blob_store = self
            .blob_store()
            .ok_or(SkillServiceError::MissingComponent {
                component: "blob_store",
            })?;

        metadata_store
            .get_skill(&tenant_id, &skill_id)
            .await?
            .ok_or_else(|| SkillServiceError::SkillNotFound {
                tenant_id: tenant_id.clone(),
                skill_id: skill_id.clone(),
            })?;

        let versions = metadata_store.list_skill_versions(&skill_id).await?;
        metadata_store.delete_skill(&tenant_id, &skill_id).await?;
        for version in &versions {
            cleanup_blobs(&*blob_store, &version.file_manifest).await;
        }
        Ok(())
    }

    pub async fn delete_skill_version(
        &self,
        tenant_id: &str,
        skill_id: &str,
        version_ref: &str,
    ) -> Result<DeletedSkillVersionResult, SkillServiceError> {
        let tenant_id =
            normalize_required_value(tenant_id.to_string(), SkillServiceError::MissingTenantId)?;
        let skill_id =
            normalize_required_value(skill_id.to_string(), SkillServiceError::MissingSkillId)?;
        let version_ref = normalize_required_value(
            version_ref.to_string(),
            SkillServiceError::SkillVersionNotFound {
                tenant_id: tenant_id.clone(),
                skill_id: skill_id.clone(),
                version: version_ref.trim().to_string(),
            },
        )?;
        let metadata_store = self
            .metadata_store()
            .ok_or(SkillServiceError::MissingComponent {
                component: "metadata_store",
            })?;
        let blob_store = self
            .blob_store()
            .ok_or(SkillServiceError::MissingComponent {
                component: "blob_store",
            })?;

        let original_skill = metadata_store
            .get_skill(&tenant_id, &skill_id)
            .await?
            .ok_or_else(|| SkillServiceError::SkillNotFound {
                tenant_id: tenant_id.clone(),
                skill_id: skill_id.clone(),
            })?;
        let mut updated_skill = original_skill.clone();
        let version =
            resolve_skill_version_record(&*metadata_store, &tenant_id, &skill_id, &version_ref)
                .await?;
        let versions = metadata_store.list_skill_versions(&skill_id).await?;

        if versions.len() > 1
            && original_skill
                .default_version
                .as_deref()
                .is_some_and(|default_version| default_version == version.version)
        {
            return Err(SkillServiceError::CannotDeleteDefaultVersion {
                tenant_id,
                skill_id,
                version: version.version,
            });
        }

        if versions.len() == 1 {
            metadata_store
                .delete_skill(&original_skill.tenant_id, &original_skill.skill_id)
                .await?;
            cleanup_blobs(&*blob_store, &version.file_manifest).await;
            return Ok(DeletedSkillVersionResult {
                deleted_skill: true,
            });
        }

        let remaining_versions = versions
            .into_iter()
            .filter(|record| record.version != version.version)
            .collect::<Vec<_>>();
        let latest_version = latest_skill_version(&remaining_versions).ok_or_else(|| {
            SkillServiceError::SkillVersionNotFound {
                tenant_id: original_skill.tenant_id.clone(),
                skill_id: original_skill.skill_id.clone(),
                version: "latest".to_string(),
            }
        })?;
        updated_skill.latest_version = Some(latest_version.version.clone());
        let default_version = updated_skill.default_version.clone().ok_or_else(|| {
            SkillServiceError::MissingDefaultVersion {
                tenant_id: original_skill.tenant_id.clone(),
                skill_id: original_skill.skill_id.clone(),
            }
        })?;
        let default_projection = remaining_versions
            .iter()
            .find(|record| record.version == default_version)
            .cloned()
            .ok_or_else(|| SkillServiceError::SkillVersionNotFound {
                tenant_id: original_skill.tenant_id.clone(),
                skill_id: original_skill.skill_id.clone(),
                version: default_version,
            })?;
        apply_skill_projection_from_version(&mut updated_skill, &default_projection);
        updated_skill.updated_at = Utc::now();
        metadata_store.put_skill(updated_skill).await?;
        if let Err(error) = metadata_store
            .delete_skill_version(&original_skill.skill_id, &version.version)
            .await
        {
            let _ = metadata_store.put_skill(original_skill).await;
            return Err(error.into());
        }
        cleanup_blobs(&*blob_store, &version.file_manifest).await;

        Ok(DeletedSkillVersionResult {
            deleted_skill: false,
        })
    }
}

fn normalize_required_value(
    value: String,
    error: SkillServiceError,
) -> Result<String, SkillServiceError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(error);
    }
    Ok(value.to_string())
}

fn normalize_upload_bundle(
    upload: SkillUpload,
    limits: SkillUploadLimits,
) -> Result<(NormalizedSkillBundle, HashMap<String, Option<String>>), SkillServiceError> {
    match upload {
        SkillUpload::Zip(bytes) => {
            if bytes.len() > limits.max_upload_size_bytes {
                return Err(SkillBundleArchiveError::BundleTooLarge {
                    max_bytes: limits.max_upload_size_bytes as u64,
                }
                .into());
            }
            let bundle = normalize_skill_bundle_zip_with_limits(&bytes, limits)?;
            Ok((bundle, HashMap::new()))
        }
        SkillUpload::Files(files) => {
            if files.is_empty() {
                return Err(SkillServiceError::MissingUploadParts);
            }
            validate_files_upload_limits(&files, limits)?;
            let media_types = files
                .iter()
                .map(|file| (file.relative_path.clone(), file.media_type.clone()))
                .collect::<HashMap<_, _>>();
            let bundle = normalize_files_upload(&files, limits)?;
            Ok((bundle, media_types))
        }
    }
}

fn normalize_files_upload(
    files: &[UploadedSkillFile],
    limits: SkillUploadLimits,
) -> Result<NormalizedSkillBundle, SkillServiceError> {
    let mut buffer = Cursor::new(Vec::new());
    {
        let mut zip = ZipWriter::new(&mut buffer);
        let options = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Stored)
            .unix_permissions(0o644);
        for file in files {
            let entry_name = format!("bundle/{}", file.relative_path);
            zip.start_file(entry_name, options)
                .map_err(|error| SkillServiceError::BundleBuild(error.to_string()))?;
            zip.write_all(&file.contents)
                .map_err(|error| SkillServiceError::BundleBuild(error.to_string()))?;
        }
        zip.finish()
            .map_err(|error| SkillServiceError::BundleBuild(error.to_string()))?;
    }

    normalize_skill_bundle_zip_with_limits(buffer.get_ref(), limits).map_err(Into::into)
}

fn validate_files_upload_limits(
    files: &[UploadedSkillFile],
    limits: SkillUploadLimits,
) -> Result<(), SkillServiceError> {
    if files.len() > limits.max_files_per_version {
        return Err(SkillBundleArchiveError::TooManyFiles {
            max_files: limits.max_files_per_version,
        }
        .into());
    }

    let mut total_size_bytes = 0usize;
    for file in files {
        if file.contents.len() > limits.max_file_size_bytes {
            return Err(SkillBundleArchiveError::EntryTooLarge {
                path: file.relative_path.clone(),
                max_bytes: limits.max_file_size_bytes as u64,
            }
            .into());
        }
        total_size_bytes = total_size_bytes.checked_add(file.contents.len()).ok_or(
            SkillBundleArchiveError::BundleTooLarge {
                max_bytes: limits.max_upload_size_bytes as u64,
            },
        )?;
        if total_size_bytes > limits.max_upload_size_bytes {
            return Err(SkillBundleArchiveError::BundleTooLarge {
                max_bytes: limits.max_upload_size_bytes as u64,
            }
            .into());
        }
    }

    Ok(())
}

fn parse_bundle_contents(
    bundle: &NormalizedSkillBundle,
) -> Result<ParsedSkillBundle, SkillServiceError> {
    let skill_md_bytes = bundle
        .files
        .iter()
        .find(|file| file.relative_path == bundle.skill_md_path)
        .ok_or_else(|| {
            SkillServiceError::BundleBuild("normalized bundle missing SKILL.md".to_string())
        })?;
    let skill_md = std::str::from_utf8(&skill_md_bytes.contents)
        .map_err(|_| SkillServiceError::SkillMdNotUtf8)?;
    let openai_yaml = bundle
        .openai_sidecar_path
        .as_ref()
        .and_then(|path| bundle.files.iter().find(|file| &file.relative_path == path))
        .and_then(|file| std::str::from_utf8(&file.contents).ok());

    parse_skill_bundle(skill_md, openai_yaml).map_err(Into::into)
}

async fn persist_bundle_files(
    blob_store: &dyn BlobStore,
    tenant_id: &str,
    skill_id: &str,
    version: &str,
    bundle: &NormalizedSkillBundle,
    media_types: &HashMap<String, Option<String>>,
) -> Result<Vec<SkillFileRecord>, SkillServiceError> {
    let mut uploaded_keys = Vec::with_capacity(bundle.files.len());
    let mut manifest = Vec::with_capacity(bundle.files.len());

    for file in &bundle.files {
        let blob_key = BlobKey(format!(
            "tenants/{tenant_id}/skills/{skill_id}/{version}/{}",
            file.relative_path
        ));
        let put_request = PutBlobRequest {
            reader: Box::pin(Cursor::new(file.contents.clone())),
            content_length: file.contents.len() as u64,
            content_type: media_types.get(&file.relative_path).cloned().flatten(),
        };

        if let Err(error) = blob_store.put_stream(&blob_key, put_request).await {
            cleanup_uploaded_keys(blob_store, &uploaded_keys).await;
            return Err(error.into());
        }

        uploaded_keys.push(blob_key.clone());
        manifest.push(SkillFileRecord {
            relative_path: file.relative_path.clone(),
            media_type: media_types.get(&file.relative_path).cloned().flatten(),
            size_bytes: file.size_bytes(),
            blob_key: Some(blob_key),
        });
    }

    Ok(manifest)
}

async fn cleanup_blobs(blob_store: &dyn BlobStore, manifest: &[SkillFileRecord]) {
    for entry in manifest {
        if let Some(blob_key) = &entry.blob_key {
            let _ = blob_store.delete(blob_key).await;
        }
    }
}

async fn cleanup_uploaded_keys(blob_store: &dyn BlobStore, uploaded_keys: &[BlobKey]) {
    for blob_key in uploaded_keys {
        let _ = blob_store.delete(blob_key).await;
    }
}

fn generate_skill_id() -> String {
    format!("skill_{}", Ulid::new().to_string().to_ascii_lowercase())
}

fn generate_version_id(existing_versions: &[SkillVersionRecord]) -> String {
    let mut candidate = Utc::now().timestamp_micros();
    let used_versions = existing_versions
        .iter()
        .map(|record| record.version.clone())
        .collect::<std::collections::HashSet<_>>();
    while used_versions.contains(&candidate.to_string()) {
        candidate += 1;
    }
    candidate.to_string()
}

fn next_version_number(existing_versions: &[SkillVersionRecord]) -> u32 {
    existing_versions
        .iter()
        .map(|record| record.version_number)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

async fn resolve_skill_version_record(
    metadata_store: &dyn SkillMetadataStore,
    tenant_id: &str,
    skill_id: &str,
    version_ref: &str,
) -> Result<SkillVersionRecord, SkillServiceError> {
    let skill = metadata_store
        .get_skill(tenant_id, skill_id)
        .await?
        .ok_or_else(|| SkillServiceError::SkillNotFound {
            tenant_id: tenant_id.to_string(),
            skill_id: skill_id.to_string(),
        })?;

    if version_ref == "latest" {
        let latest_version =
            skill
                .latest_version
                .ok_or_else(|| SkillServiceError::SkillVersionNotFound {
                    tenant_id: tenant_id.to_string(),
                    skill_id: skill_id.to_string(),
                    version: "latest".to_string(),
                })?;
        return metadata_store
            .get_skill_version(skill_id, &latest_version)
            .await?
            .ok_or_else(|| SkillServiceError::SkillVersionNotFound {
                tenant_id: tenant_id.to_string(),
                skill_id: skill_id.to_string(),
                version: latest_version,
            });
    }

    if let Some(record) = metadata_store
        .get_skill_version(skill_id, version_ref)
        .await?
    {
        return Ok(record);
    }

    if let Ok(version_number) = version_ref.parse::<u32>() {
        let versions = metadata_store.list_skill_versions(skill_id).await?;
        if let Some(record) = versions
            .into_iter()
            .find(|record| record.version_number == version_number)
        {
            return Ok(record);
        }
    }

    Err(SkillServiceError::SkillVersionNotFound {
        tenant_id: tenant_id.to_string(),
        skill_id: skill_id.to_string(),
        version: version_ref.to_string(),
    })
}

fn apply_skill_projection_from_version(skill: &mut SkillRecord, version: &SkillVersionRecord) {
    skill.name.clone_from(&version.name);
    skill
        .short_description
        .clone_from(&version.short_description);
    skill.description = Some(version.description.clone());
    skill.has_code_files = version
        .file_manifest
        .iter()
        .any(|file| is_code_file_path(&file.relative_path));
}

fn latest_skill_version(versions: &[SkillVersionRecord]) -> Option<SkillVersionRecord> {
    versions.iter().cloned().max_by(|left, right| {
        left.version_number
            .cmp(&right.version_number)
            .then_with(|| left.version.cmp(&right.version))
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::{anyhow, Result};
    use smg_blob_storage::FilesystemBlobStore;
    use tempfile::TempDir;

    use super::{
        CreateSkillRequest, CreateSkillVersionRequest, SkillBundleArchiveError, SkillService,
        SkillServiceError, SkillServiceMode, SkillUpload, SkillUploadLimits, UpdateSkillRequest,
        UpdateSkillVersionRequest, UploadedSkillFile,
    };

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

    #[tokio::test]
    async fn create_skill_persists_initial_version_and_bundle_blobs() -> Result<()> {
        let root = TempDir::new()?;
        let blob_store = Arc::new(FilesystemBlobStore::new(root.path())?);
        let service = SkillService::in_memory(blob_store.clone());

        let result = service
            .create_skill(CreateSkillRequest {
                tenant_id: "tenant-a".to_string(),
                upload: SkillUpload::Files(vec![
                    UploadedSkillFile {
                        relative_path: "SKILL.md".to_string(),
                        contents: b"---\nname: acme:map\ndescription: Map the repo\nmetadata:\n  short-description: Map it\n---\nUse rg.".to_vec(),
                        media_type: Some("text/markdown".to_string()),
                    },
                    UploadedSkillFile {
                        relative_path: "scripts/run.py".to_string(),
                        contents: b"print('hi')".to_vec(),
                        media_type: Some("text/x-python".to_string()),
                    },
                ]),
            })
            .await?;

        assert_eq!(result.skill.tenant_id, "tenant-a");
        assert_eq!(result.version.version_number, 1);
        assert_eq!(result.version.file_manifest.len(), 2);
        assert!(result
            .version
            .file_manifest
            .iter()
            .all(|entry| entry.blob_key.is_some()));
        assert!(result.skill.has_code_files);

        let metadata_store = service
            .metadata_store()
            .ok_or_else(|| anyhow!("metadata store missing"))?;
        assert!(metadata_store
            .get_skill("tenant-a", &result.skill.skill_id)
            .await?
            .is_some());
        assert!(metadata_store
            .get_skill_version(&result.skill.skill_id, &result.version.version)
            .await?
            .is_some());

        Ok(())
    }

    #[tokio::test]
    async fn create_skill_enforces_configured_file_count_limit() -> Result<()> {
        let root = TempDir::new()?;
        let blob_store = Arc::new(FilesystemBlobStore::new(root.path())?);
        let service = SkillService::in_memory_with_limits(
            blob_store,
            SkillUploadLimits {
                max_upload_size_bytes: 1024,
                max_files_per_version: 1,
                max_file_size_bytes: 1024,
            },
        );

        let error = service
            .create_skill(CreateSkillRequest {
                tenant_id: "tenant-a".to_string(),
                upload: SkillUpload::Files(vec![
                    UploadedSkillFile {
                        relative_path: "SKILL.md".to_string(),
                        contents: b"---\nname: acme:map\ndescription: Map the repo\n---\nUse rg."
                            .to_vec(),
                        media_type: Some("text/markdown".to_string()),
                    },
                    UploadedSkillFile {
                        relative_path: "notes.txt".to_string(),
                        contents: b"notes".to_vec(),
                        media_type: Some("text/plain".to_string()),
                    },
                ]),
            })
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            SkillServiceError::BundleArchive(SkillBundleArchiveError::TooManyFiles {
                max_files: 1
            })
        ));
        Ok(())
    }

    #[tokio::test]
    async fn create_skill_version_increments_monotonic_version_number() -> Result<()> {
        let root = TempDir::new()?;
        let blob_store = Arc::new(FilesystemBlobStore::new(root.path())?);
        let service = SkillService::in_memory(blob_store);

        let created = service
            .create_skill(CreateSkillRequest {
                tenant_id: "tenant-a".to_string(),
                upload: SkillUpload::Files(vec![UploadedSkillFile {
                    relative_path: "SKILL.md".to_string(),
                    contents: b"---\nname: acme:map\ndescription: Map the repo\n---\nUse rg."
                        .to_vec(),
                    media_type: Some("text/markdown".to_string()),
                }]),
            })
            .await?;

        let next = service
            .create_skill_version(CreateSkillVersionRequest {
                tenant_id: "tenant-a".to_string(),
                skill_id: created.skill.skill_id.clone(),
                upload: SkillUpload::Files(vec![
                    UploadedSkillFile {
                        relative_path: "SKILL.md".to_string(),
                        contents: b"---\nname: acme:map\ndescription: Map the repo better\n---\nUse rg and fd.".to_vec(),
                        media_type: Some("text/markdown".to_string()),
                    },
                    UploadedSkillFile {
                        relative_path: "scripts/run.py".to_string(),
                        contents: b"print('v2')".to_vec(),
                        media_type: Some("text/x-python".to_string()),
                    },
                ]),
            })
            .await?;

        assert_eq!(next.version.version_number, 2);
        assert_ne!(next.version.version, created.version.version);
        assert_eq!(
            next.skill.latest_version,
            Some(next.version.version.clone())
        );
        assert_eq!(
            next.skill.default_version,
            Some(created.version.version.clone())
        );

        Ok(())
    }

    #[tokio::test]
    async fn update_skill_switches_default_projection() -> Result<()> {
        let root = TempDir::new()?;
        let blob_store = Arc::new(FilesystemBlobStore::new(root.path())?);
        let service = SkillService::in_memory(blob_store);

        let created = service
            .create_skill(CreateSkillRequest {
                tenant_id: "tenant-a".to_string(),
                upload: SkillUpload::Files(vec![UploadedSkillFile {
                    relative_path: "SKILL.md".to_string(),
                    contents: b"---\nname: acme:map\ndescription: Map the repo\n---\nUse rg."
                        .to_vec(),
                    media_type: Some("text/markdown".to_string()),
                }]),
            })
            .await?;

        let second = service
            .create_skill_version(CreateSkillVersionRequest {
                tenant_id: "tenant-a".to_string(),
                skill_id: created.skill.skill_id.clone(),
                upload: SkillUpload::Files(vec![
                    UploadedSkillFile {
                        relative_path: "SKILL.md".to_string(),
                        contents:
                            b"---\nname: acme:search\ndescription: Search the repo\n---\nUse fd."
                                .to_vec(),
                        media_type: Some("text/markdown".to_string()),
                    },
                    UploadedSkillFile {
                        relative_path: "scripts/run.py".to_string(),
                        contents: b"print('v2')".to_vec(),
                        media_type: Some("text/x-python".to_string()),
                    },
                ]),
            })
            .await?;

        assert_eq!(second.skill.name, "acme:map");
        assert!(!second.skill.has_code_files);

        let updated = service
            .update_skill(UpdateSkillRequest {
                tenant_id: "tenant-a".to_string(),
                skill_id: created.skill.skill_id.clone(),
                default_version_ref: "2".to_string(),
            })
            .await?;

        assert_eq!(
            updated.default_version,
            Some(second.version.version.clone())
        );
        assert_eq!(updated.name, "acme:search");
        assert!(updated.has_code_files);

        Ok(())
    }

    #[tokio::test]
    async fn delete_default_version_requires_switch_when_multiple_versions_remain() -> Result<()> {
        let root = TempDir::new()?;
        let blob_store = Arc::new(FilesystemBlobStore::new(root.path())?);
        let service = SkillService::in_memory(blob_store);

        let created = service
            .create_skill(CreateSkillRequest {
                tenant_id: "tenant-a".to_string(),
                upload: SkillUpload::Files(vec![UploadedSkillFile {
                    relative_path: "SKILL.md".to_string(),
                    contents: b"---\nname: acme:map\ndescription: Map the repo\n---\nUse rg."
                        .to_vec(),
                    media_type: Some("text/markdown".to_string()),
                }]),
            })
            .await?;
        let second = service
            .create_skill_version(CreateSkillVersionRequest {
                tenant_id: "tenant-a".to_string(),
                skill_id: created.skill.skill_id.clone(),
                upload: SkillUpload::Files(vec![UploadedSkillFile {
                    relative_path: "SKILL.md".to_string(),
                    contents: b"---\nname: acme:search\ndescription: Search the repo\n---\nUse fd."
                        .to_vec(),
                    media_type: Some("text/markdown".to_string()),
                }]),
            })
            .await?;

        let error = service
            .delete_skill_version(
                "tenant-a",
                &created.skill.skill_id,
                &created.version.version,
            )
            .await
            .expect_err("default version delete should fail");
        assert!(matches!(
            error,
            SkillServiceError::CannotDeleteDefaultVersion { .. }
        ));

        let switched = service
            .update_skill(UpdateSkillRequest {
                tenant_id: "tenant-a".to_string(),
                skill_id: created.skill.skill_id.clone(),
                default_version_ref: second.version.version.clone(),
            })
            .await?;
        assert_eq!(
            switched.default_version,
            Some(second.version.version.clone())
        );

        service
            .delete_skill_version(
                "tenant-a",
                &created.skill.skill_id,
                &created.version.version,
            )
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn delete_last_version_removes_skill() -> Result<()> {
        let root = TempDir::new()?;
        let blob_store = Arc::new(FilesystemBlobStore::new(root.path())?);
        let service = SkillService::in_memory(blob_store);

        let created = service
            .create_skill(CreateSkillRequest {
                tenant_id: "tenant-a".to_string(),
                upload: SkillUpload::Files(vec![UploadedSkillFile {
                    relative_path: "SKILL.md".to_string(),
                    contents: b"---\nname: acme:map\ndescription: Map the repo\n---\nUse rg."
                        .to_vec(),
                    media_type: Some("text/markdown".to_string()),
                }]),
            })
            .await?;

        let deleted = service
            .delete_skill_version(
                "tenant-a",
                &created.skill.skill_id,
                &created.version.version,
            )
            .await?;
        assert!(deleted.deleted_skill);

        let metadata_store = service
            .metadata_store()
            .ok_or_else(|| anyhow!("metadata store missing"))?;
        assert!(metadata_store
            .get_skill("tenant-a", &created.skill.skill_id)
            .await?
            .is_none());
        assert!(metadata_store
            .list_skill_versions(&created.skill.skill_id)
            .await?
            .is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn update_skill_version_marks_deprecated_and_updates_skill_timestamp() -> Result<()> {
        let root = TempDir::new()?;
        let blob_store = Arc::new(FilesystemBlobStore::new(root.path())?);
        let service = SkillService::in_memory(blob_store);

        let created = service
            .create_skill(CreateSkillRequest {
                tenant_id: "tenant-a".to_string(),
                upload: SkillUpload::Files(vec![UploadedSkillFile {
                    relative_path: "SKILL.md".to_string(),
                    contents: b"---\nname: acme:map\ndescription: Map the repo\n---\nUse rg."
                        .to_vec(),
                    media_type: Some("text/markdown".to_string()),
                }]),
            })
            .await?;
        let before_updated_at = created.skill.updated_at;

        let updated = service
            .update_skill_version(UpdateSkillVersionRequest {
                tenant_id: "tenant-a".to_string(),
                skill_id: created.skill.skill_id.clone(),
                version_ref: created.version.version.clone(),
                deprecated: true,
            })
            .await?;
        assert!(updated.deprecated);

        let skill = service
            .get_skill("tenant-a", &created.skill.skill_id)
            .await?;
        assert!(skill.updated_at >= before_updated_at);

        Ok(())
    }

    #[tokio::test]
    async fn list_skills_filters_by_source_and_name() -> Result<()> {
        let root = TempDir::new()?;
        let blob_store = Arc::new(FilesystemBlobStore::new(root.path())?);
        let service = SkillService::in_memory(blob_store);

        service
            .create_skill(CreateSkillRequest {
                tenant_id: "tenant-a".to_string(),
                upload: SkillUpload::Files(vec![UploadedSkillFile {
                    relative_path: "SKILL.md".to_string(),
                    contents: b"---\nname: acme:map\ndescription: Map the repo\n---\nUse rg."
                        .to_vec(),
                    media_type: Some("text/markdown".to_string()),
                }]),
            })
            .await?;
        service
            .create_skill(CreateSkillRequest {
                tenant_id: "tenant-a".to_string(),
                upload: SkillUpload::Files(vec![UploadedSkillFile {
                    relative_path: "SKILL.md".to_string(),
                    contents: b"---\nname: acme:search\ndescription: Search the repo\n---\nUse fd."
                        .to_vec(),
                    media_type: Some("text/markdown".to_string()),
                }]),
            })
            .await?;
        service
            .create_skill(CreateSkillRequest {
                tenant_id: "tenant-b".to_string(),
                upload: SkillUpload::Files(vec![UploadedSkillFile {
                    relative_path: "SKILL.md".to_string(),
                    contents: b"---\nname: acme:map\ndescription: Tenant B skill\n---\nUse rg."
                        .to_vec(),
                    media_type: Some("text/markdown".to_string()),
                }]),
            })
            .await?;

        let tenant_a = service
            .list_skills("tenant-a", Some("custom"), None)
            .await?;
        assert_eq!(tenant_a.len(), 2);
        assert!(tenant_a.iter().all(|record| record.tenant_id == "tenant-a"));

        let filtered = service
            .list_skills("tenant-a", Some("custom"), Some("acme:map"))
            .await?;
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "acme:map");

        Ok(())
    }

    #[tokio::test]
    async fn get_skill_version_resolves_latest_and_integer_version_number() -> Result<()> {
        let root = TempDir::new()?;
        let blob_store = Arc::new(FilesystemBlobStore::new(root.path())?);
        let service = SkillService::in_memory(blob_store);

        let created = service
            .create_skill(CreateSkillRequest {
                tenant_id: "tenant-a".to_string(),
                upload: SkillUpload::Files(vec![UploadedSkillFile {
                    relative_path: "SKILL.md".to_string(),
                    contents: b"---\nname: acme:map\ndescription: Map the repo\n---\nUse rg."
                        .to_vec(),
                    media_type: Some("text/markdown".to_string()),
                }]),
            })
            .await?;

        let second = service
            .create_skill_version(CreateSkillVersionRequest {
                tenant_id: "tenant-a".to_string(),
                skill_id: created.skill.skill_id.clone(),
                upload: SkillUpload::Files(vec![UploadedSkillFile {
                    relative_path: "SKILL.md".to_string(),
                    contents: b"---\nname: acme:map\ndescription: Updated mapping skill\n---\nUse rg --files."
                        .to_vec(),
                    media_type: Some("text/markdown".to_string()),
                }]),
            })
            .await?;

        let latest = service
            .get_skill_version("tenant-a", &created.skill.skill_id, "latest")
            .await?;
        assert_eq!(latest.version, second.version.version);

        let by_number = service
            .get_skill_version("tenant-a", &created.skill.skill_id, "2")
            .await?;
        assert_eq!(by_number.version, second.version.version);

        Ok(())
    }
}
