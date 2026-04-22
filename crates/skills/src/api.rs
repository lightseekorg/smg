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
        normalize_skill_bundle_zip, parse_skill_bundle, SkillBundleArchiveError, SkillParseError,
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
pub struct SkillCreateResult {
    pub skill: SkillRecord,
    pub version: SkillVersionRecord,
    pub warnings: Vec<SkillParseWarning>,
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

        let (bundle, media_types) = normalize_upload_bundle(request.upload)?;
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

        if let Err(error) = metadata_store.put_skill(skill_record.clone()).await {
            cleanup_blobs(&*blob_store, &version_record.file_manifest).await;
            return Err(error.into());
        }

        if let Err(error) = metadata_store
            .put_skill_version(version_record.clone())
            .await
        {
            let _ = metadata_store.delete_skill(&tenant_id, &skill_id).await;
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

        let (bundle, media_types) = normalize_upload_bundle(request.upload)?;
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

        if let Err(error) = metadata_store
            .put_skill_version(version_record.clone())
            .await
        {
            cleanup_blobs(&*blob_store, &version_record.file_manifest).await;
            return Err(error.into());
        }

        let mut updated_skill = existing_skill.clone();
        updated_skill.name = parsed_bundle.name.clone();
        updated_skill.short_description = parsed_bundle.short_description.clone();
        updated_skill.description = Some(parsed_bundle.description.clone());
        updated_skill.has_code_files = bundle.has_code_files;
        updated_skill.latest_version = Some(version.clone());
        if updated_skill.default_version.is_none() {
            updated_skill.default_version = Some(version.clone());
        }
        updated_skill.updated_at = now;

        if let Err(error) = metadata_store.put_skill(updated_skill.clone()).await {
            let _ = metadata_store
                .delete_skill_version(&skill_id, &version)
                .await;
            cleanup_blobs(&*blob_store, &version_record.file_manifest).await;
            return Err(error.into());
        }

        Ok(SkillCreateResult {
            skill: updated_skill,
            version: version_record,
            warnings: parsed_bundle.warnings,
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
) -> Result<(NormalizedSkillBundle, HashMap<String, Option<String>>), SkillServiceError> {
    match upload {
        SkillUpload::Zip(bytes) => {
            let bundle = normalize_skill_bundle_zip(&bytes)?;
            Ok((bundle, HashMap::new()))
        }
        SkillUpload::Files(files) => {
            if files.is_empty() {
                return Err(SkillServiceError::MissingUploadParts);
            }
            let media_types = files
                .iter()
                .map(|file| (file.relative_path.clone(), file.media_type.clone()))
                .collect::<HashMap<_, _>>();
            let bundle = normalize_files_upload(&files)?;
            Ok((bundle, media_types))
        }
    }
}

fn normalize_files_upload(
    files: &[UploadedSkillFile],
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

    normalize_skill_bundle_zip(buffer.get_ref()).map_err(Into::into)
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::{anyhow, Result};
    use smg_blob_storage::FilesystemBlobStore;
    use tempfile::TempDir;

    use super::{
        CreateSkillRequest, CreateSkillVersionRequest, SkillService, SkillServiceMode, SkillUpload,
        UploadedSkillFile,
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
}
