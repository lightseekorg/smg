use axum::response::Response;
use openai_protocol::{
    messages::CreateMessageRequest,
    responses::{
        CodeInterpreterTool, LocalShellEnvironment, ResponseInput, ResponseInputOutputItem,
        ResponseTool, ResponseToolEnvironment, ResponsesRequest, ShellCallEnvironment,
        ShellEnvironment, ShellTool,
    },
    skills::{
        MessagesSkillRef, OpaqueOpenAIObject, ResponsesSkillEntry, ResponsesSkillRef,
        SkillVersionRef,
    },
};
use serde_json::Value;
use smg_skills::{SkillRecord, SkillService, SkillVersionRecord, SkillsStoreError};
use tracing::error;

use crate::{routers::error as route_error, tenant::TenantKey};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedSkillManifest {
    refs: Vec<ResolvedSkillRef>,
}

impl ResolvedSkillManifest {
    #[must_use]
    pub fn new(refs: Vec<ResolvedSkillRef>) -> Self {
        Self { refs }
    }

    #[must_use]
    pub fn refs(&self) -> &[ResolvedSkillRef] {
        &self.refs
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.refs.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedSkillRef {
    AnthropicProvider {
        skill_id: String,
        raw_version: Option<String>,
    },
    OpenAIProvider {
        skill_id: String,
        raw_version: Option<String>,
    },
    OpenAIOpaquePassThrough {
        raw: Value,
    },
    SmgStorage {
        skill_id: String,
        requested_version: Option<SkillVersionRef>,
        pinned: PinnedSkillVersion,
    },
    ClientLocalPath {
        name: String,
        description: String,
        path: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedSkillVersion {
    pub version: String,
    pub version_number: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum SkillResolutionError {
    #[error("skills are not enabled")]
    SkillsNotEnabled,

    #[error("skills metadata store is not available")]
    MissingMetadataStore,

    #[error("skill not found")]
    SkillNotFound,

    #[error("skill version not found")]
    SkillVersionNotFound,

    #[error("skills metadata lookup failed: {0}")]
    Store(#[from] SkillsStoreError),
}

impl SkillResolutionError {
    #[must_use]
    pub fn into_response(self) -> Response {
        match self {
            Self::SkillsNotEnabled => route_error::bad_request(
                "skills_not_enabled",
                "SMG skills are not enabled for this gateway",
            ),
            Self::SkillNotFound => {
                route_error::bad_request("skill_not_found", "Referenced SMG skill was not found")
            }
            Self::SkillVersionNotFound => route_error::bad_request(
                "skill_version_not_found",
                "Referenced SMG skill version was not found",
            ),
            Self::MissingMetadataStore | Self::Store(_) => {
                error!(error = %self, "failed to resolve request skills");
                route_error::internal_error(
                    "skills_resolution_failed",
                    "Failed to resolve request skills",
                )
            }
        }
    }
}

pub async fn resolve_messages_skill_manifest(
    skill_service: Option<&SkillService>,
    tenant_key: &TenantKey,
    request: &CreateMessageRequest,
) -> Result<ResolvedSkillManifest, SkillResolutionError> {
    let Some(skills) = request
        .container
        .as_ref()
        .and_then(|container| container.skills.as_ref())
    else {
        return Ok(ResolvedSkillManifest::default());
    };
    if skills.is_empty() {
        return Ok(ResolvedSkillManifest::default());
    }

    let mut refs = Vec::with_capacity(skills.len());
    for skill in skills {
        refs.push(resolve_messages_skill_ref(skill_service, tenant_key, skill).await?);
    }
    Ok(ResolvedSkillManifest::new(refs))
}

pub async fn resolve_responses_skill_manifest(
    skill_service: Option<&SkillService>,
    tenant_key: &TenantKey,
    request: &ResponsesRequest,
) -> Result<ResolvedSkillManifest, SkillResolutionError> {
    let mut refs = Vec::new();

    if let Some(tools) = &request.tools {
        for tool in tools {
            resolve_response_tool_skills(skill_service, tenant_key, tool, &mut refs).await?;
        }
    }

    if let ResponseInput::Items(items) = &request.input {
        for item in items {
            resolve_response_input_item_skills(skill_service, tenant_key, item, &mut refs).await?;
        }
    }

    Ok(ResolvedSkillManifest::new(refs))
}

async fn resolve_messages_skill_ref(
    skill_service: Option<&SkillService>,
    tenant_key: &TenantKey,
    skill: &MessagesSkillRef,
) -> Result<ResolvedSkillRef, SkillResolutionError> {
    match skill {
        MessagesSkillRef::Anthropic { skill_id, version } => {
            Ok(ResolvedSkillRef::AnthropicProvider {
                skill_id: skill_id.clone(),
                raw_version: version.clone(),
            })
        }
        MessagesSkillRef::Custom { skill_id, version } => {
            let pinned =
                resolve_required_smg_skill(skill_service, tenant_key, skill_id, version.as_ref())
                    .await?;
            Ok(ResolvedSkillRef::SmgStorage {
                skill_id: skill_id.clone(),
                requested_version: version.clone(),
                pinned,
            })
        }
    }
}

async fn resolve_response_tool_skills(
    skill_service: Option<&SkillService>,
    tenant_key: &TenantKey,
    tool: &ResponseTool,
    refs: &mut Vec<ResolvedSkillRef>,
) -> Result<(), SkillResolutionError> {
    match tool {
        ResponseTool::CodeInterpreter(CodeInterpreterTool { environment, .. }) => {
            resolve_response_tool_environment_skills(
                skill_service,
                tenant_key,
                environment.as_ref(),
                refs,
            )
            .await
        }
        ResponseTool::Shell(ShellTool { environment }) => {
            resolve_shell_environment_skills(skill_service, tenant_key, environment.as_ref(), refs)
                .await
        }
        _ => Ok(()),
    }
}

async fn resolve_response_input_item_skills(
    skill_service: Option<&SkillService>,
    tenant_key: &TenantKey,
    item: &ResponseInputOutputItem,
    refs: &mut Vec<ResolvedSkillRef>,
) -> Result<(), SkillResolutionError> {
    if let ResponseInputOutputItem::ShellCall { environment, .. } = item {
        resolve_shell_call_environment_skills(
            skill_service,
            tenant_key,
            environment.as_ref(),
            refs,
        )
        .await?;
    }
    Ok(())
}

async fn resolve_response_tool_environment_skills(
    skill_service: Option<&SkillService>,
    tenant_key: &TenantKey,
    environment: Option<&ResponseToolEnvironment>,
    refs: &mut Vec<ResolvedSkillRef>,
) -> Result<(), SkillResolutionError> {
    if let Some(skills) = environment.and_then(|environment| environment.skills.as_ref()) {
        resolve_responses_skill_entries(skill_service, tenant_key, skills, refs).await?;
    }
    Ok(())
}

async fn resolve_shell_environment_skills(
    skill_service: Option<&SkillService>,
    tenant_key: &TenantKey,
    environment: Option<&ShellEnvironment>,
    refs: &mut Vec<ResolvedSkillRef>,
) -> Result<(), SkillResolutionError> {
    match environment {
        Some(ShellEnvironment::ContainerAuto(environment)) => {
            if let Some(skills) = &environment.skills {
                resolve_responses_skill_entries(skill_service, tenant_key, skills, refs).await?;
            }
        }
        Some(ShellEnvironment::Local(LocalShellEnvironment { skills })) => {
            if let Some(skills) = skills {
                resolve_responses_skill_entries(skill_service, tenant_key, skills, refs).await?;
            }
        }
        Some(ShellEnvironment::ContainerReference(_)) | None => {}
    }
    Ok(())
}

async fn resolve_shell_call_environment_skills(
    skill_service: Option<&SkillService>,
    tenant_key: &TenantKey,
    environment: Option<&ShellCallEnvironment>,
    refs: &mut Vec<ResolvedSkillRef>,
) -> Result<(), SkillResolutionError> {
    if let Some(ShellCallEnvironment::Local(LocalShellEnvironment {
        skills: Some(skills),
    })) = environment
    {
        resolve_responses_skill_entries(skill_service, tenant_key, skills, refs).await?;
    }
    Ok(())
}

async fn resolve_responses_skill_entries(
    skill_service: Option<&SkillService>,
    tenant_key: &TenantKey,
    skills: &[ResponsesSkillEntry],
    refs: &mut Vec<ResolvedSkillRef>,
) -> Result<(), SkillResolutionError> {
    refs.reserve(skills.len());
    for skill in skills {
        refs.push(resolve_responses_skill_entry(skill_service, tenant_key, skill).await?);
    }
    Ok(())
}

async fn resolve_responses_skill_entry(
    skill_service: Option<&SkillService>,
    tenant_key: &TenantKey,
    skill: &ResponsesSkillEntry,
) -> Result<ResolvedSkillRef, SkillResolutionError> {
    match skill {
        ResponsesSkillEntry::Typed(ResponsesSkillRef::Reference { skill_id, version }) => {
            resolve_responses_reference(skill_service, tenant_key, skill_id, version.as_ref()).await
        }
        ResponsesSkillEntry::Typed(ResponsesSkillRef::Local {
            name,
            description,
            path,
        }) => Ok(ResolvedSkillRef::ClientLocalPath {
            name: name.clone(),
            description: description.clone(),
            path: path.clone(),
        }),
        ResponsesSkillEntry::OpaqueOpenAI(OpaqueOpenAIObject(raw)) => {
            Ok(ResolvedSkillRef::OpenAIOpaquePassThrough {
                raw: Value::Object(raw.clone()),
            })
        }
    }
}

async fn resolve_responses_reference(
    skill_service: Option<&SkillService>,
    tenant_key: &TenantKey,
    skill_id: &str,
    version: Option<&SkillVersionRef>,
) -> Result<ResolvedSkillRef, SkillResolutionError> {
    let Some(service) = skill_service else {
        return Ok(ResolvedSkillRef::OpenAIProvider {
            skill_id: skill_id.to_string(),
            raw_version: version.map(skill_version_ref_to_string),
        });
    };
    let metadata_store = service
        .metadata_store()
        .ok_or(SkillResolutionError::MissingMetadataStore)?;

    let Some(skill) = metadata_store
        .get_skill(tenant_key.as_str(), skill_id)
        .await?
    else {
        return Ok(ResolvedSkillRef::OpenAIProvider {
            skill_id: skill_id.to_string(),
            raw_version: version.map(skill_version_ref_to_string),
        });
    };

    let pinned = resolve_smg_skill_version(service, skill, version).await?;
    Ok(ResolvedSkillRef::SmgStorage {
        skill_id: skill_id.to_string(),
        requested_version: version.cloned(),
        pinned,
    })
}

async fn resolve_required_smg_skill(
    skill_service: Option<&SkillService>,
    tenant_key: &TenantKey,
    skill_id: &str,
    version: Option<&SkillVersionRef>,
) -> Result<PinnedSkillVersion, SkillResolutionError> {
    let service = skill_service.ok_or(SkillResolutionError::SkillsNotEnabled)?;
    let metadata_store = service
        .metadata_store()
        .ok_or(SkillResolutionError::MissingMetadataStore)?;
    let skill = metadata_store
        .get_skill(tenant_key.as_str(), skill_id)
        .await?
        .ok_or(SkillResolutionError::SkillNotFound)?;
    resolve_smg_skill_version(service, skill, version).await
}

async fn resolve_smg_skill_version(
    service: &SkillService,
    skill: SkillRecord,
    version: Option<&SkillVersionRef>,
) -> Result<PinnedSkillVersion, SkillResolutionError> {
    let record = match version {
        None => {
            let default_version = skill
                .default_version
                .as_deref()
                .ok_or(SkillResolutionError::SkillVersionNotFound)?;
            get_exact_skill_version(service, &skill, default_version).await?
        }
        Some(SkillVersionRef::Latest) => {
            let latest_version = skill
                .latest_version
                .as_deref()
                .ok_or(SkillResolutionError::SkillVersionNotFound)?;
            get_exact_skill_version(service, &skill, latest_version).await?
        }
        Some(SkillVersionRef::Timestamp(version)) => {
            get_exact_skill_version(service, &skill, version).await?
        }
        Some(SkillVersionRef::Integer(version_number)) => {
            get_skill_version_by_number(service, &skill, *version_number).await?
        }
    };

    Ok(PinnedSkillVersion {
        version: record.version,
        version_number: record.version_number,
    })
}

async fn get_exact_skill_version(
    service: &SkillService,
    skill: &SkillRecord,
    version: &str,
) -> Result<SkillVersionRecord, SkillResolutionError> {
    let metadata_store = service
        .metadata_store()
        .ok_or(SkillResolutionError::MissingMetadataStore)?;
    metadata_store
        .get_skill_version(&skill.skill_id, version)
        .await?
        .ok_or(SkillResolutionError::SkillVersionNotFound)
}

async fn get_skill_version_by_number(
    service: &SkillService,
    skill: &SkillRecord,
    version_number: u32,
) -> Result<SkillVersionRecord, SkillResolutionError> {
    let metadata_store = service
        .metadata_store()
        .ok_or(SkillResolutionError::MissingMetadataStore)?;
    metadata_store
        .list_skill_versions(&skill.skill_id)
        .await?
        .into_iter()
        .find(|record| record.version_number == version_number)
        .ok_or(SkillResolutionError::SkillVersionNotFound)
}

fn skill_version_ref_to_string(version: &SkillVersionRef) -> String {
    match version {
        SkillVersionRef::Latest => "latest".to_string(),
        SkillVersionRef::Integer(version) => version.to_string(),
        SkillVersionRef::Timestamp(version) => version.clone(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::{anyhow, Result};
    use openai_protocol::{messages::CreateMessageRequest, responses::ResponsesRequest};
    use smg_blob_storage::FilesystemBlobStore;
    use smg_skills::{
        CreateSkillRequest, CreateSkillVersionRequest, SkillService, SkillUpload,
        UpdateSkillRequest, UploadedSkillFile,
    };
    use tempfile::TempDir;

    use super::*;

    const TENANT_ID: &str = "auth:test-tenant";

    async fn create_test_service() -> Result<(TempDir, SkillService, String)> {
        let root = TempDir::new()?;
        let blob_store = Arc::new(FilesystemBlobStore::new(root.path())?);
        let service = SkillService::in_memory(blob_store);
        let result = service
            .create_skill(CreateSkillRequest {
                tenant_id: TENANT_ID.to_string(),
                upload: SkillUpload::Files(vec![UploadedSkillFile {
                    relative_path: "SKILL.md".to_string(),
                    contents: b"---\nname: acme-map\ndescription: Map the repo\n---\nUse rg."
                        .to_vec(),
                    media_type: Some("text/markdown".to_string()),
                }]),
            })
            .await?;
        Ok((root, service, result.skill.skill_id))
    }

    async fn create_second_version(service: &SkillService, skill_id: &str) -> Result<String> {
        let result = service
            .create_skill_version(CreateSkillVersionRequest {
                tenant_id: TENANT_ID.to_string(),
                skill_id: skill_id.to_string(),
                upload: SkillUpload::Files(vec![UploadedSkillFile {
                    relative_path: "SKILL.md".to_string(),
                    contents: b"---\nname: acme-map-v2\ndescription: Map the repo v2\n---\nUse fd."
                        .to_vec(),
                    media_type: Some("text/markdown".to_string()),
                }]),
            })
            .await?;
        Ok(result.version.version)
    }

    fn tenant_key() -> TenantKey {
        TenantKey::from(TENANT_ID)
    }

    fn messages_request(skill_id: &str, version: Option<Value>) -> Result<CreateMessageRequest> {
        let mut skill = serde_json::json!({
            "type": "custom",
            "skill_id": skill_id
        });
        if let Some(version) = version {
            skill["version"] = version;
        }

        Ok(serde_json::from_value(serde_json::json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "container": {
                "skills": [skill]
            }
        }))?)
    }

    fn responses_request(skills: Vec<Value>) -> Result<ResponsesRequest> {
        Ok(serde_json::from_value(serde_json::json!({
            "model": "gpt-5.1",
            "input": "hi",
            "tools": [{
                "type": "code_interpreter",
                "environment": {
                    "skills": skills
                }
            }]
        }))?)
    }

    fn only_ref(manifest: &ResolvedSkillManifest) -> Result<&ResolvedSkillRef> {
        manifest
            .refs()
            .first()
            .ok_or_else(|| anyhow!("expected one resolved skill"))
    }

    #[tokio::test]
    async fn messages_custom_default_version_is_pinned_at_resolution() -> Result<()> {
        let (_root, service, skill_id) = create_test_service().await?;
        let request = messages_request(&skill_id, None)?;
        let manifest =
            resolve_messages_skill_manifest(Some(&service), &tenant_key(), &request).await?;

        let second_version = create_second_version(&service, &skill_id).await?;
        service
            .update_skill(UpdateSkillRequest {
                tenant_id: TENANT_ID.to_string(),
                skill_id: skill_id.clone(),
                default_version_ref: second_version.clone(),
            })
            .await?;

        match only_ref(&manifest)? {
            ResolvedSkillRef::SmgStorage { pinned, .. } => {
                assert_eq!(pinned.version_number, 1);
                assert_ne!(pinned.version, second_version);
            }
            other => return Err(anyhow!("expected SMG storage ref, got {other:?}")),
        }

        Ok(())
    }

    #[tokio::test]
    async fn messages_custom_latest_version_is_pinned_at_resolution() -> Result<()> {
        let (_root, service, skill_id) = create_test_service().await?;
        let request = messages_request(&skill_id, Some(Value::String("latest".to_string())))?;
        let manifest =
            resolve_messages_skill_manifest(Some(&service), &tenant_key(), &request).await?;

        let second_version = create_second_version(&service, &skill_id).await?;

        match only_ref(&manifest)? {
            ResolvedSkillRef::SmgStorage { pinned, .. } => {
                assert_eq!(pinned.version_number, 1);
                assert_ne!(pinned.version, second_version);
            }
            other => return Err(anyhow!("expected SMG storage ref, got {other:?}")),
        }

        Ok(())
    }

    #[tokio::test]
    async fn responses_reference_uses_storage_lookup_instead_of_id_shape() -> Result<()> {
        let (_root, service, skill_id) = create_test_service().await?;
        let request = responses_request(vec![
            serde_json::json!({
                "type": "skill_reference",
                "skill_id": skill_id
            }),
            serde_json::json!({
                "type": "skill_reference",
                "skill_id": "openai-spreadsheets",
                "version": 2
            }),
        ])?;

        let manifest =
            resolve_responses_skill_manifest(Some(&service), &tenant_key(), &request).await?;

        assert_eq!(manifest.refs().len(), 2);
        match &manifest.refs()[0] {
            ResolvedSkillRef::SmgStorage { pinned, .. } => {
                assert_eq!(pinned.version_number, 1);
            }
            other => return Err(anyhow!("expected SMG storage ref, got {other:?}")),
        }
        match &manifest.refs()[1] {
            ResolvedSkillRef::OpenAIProvider {
                skill_id,
                raw_version,
            } => {
                assert_eq!(skill_id, "openai-spreadsheets");
                assert_eq!(raw_version.as_deref(), Some("2"));
            }
            other => return Err(anyhow!("expected OpenAI provider ref, got {other:?}")),
        }

        Ok(())
    }

    #[tokio::test]
    async fn responses_local_and_opaque_entries_are_pass_through() -> Result<()> {
        let request = responses_request(vec![
            serde_json::json!({
                "type": "local",
                "name": "repo",
                "description": "local checkout",
                "path": "/workspace/repo"
            }),
            serde_json::json!({
                "type": "openai_inline_skill",
                "name": "provider-owned",
                "payload": {"any": "shape"}
            }),
        ])?;

        let manifest = resolve_responses_skill_manifest(None, &tenant_key(), &request).await?;

        assert_eq!(manifest.refs().len(), 2);
        assert!(matches!(
            &manifest.refs()[0],
            ResolvedSkillRef::ClientLocalPath { name, .. } if name == "repo"
        ));
        assert!(matches!(
            &manifest.refs()[1],
            ResolvedSkillRef::OpenAIOpaquePassThrough { raw }
                if raw.get("type").and_then(Value::as_str) == Some("openai_inline_skill")
        ));

        Ok(())
    }

    #[tokio::test]
    async fn messages_custom_requires_enabled_smg_skills() -> Result<()> {
        let request = messages_request("skill_missing", None)?;
        let error = resolve_messages_skill_manifest(None, &tenant_key(), &request)
            .await
            .err()
            .ok_or_else(|| anyhow!("expected skills-not-enabled error"))?;

        assert!(matches!(error, SkillResolutionError::SkillsNotEnabled));
        Ok(())
    }
}
