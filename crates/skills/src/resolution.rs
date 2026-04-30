//! Request-local skill reference resolution.
//!
//! This module owns the skills feature semantics for Messages and Responses
//! request shapes. The gateway invokes it after tenant metadata is resolved and
//! before forwarding to provider-specific routers.

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

use crate::{PinnedSkillVersion, SkillService, SkillServiceError, SkillVersionSelector};

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

#[derive(Debug, thiserror::Error)]
pub enum SkillResolutionError {
    #[error("skills are not enabled")]
    SkillsNotEnabled,

    #[error(transparent)]
    Service(#[from] SkillServiceError),
}

pub async fn resolve_messages_skill_manifest(
    skill_service: Option<&SkillService>,
    tenant_id: &str,
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
        refs.push(resolve_messages_skill_ref(skill_service, tenant_id, skill).await?);
    }
    Ok(ResolvedSkillManifest::new(refs))
}

pub async fn resolve_responses_skill_manifest(
    skill_service: Option<&SkillService>,
    tenant_id: &str,
    request: &ResponsesRequest,
) -> Result<ResolvedSkillManifest, SkillResolutionError> {
    let mut refs = Vec::new();

    if let Some(tools) = &request.tools {
        for tool in tools {
            resolve_response_tool_skills(skill_service, tenant_id, tool, &mut refs).await?;
        }
    }

    if let ResponseInput::Items(items) = &request.input {
        for item in items {
            resolve_response_input_item_skills(skill_service, tenant_id, item, &mut refs).await?;
        }
    }

    Ok(ResolvedSkillManifest::new(refs))
}

async fn resolve_messages_skill_ref(
    skill_service: Option<&SkillService>,
    tenant_id: &str,
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
            let service = skill_service.ok_or(SkillResolutionError::SkillsNotEnabled)?;
            let pinned = service
                .pin_skill_version(
                    tenant_id,
                    skill_id,
                    skill_version_selector(version.as_ref()),
                )
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
    tenant_id: &str,
    tool: &ResponseTool,
    refs: &mut Vec<ResolvedSkillRef>,
) -> Result<(), SkillResolutionError> {
    match tool {
        ResponseTool::CodeInterpreter(CodeInterpreterTool { environment, .. }) => {
            resolve_response_tool_environment_skills(
                skill_service,
                tenant_id,
                environment.as_ref(),
                refs,
            )
            .await
        }
        ResponseTool::Shell(ShellTool { environment }) => {
            resolve_shell_environment_skills(skill_service, tenant_id, environment.as_ref(), refs)
                .await
        }
        _ => Ok(()),
    }
}

async fn resolve_response_input_item_skills(
    skill_service: Option<&SkillService>,
    tenant_id: &str,
    item: &ResponseInputOutputItem,
    refs: &mut Vec<ResolvedSkillRef>,
) -> Result<(), SkillResolutionError> {
    if let ResponseInputOutputItem::ShellCall { environment, .. } = item {
        resolve_shell_call_environment_skills(skill_service, tenant_id, environment.as_ref(), refs)
            .await?;
    }
    Ok(())
}

async fn resolve_response_tool_environment_skills(
    skill_service: Option<&SkillService>,
    tenant_id: &str,
    environment: Option<&ResponseToolEnvironment>,
    refs: &mut Vec<ResolvedSkillRef>,
) -> Result<(), SkillResolutionError> {
    if let Some(skills) = environment.and_then(|environment| environment.skills.as_ref()) {
        resolve_responses_skill_entries(skill_service, tenant_id, skills, refs).await?;
    }
    Ok(())
}

async fn resolve_shell_environment_skills(
    skill_service: Option<&SkillService>,
    tenant_id: &str,
    environment: Option<&ShellEnvironment>,
    refs: &mut Vec<ResolvedSkillRef>,
) -> Result<(), SkillResolutionError> {
    match environment {
        Some(ShellEnvironment::ContainerAuto(environment)) => {
            if let Some(skills) = &environment.skills {
                resolve_responses_skill_entries(skill_service, tenant_id, skills, refs).await?;
            }
        }
        Some(ShellEnvironment::Local(LocalShellEnvironment { skills })) => {
            if let Some(skills) = skills {
                resolve_responses_skill_entries(skill_service, tenant_id, skills, refs).await?;
            }
        }
        Some(ShellEnvironment::ContainerReference(_)) | None => {}
    }
    Ok(())
}

async fn resolve_shell_call_environment_skills(
    skill_service: Option<&SkillService>,
    tenant_id: &str,
    environment: Option<&ShellCallEnvironment>,
    refs: &mut Vec<ResolvedSkillRef>,
) -> Result<(), SkillResolutionError> {
    if let Some(ShellCallEnvironment::Local(LocalShellEnvironment {
        skills: Some(skills),
    })) = environment
    {
        resolve_responses_skill_entries(skill_service, tenant_id, skills, refs).await?;
    }
    Ok(())
}

async fn resolve_responses_skill_entries(
    skill_service: Option<&SkillService>,
    tenant_id: &str,
    skills: &[ResponsesSkillEntry],
    refs: &mut Vec<ResolvedSkillRef>,
) -> Result<(), SkillResolutionError> {
    refs.reserve(skills.len());
    for skill in skills {
        refs.push(resolve_responses_skill_entry(skill_service, tenant_id, skill).await?);
    }
    Ok(())
}

async fn resolve_responses_skill_entry(
    skill_service: Option<&SkillService>,
    tenant_id: &str,
    skill: &ResponsesSkillEntry,
) -> Result<ResolvedSkillRef, SkillResolutionError> {
    match skill {
        ResponsesSkillEntry::Typed(ResponsesSkillRef::Reference { skill_id, version }) => {
            resolve_responses_reference(skill_service, tenant_id, skill_id, version.as_ref()).await
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
    tenant_id: &str,
    skill_id: &str,
    version: Option<&SkillVersionRef>,
) -> Result<ResolvedSkillRef, SkillResolutionError> {
    let Some(service) = skill_service else {
        return Ok(ResolvedSkillRef::OpenAIProvider {
            skill_id: skill_id.to_string(),
            raw_version: version.map(skill_version_ref_to_string),
        });
    };

    let Some(pinned) = service
        .try_pin_skill_version(tenant_id, skill_id, skill_version_selector(version))
        .await?
    else {
        return Ok(ResolvedSkillRef::OpenAIProvider {
            skill_id: skill_id.to_string(),
            raw_version: version.map(skill_version_ref_to_string),
        });
    };

    Ok(ResolvedSkillRef::SmgStorage {
        skill_id: skill_id.to_string(),
        requested_version: version.cloned(),
        pinned,
    })
}

fn skill_version_selector(version: Option<&SkillVersionRef>) -> SkillVersionSelector<'_> {
    match version {
        None => SkillVersionSelector::Default,
        Some(SkillVersionRef::Latest) => SkillVersionSelector::Latest,
        Some(SkillVersionRef::Integer(version_number)) => {
            SkillVersionSelector::VersionNumber(*version_number)
        }
        Some(SkillVersionRef::Timestamp(version)) => SkillVersionSelector::Exact(version),
    }
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
    use tempfile::TempDir;

    use super::*;
    use crate::{
        CreateSkillRequest, CreateSkillVersionRequest, SkillService, SkillUpload,
        UpdateSkillRequest, UploadedSkillFile,
    };

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
        let manifest = resolve_messages_skill_manifest(Some(&service), TENANT_ID, &request).await?;

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
        let manifest = resolve_messages_skill_manifest(Some(&service), TENANT_ID, &request).await?;

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
            resolve_responses_skill_manifest(Some(&service), TENANT_ID, &request).await?;

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

        let manifest = resolve_responses_skill_manifest(None, TENANT_ID, &request).await?;

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
        let error = resolve_messages_skill_manifest(None, TENANT_ID, &request)
            .await
            .err()
            .ok_or_else(|| anyhow!("expected skills-not-enabled error"))?;

        assert!(matches!(error, SkillResolutionError::SkillsNotEnabled));
        Ok(())
    }
}
