use axum::{
    extract::{multipart::MultipartError, Multipart, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::skills::{
    SkillMutationResponse, SkillResponse, SkillVersionFileResponse, SkillVersionResponse,
    SkillWarningResponse, SkillsErrorBody, SkillsErrorEnvelope, SKILLS_MULTIPART_BUNDLE_FIELD,
    SKILLS_MULTIPART_FILES_FIELD, SKILLS_MULTIPART_FILE_FIELD, SKILLS_MULTIPART_TENANT_ID_FIELD,
};
use serde_json::Value;
use smg_skills::{
    CreateSkillRequest, CreateSkillVersionRequest, SkillCreateResult, SkillServiceError,
    SkillUpload, UploadedSkillFile,
};

use crate::{middleware::resolve_admin_target_tenant_key, server::AppState};

#[derive(Debug)]
struct ParsedSkillUpload {
    tenant_id: String,
    upload: SkillUpload,
}

#[derive(Debug)]
enum SkillsApiError {
    BadRequest { code: &'static str, message: String },
    NotFound { code: &'static str, message: String },
    Conflict { code: &'static str, message: String },
    Internal { code: &'static str, message: String },
}

impl IntoResponse for SkillsApiError {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            Self::BadRequest { code, message } => (StatusCode::BAD_REQUEST, code, message),
            Self::NotFound { code, message } => (StatusCode::NOT_FOUND, code, message),
            Self::Conflict { code, message } => (StatusCode::CONFLICT, code, message),
            Self::Internal { code, message } => (StatusCode::INTERNAL_SERVER_ERROR, code, message),
        };

        (
            status,
            Json(SkillsErrorEnvelope {
                error: SkillsErrorBody {
                    code: code.to_string(),
                    message,
                },
            }),
        )
            .into_response()
    }
}

impl From<SkillServiceError> for SkillsApiError {
    fn from(error: SkillServiceError) -> Self {
        match error {
            SkillServiceError::MissingTenantId => Self::BadRequest {
                code: "missing_target_tenant",
                message: error.to_string(),
            },
            SkillServiceError::MissingSkillId => Self::BadRequest {
                code: "missing_skill_id",
                message: error.to_string(),
            },
            SkillServiceError::SkillNotFound { .. } => Self::NotFound {
                code: "skill_not_found",
                message: error.to_string(),
            },
            SkillServiceError::MissingUploadParts => Self::BadRequest {
                code: "missing_upload_parts",
                message: error.to_string(),
            },
            SkillServiceError::MixedUploadModes => Self::BadRequest {
                code: "mixed_upload_modes",
                message: error.to_string(),
            },
            SkillServiceError::InvalidZipUpload => Self::BadRequest {
                code: "invalid_zip_upload",
                message: error.to_string(),
            },
            SkillServiceError::MissingFileName => Self::BadRequest {
                code: "missing_file_name",
                message: error.to_string(),
            },
            SkillServiceError::SkillMdNotUtf8 => Self::BadRequest {
                code: "skill_md_not_utf8",
                message: error.to_string(),
            },
            SkillServiceError::BundleArchive(_) | SkillServiceError::BundleParse(_) => {
                Self::BadRequest {
                    code: "invalid_skill_bundle",
                    message: error.to_string(),
                }
            }
            SkillServiceError::Store(smg_skills::SkillsStoreError::InvalidData(_)) => {
                Self::Conflict {
                    code: "skills_conflict",
                    message: error.to_string(),
                }
            }
            SkillServiceError::BundleBuild(_)
            | SkillServiceError::BlobStore(_)
            | SkillServiceError::Store(_)
            | SkillServiceError::MissingComponent { .. } => Self::Internal {
                code: "skills_internal_error",
                message: error.to_string(),
            },
        }
    }
}

pub async fn create_skill(
    State(state): State<std::sync::Arc<AppState>>,
    multipart: Multipart,
) -> Response {
    match create_skill_impl(state, multipart).await {
        Ok(response) => (StatusCode::CREATED, Json(response)).into_response(),
        Err(error) => error.into_response(),
    }
}

pub async fn create_skill_version(
    State(state): State<std::sync::Arc<AppState>>,
    Path(skill_id): Path<String>,
    multipart: Multipart,
) -> Response {
    match create_skill_version_impl(state, skill_id, multipart).await {
        Ok(response) => (StatusCode::CREATED, Json(response)).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn create_skill_impl(
    state: std::sync::Arc<AppState>,
    multipart: Multipart,
) -> Result<SkillMutationResponse, SkillsApiError> {
    let skill_service =
        state
            .context
            .skill_service
            .clone()
            .ok_or_else(|| SkillsApiError::Internal {
                code: "skills_not_configured",
                message: "skills service is not configured".to_string(),
            })?;
    let parsed = parse_skill_upload(multipart).await?;
    let result = skill_service
        .create_skill(CreateSkillRequest {
            tenant_id: parsed.tenant_id,
            upload: parsed.upload,
        })
        .await
        .map_err(SkillsApiError::from)?;
    build_mutation_response(result)
}

async fn create_skill_version_impl(
    state: std::sync::Arc<AppState>,
    skill_id: String,
    multipart: Multipart,
) -> Result<SkillMutationResponse, SkillsApiError> {
    let skill_service =
        state
            .context
            .skill_service
            .clone()
            .ok_or_else(|| SkillsApiError::Internal {
                code: "skills_not_configured",
                message: "skills service is not configured".to_string(),
            })?;
    let parsed = parse_skill_upload(multipart).await?;
    let result = skill_service
        .create_skill_version(CreateSkillVersionRequest {
            tenant_id: parsed.tenant_id,
            skill_id,
            upload: parsed.upload,
        })
        .await
        .map_err(SkillsApiError::from)?;
    build_mutation_response(result)
}

async fn parse_skill_upload(mut multipart: Multipart) -> Result<ParsedSkillUpload, SkillsApiError> {
    let mut tenant_id = None;
    let mut zip_upload = None;
    let mut files = Vec::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| invalid_multipart(format!("failed to read multipart field: {error}")))?
    {
        let field_name = field.name().unwrap_or_default().to_string();
        match field_name.as_str() {
            SKILLS_MULTIPART_TENANT_ID_FIELD => {
                if tenant_id.is_some() {
                    return Err(unexpected_field("tenant_id may only be provided once"));
                }
                let value = field
                    .text()
                    .await
                    .map_err(|error| bad_text_field(SKILLS_MULTIPART_TENANT_ID_FIELD, error))?;
                tenant_id = Some(
                    resolve_admin_target_tenant_key(&value)
                        .map_err(|error| SkillsApiError::BadRequest {
                            code: "missing_target_tenant",
                            message: error.to_string(),
                        })?
                        .to_string(),
                );
            }
            SKILLS_MULTIPART_FILE_FIELD | SKILLS_MULTIPART_BUNDLE_FIELD => {
                if zip_upload.is_some() || !files.is_empty() {
                    return Err(SkillsApiError::BadRequest {
                        code: "mixed_upload_modes",
                        message: "multipart upload cannot mix zip archive and files[] parts"
                            .to_string(),
                    });
                }
                let file_name = field.file_name().map(str::to_string);
                let content_type = field.content_type().map(str::to_string);
                if !looks_like_zip_upload(file_name.as_deref(), content_type.as_deref()) {
                    return Err(SkillsApiError::BadRequest {
                        code: "invalid_zip_upload",
                        message:
                            "zip archive uploads must use a .zip filename or application/zip content type"
                                .to_string(),
                    });
                }
                let bytes = field.bytes().await.map_err(|error| {
                    invalid_multipart(format!("failed to read zip upload: {error}"))
                })?;
                zip_upload = Some(bytes.to_vec());
            }
            "files" | SKILLS_MULTIPART_FILES_FIELD => {
                if zip_upload.is_some() {
                    return Err(SkillsApiError::BadRequest {
                        code: "mixed_upload_modes",
                        message: "multipart upload cannot mix zip archive and files[] parts"
                            .to_string(),
                    });
                }
                let relative_path = field.file_name().map(str::to_string).ok_or_else(|| {
                    SkillsApiError::BadRequest {
                        code: "missing_file_name",
                        message: "multipart file name is required for files[] uploads".to_string(),
                    }
                })?;
                let media_type = field.content_type().map(str::to_string);
                let contents = field.bytes().await.map_err(|error| {
                    invalid_multipart(format!("failed to read uploaded file bytes: {error}"))
                })?;
                files.push(UploadedSkillFile {
                    relative_path,
                    contents: contents.to_vec(),
                    media_type,
                });
            }
            _ => {
                let _ = field.bytes().await;
                return Err(unexpected_field(&format!(
                    "unexpected multipart field '{field_name}'"
                )));
            }
        }
    }

    let tenant_id = tenant_id.ok_or_else(|| SkillsApiError::BadRequest {
        code: "missing_target_tenant",
        message: "target tenant id is required".to_string(),
    })?;
    let upload = match (zip_upload, files.is_empty()) {
        (Some(bytes), true) => SkillUpload::Zip(bytes),
        (None, false) => SkillUpload::Files(files),
        (Some(_), false) => {
            return Err(SkillsApiError::BadRequest {
                code: "mixed_upload_modes",
                message: "multipart upload cannot mix zip archive and files[] parts".to_string(),
            });
        }
        (None, true) => {
            return Err(SkillsApiError::BadRequest {
                code: "missing_upload_parts",
                message:
                    "multipart upload must contain either one zip archive or one or more files[] parts"
                        .to_string(),
            });
        }
    };

    Ok(ParsedSkillUpload { tenant_id, upload })
}

fn looks_like_zip_upload(file_name: Option<&str>, content_type: Option<&str>) -> bool {
    file_name.is_some_and(|name| name.to_ascii_lowercase().ends_with(".zip"))
        || content_type.is_some_and(|value| {
            value.eq_ignore_ascii_case("application/zip")
                || value.eq_ignore_ascii_case("application/x-zip-compressed")
        })
}

fn invalid_multipart(message: String) -> SkillsApiError {
    SkillsApiError::BadRequest {
        code: "invalid_multipart",
        message,
    }
}

fn unexpected_field(message: &str) -> SkillsApiError {
    SkillsApiError::BadRequest {
        code: "unexpected_field",
        message: message.to_string(),
    }
}

fn bad_text_field(field: &str, error: MultipartError) -> SkillsApiError {
    SkillsApiError::BadRequest {
        code: "invalid_multipart",
        message: format!("failed to read '{field}' field: {error}"),
    }
}

fn build_mutation_response(
    value: SkillCreateResult,
) -> Result<SkillMutationResponse, SkillsApiError> {
    Ok(SkillMutationResponse {
        skill: SkillResponse {
            id: value.skill.skill_id.clone(),
            name: value.skill.name.clone(),
            short_description: value.skill.short_description.clone(),
            description: value.skill.description.clone().unwrap_or_default(),
            source: value.skill.source.clone(),
            latest_version: value.skill.latest_version.clone(),
            default_version: value.skill.default_version.clone(),
            has_code_files: value.skill.has_code_files,
            created_at: value.skill.created_at.to_rfc3339(),
            updated_at: value.skill.updated_at.to_rfc3339(),
        },
        version: SkillVersionResponse {
            skill_id: value.version.skill_id.clone(),
            version: value.version.version.clone(),
            version_number: value.version.version_number,
            name: value.version.name.clone(),
            short_description: value.version.short_description.clone(),
            description: value.version.description.clone(),
            interface: serialize_optional_json(value.version.interface.as_ref())?,
            dependencies: serialize_optional_json(value.version.dependencies.as_ref())?,
            policy: serialize_optional_json(value.version.policy.as_ref())?,
            deprecated: value.version.deprecated,
            files: value
                .version
                .file_manifest
                .iter()
                .map(|entry| SkillVersionFileResponse {
                    path: entry.relative_path.clone(),
                    size_bytes: entry.size_bytes,
                })
                .collect(),
            created_at: value.version.created_at.to_rfc3339(),
        },
        warnings: value
            .warnings
            .into_iter()
            .map(|warning| SkillWarningResponse {
                kind: format!("{:?}", warning.kind),
                path: warning.path,
                message: warning.message,
            })
            .collect(),
    })
}

fn serialize_optional_json<T: serde::Serialize>(
    value: Option<&T>,
) -> Result<Option<Value>, SkillsApiError> {
    value
        .map(|value| {
            serde_json::to_value(value).map_err(|error| SkillsApiError::Internal {
                code: "skills_internal_error",
                message: format!("failed to serialize skills response payload: {error}"),
            })
        })
        .transpose()
}
