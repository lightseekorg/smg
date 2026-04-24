use axum::{
    extract::{multipart::Field, Multipart, Path, Query, State},
    http::{
        header::{self, HeaderMap, HeaderValue},
        StatusCode,
    },
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use openai_protocol::skills::{
    SkillGetQuery, SkillMutationResponse, SkillPatchRequest, SkillResponse,
    SkillVersionFileResponse, SkillVersionPatchRequest, SkillVersionRef, SkillVersionResponse,
    SkillVersionsListQuery, SkillVersionsListResponse, SkillWarningResponse, SkillsErrorBody,
    SkillsErrorEnvelope, SkillsListQuery, SkillsListResponse, SKILLS_MULTIPART_BUNDLE_FIELD,
    SKILLS_MULTIPART_FILES_FIELD, SKILLS_MULTIPART_FILE_FIELD, SKILLS_MULTIPART_TENANT_ID_FIELD,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use smg_skills::{
    CreateSkillRequest, CreateSkillVersionRequest, SkillCreateResult, SkillServiceError,
    SkillUpload, SkillUploadLimits, SkillsAdminOperation, UpdateSkillRequest,
    UpdateSkillVersionRequest, UploadedSkillFile,
};
use tracing::error;

use crate::{middleware::resolve_admin_target_tenant_key, server::AppState};

const DEFAULT_SKILLS_LIST_LIMIT: usize = 100;
const MAX_SKILLS_LIST_LIMIT: usize = 100;
const MAX_TENANT_ID_FIELD_BYTES: usize = 4096;

#[derive(Debug)]
struct ParsedSkillUpload {
    tenant_id: String,
    upload: SkillUpload,
}

#[derive(Debug)]
enum SkillsApiError {
    BadRequest { code: &'static str, message: String },
    Forbidden { code: &'static str, message: String },
    NotFound { code: &'static str, message: String },
    Conflict { code: &'static str, message: String },
    Internal { code: &'static str, message: String },
}

impl IntoResponse for SkillsApiError {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            Self::BadRequest { code, message } => (StatusCode::BAD_REQUEST, code, message),
            Self::Forbidden { code, message } => (StatusCode::FORBIDDEN, code, message),
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
            SkillServiceError::SkillVersionNotFound { .. } => Self::NotFound {
                code: "skill_version_not_found",
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
            SkillServiceError::CannotDeleteDefaultVersion { .. } => Self::Conflict {
                code: "default_version_conflict",
                message: error.to_string(),
            },
            SkillServiceError::BundleBuild(_)
            | SkillServiceError::BlobStore(_)
            | SkillServiceError::Store(_)
            | SkillServiceError::MissingComponent { .. } => internal_skill_service_error(
                error,
                "skills_internal_error",
                "skills request failed due to an internal error",
            ),
            SkillServiceError::MissingDefaultVersion { .. } => internal_skill_service_error(
                error,
                "default_version_missing",
                "skills metadata is internally inconsistent",
            ),
        }
    }
}

fn internal_skill_service_error(
    error: SkillServiceError,
    code: &'static str,
    public_message: &'static str,
) -> SkillsApiError {
    error!(error = %error, "skills API internal error");
    SkillsApiError::Internal {
        code,
        message: public_message.to_string(),
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

pub async fn list_skills(
    State(state): State<std::sync::Arc<AppState>>,
    Query(query): Query<SkillsListQuery>,
    headers: HeaderMap,
) -> Response {
    match list_skills_impl(state, query, &headers).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

pub async fn get_skill(
    State(state): State<std::sync::Arc<AppState>>,
    Path(skill_id): Path<String>,
    Query(query): Query<SkillGetQuery>,
    headers: HeaderMap,
) -> Response {
    get_skill_impl(state, skill_id, query, &headers)
        .await
        .unwrap_or_else(|error| error.into_response())
}

pub async fn list_skill_versions(
    State(state): State<std::sync::Arc<AppState>>,
    Path(skill_id): Path<String>,
    Query(query): Query<SkillVersionsListQuery>,
    headers: HeaderMap,
) -> Response {
    list_skill_versions_impl(state, skill_id, query, &headers)
        .await
        .unwrap_or_else(|error| error.into_response())
}

pub async fn get_skill_version(
    State(state): State<std::sync::Arc<AppState>>,
    Path((skill_id, version)): Path<(String, String)>,
    Query(query): Query<SkillGetQuery>,
    headers: HeaderMap,
) -> Response {
    get_skill_version_impl(state, skill_id, version, query, &headers)
        .await
        .unwrap_or_else(|error| error.into_response())
}

pub async fn patch_skill(
    State(state): State<std::sync::Arc<AppState>>,
    Path(skill_id): Path<String>,
    Query(query): Query<SkillGetQuery>,
    Json(body): Json<SkillPatchRequest>,
) -> Response {
    match patch_skill_impl(state, skill_id, query, body).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

pub async fn patch_skill_version(
    State(state): State<std::sync::Arc<AppState>>,
    Path((skill_id, version)): Path<(String, String)>,
    Query(query): Query<SkillGetQuery>,
    Json(body): Json<SkillVersionPatchRequest>,
) -> Response {
    match patch_skill_version_impl(state, skill_id, version, query, body).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

pub async fn delete_skill(
    State(state): State<std::sync::Arc<AppState>>,
    Path(skill_id): Path<String>,
    Query(query): Query<SkillGetQuery>,
) -> Response {
    match delete_skill_impl(state, skill_id, query).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(error) => error.into_response(),
    }
}

pub async fn delete_skill_version(
    State(state): State<std::sync::Arc<AppState>>,
    Path((skill_id, version)): Path<(String, String)>,
    Query(query): Query<SkillGetQuery>,
) -> Response {
    match delete_skill_version_impl(state, skill_id, version, query).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
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
    ensure_admin_operation_allowed(&state, SkillsAdminOperation::CreateAnyTenant)?;
    let parsed = parse_skill_upload(multipart, skill_service.upload_limits()).await?;
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
    ensure_admin_operation_allowed(&state, SkillsAdminOperation::CreateAnyTenant)?;
    let parsed = parse_skill_upload(multipart, skill_service.upload_limits()).await?;
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

async fn patch_skill_impl(
    state: std::sync::Arc<AppState>,
    skill_id: String,
    query: SkillGetQuery,
    body: SkillPatchRequest,
) -> Result<SkillResponse, SkillsApiError> {
    let skill_service =
        state
            .context
            .skill_service
            .clone()
            .ok_or_else(|| SkillsApiError::Internal {
                code: "skills_not_configured",
                message: "skills service is not configured".to_string(),
            })?;
    ensure_admin_operation_allowed(&state, SkillsAdminOperation::UpdateAnyTenant)?;
    let tenant_id = resolve_target_tenant_id(query.tenant_id.as_deref())?;
    let updated = skill_service
        .update_skill(UpdateSkillRequest {
            tenant_id,
            skill_id,
            default_version_ref: skill_version_ref_to_string(&body.default_version),
        })
        .await
        .map_err(SkillsApiError::from)?;
    Ok(skill_response_from_record(&updated))
}

async fn patch_skill_version_impl(
    state: std::sync::Arc<AppState>,
    skill_id: String,
    version: String,
    query: SkillGetQuery,
    body: SkillVersionPatchRequest,
) -> Result<SkillVersionResponse, SkillsApiError> {
    let skill_service =
        state
            .context
            .skill_service
            .clone()
            .ok_or_else(|| SkillsApiError::Internal {
                code: "skills_not_configured",
                message: "skills service is not configured".to_string(),
            })?;
    ensure_admin_operation_allowed(&state, SkillsAdminOperation::UpdateAnyTenant)?;
    let tenant_id = resolve_target_tenant_id(query.tenant_id.as_deref())?;
    let updated = skill_service
        .update_skill_version(UpdateSkillVersionRequest {
            tenant_id,
            skill_id,
            version_ref: version,
            deprecated: body.deprecated,
        })
        .await
        .map_err(SkillsApiError::from)?;
    skill_version_response_from_record(&updated)
}

async fn delete_skill_impl(
    state: std::sync::Arc<AppState>,
    skill_id: String,
    query: SkillGetQuery,
) -> Result<(), SkillsApiError> {
    let skill_service =
        state
            .context
            .skill_service
            .clone()
            .ok_or_else(|| SkillsApiError::Internal {
                code: "skills_not_configured",
                message: "skills service is not configured".to_string(),
            })?;
    ensure_admin_operation_allowed(&state, SkillsAdminOperation::DeleteAnyTenant)?;
    let tenant_id = resolve_target_tenant_id(query.tenant_id.as_deref())?;
    skill_service
        .delete_skill(&tenant_id, &skill_id)
        .await
        .map_err(SkillsApiError::from)
}

async fn delete_skill_version_impl(
    state: std::sync::Arc<AppState>,
    skill_id: String,
    version: String,
    query: SkillGetQuery,
) -> Result<(), SkillsApiError> {
    let skill_service =
        state
            .context
            .skill_service
            .clone()
            .ok_or_else(|| SkillsApiError::Internal {
                code: "skills_not_configured",
                message: "skills service is not configured".to_string(),
            })?;
    ensure_admin_operation_allowed(&state, SkillsAdminOperation::DeleteAnyTenant)?;
    let tenant_id = resolve_target_tenant_id(query.tenant_id.as_deref())?;
    skill_service
        .delete_skill_version(&tenant_id, &skill_id, &version)
        .await
        .map(|_| ())
        .map_err(SkillsApiError::from)
}

async fn list_skills_impl(
    state: std::sync::Arc<AppState>,
    query: SkillsListQuery,
    request_headers: &HeaderMap,
) -> Result<Response, SkillsApiError> {
    let skill_service =
        state
            .context
            .skill_service
            .clone()
            .ok_or_else(|| SkillsApiError::Internal {
                code: "skills_not_configured",
                message: "skills service is not configured".to_string(),
            })?;
    ensure_admin_operation_allowed(&state, SkillsAdminOperation::ReadAnyTenant)?;
    let tenant_id = resolve_target_tenant_id(query.tenant_id.as_deref())?;
    let limit = validate_list_limit(query.limit)?;
    let mut records = skill_service
        .list_skills(&tenant_id, query.source.as_deref(), query.name.as_deref())
        .await
        .map_err(SkillsApiError::from)?;
    records.sort_by(|left, right| {
        right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| left.skill_id.cmp(&right.skill_id))
    });
    let start = skills_list_start_index(&records, query.after.as_deref())?;

    let etag = build_list_etag(
        &query,
        records
            .iter()
            .map(|record| (record.skill_id.clone(), record.updated_at.to_rfc3339())),
    )?;
    let last_modified = list_last_modified(records.iter().map(|record| record.updated_at));
    if is_not_modified(request_headers, &etag, last_modified) {
        return not_modified_response(&etag, last_modified);
    }

    let page = paginate_from_start(records, start, limit);
    let body = SkillsListResponse {
        object: "list".to_string(),
        first_id: page
            .items
            .first()
            .map(|record| SkillsListCursor::from_record(record).encode())
            .transpose()?,
        last_id: page
            .items
            .last()
            .map(|record| SkillsListCursor::from_record(record).encode())
            .transpose()?,
        has_more: page.has_more,
        data: page.items.iter().map(skill_response_from_record).collect(),
    };

    with_cache_headers(etag, last_modified, Json(body))
}

async fn get_skill_impl(
    state: std::sync::Arc<AppState>,
    skill_id: String,
    query: SkillGetQuery,
    request_headers: &HeaderMap,
) -> Result<Response, SkillsApiError> {
    let skill_service =
        state
            .context
            .skill_service
            .clone()
            .ok_or_else(|| SkillsApiError::Internal {
                code: "skills_not_configured",
                message: "skills service is not configured".to_string(),
            })?;
    ensure_admin_operation_allowed(&state, SkillsAdminOperation::ReadAnyTenant)?;
    let tenant_id = resolve_target_tenant_id(query.tenant_id.as_deref())?;
    let record = skill_service
        .get_skill(&tenant_id, &skill_id)
        .await
        .map_err(SkillsApiError::from)?;
    let response_body = skill_response_from_record(&record);
    let etag = build_resource_etag(&response_body, false)?;
    let last_modified = record.updated_at;
    if is_not_modified(request_headers, &etag, last_modified) {
        return not_modified_response(&etag, last_modified);
    }

    with_cache_headers(etag, last_modified, Json(response_body))
}

async fn list_skill_versions_impl(
    state: std::sync::Arc<AppState>,
    skill_id: String,
    query: SkillVersionsListQuery,
    request_headers: &HeaderMap,
) -> Result<Response, SkillsApiError> {
    let skill_service =
        state
            .context
            .skill_service
            .clone()
            .ok_or_else(|| SkillsApiError::Internal {
                code: "skills_not_configured",
                message: "skills service is not configured".to_string(),
            })?;
    ensure_admin_operation_allowed(&state, SkillsAdminOperation::ReadAnyTenant)?;
    let tenant_id = resolve_target_tenant_id(query.tenant_id.as_deref())?;
    let limit = validate_list_limit(query.limit)?;
    let skill = skill_service
        .get_skill(&tenant_id, &skill_id)
        .await
        .map_err(SkillsApiError::from)?;
    let mut versions = skill_service
        .list_skill_versions(&tenant_id, &skill_id, query.include_deprecated)
        .await
        .map_err(SkillsApiError::from)?;
    versions.sort_by(|left, right| {
        right
            .version_number
            .cmp(&left.version_number)
            .then_with(|| left.version.cmp(&right.version))
    });
    let start = start_index_from_id_cursor(&versions, query.after.as_deref(), |record| {
        record.version.as_str()
    })?;

    let etag = build_list_etag(
        &query,
        versions.iter().map(|record| {
            (
                record.version.clone(),
                format!("{}:{}", record.created_at.to_rfc3339(), record.deprecated),
            )
        }),
    )?;
    let last_modified = skill.updated_at;
    if is_not_modified(request_headers, &etag, last_modified) {
        return not_modified_response(&etag, last_modified);
    }

    let page = paginate_from_start(versions, start, limit);
    let body = SkillVersionsListResponse {
        object: "list".to_string(),
        first_id: page.items.first().map(|record| record.version.clone()),
        last_id: page.items.last().map(|record| record.version.clone()),
        has_more: page.has_more,
        data: page
            .items
            .iter()
            .map(skill_version_response_from_record)
            .collect::<Result<Vec<_>, _>>()?,
    };

    with_cache_headers(etag, last_modified, Json(body))
}

async fn get_skill_version_impl(
    state: std::sync::Arc<AppState>,
    skill_id: String,
    version: String,
    query: SkillGetQuery,
    request_headers: &HeaderMap,
) -> Result<Response, SkillsApiError> {
    let skill_service =
        state
            .context
            .skill_service
            .clone()
            .ok_or_else(|| SkillsApiError::Internal {
                code: "skills_not_configured",
                message: "skills service is not configured".to_string(),
            })?;
    ensure_admin_operation_allowed(&state, SkillsAdminOperation::ReadAnyTenant)?;
    let tenant_id = resolve_target_tenant_id(query.tenant_id.as_deref())?;
    let skill = skill_service
        .get_skill(&tenant_id, &skill_id)
        .await
        .map_err(SkillsApiError::from)?;
    let record = skill_service
        .get_skill_version(&tenant_id, &skill_id, &version)
        .await
        .map_err(SkillsApiError::from)?;
    let response_body = skill_version_response_from_record(&record)?;
    let etag = build_resource_etag(&response_body, false)?;
    let last_modified = skill.updated_at;
    if is_not_modified(request_headers, &etag, last_modified) {
        return not_modified_response(&etag, last_modified);
    }

    with_cache_headers(etag, last_modified, Json(response_body))
}

struct Page<T> {
    items: Vec<T>,
    has_more: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct SkillsListCursorPayload {
    updated_at: String,
    skill_id: String,
}

#[derive(Debug)]
struct SkillsListCursor {
    updated_at: chrono::DateTime<chrono::Utc>,
    skill_id: String,
}

impl SkillsListCursor {
    fn from_record(record: &smg_skills::SkillRecord) -> Self {
        Self {
            updated_at: record.updated_at,
            skill_id: record.skill_id.clone(),
        }
    }

    fn encode(&self) -> Result<String, SkillsApiError> {
        let payload = SkillsListCursorPayload {
            updated_at: self.updated_at.to_rfc3339(),
            skill_id: self.skill_id.clone(),
        };
        let bytes = serde_json::to_vec(&payload).map_err(|error| SkillsApiError::Internal {
            code: "skills_internal_error",
            message: format!("failed to serialize skills list cursor: {error}"),
        })?;
        Ok(URL_SAFE_NO_PAD.encode(bytes))
    }

    fn decode(raw: &str) -> Result<Self, SkillsApiError> {
        let bytes = URL_SAFE_NO_PAD
            .decode(raw)
            .map_err(|_| invalid_after_cursor("after cursor is not valid base64url".to_string()))?;
        let payload: SkillsListCursorPayload = serde_json::from_slice(&bytes).map_err(|_| {
            invalid_after_cursor("after cursor is not valid skills cursor JSON".to_string())
        })?;
        let updated_at = chrono::DateTime::parse_from_rfc3339(&payload.updated_at)
            .map_err(|_| {
                invalid_after_cursor(
                    "after cursor does not contain a valid RFC3339 timestamp".to_string(),
                )
            })?
            .with_timezone(&chrono::Utc);
        if payload.skill_id.trim().is_empty() {
            return Err(invalid_after_cursor(
                "after cursor does not contain a skill id".to_string(),
            ));
        }
        Ok(Self {
            updated_at,
            skill_id: payload.skill_id,
        })
    }
}

fn resolve_target_tenant_id(raw_tenant_id: Option<&str>) -> Result<String, SkillsApiError> {
    let tenant_id = raw_tenant_id.ok_or_else(|| SkillsApiError::BadRequest {
        code: "missing_target_tenant",
        message: "target tenant id is required".to_string(),
    })?;
    resolve_admin_target_tenant_key(tenant_id)
        .map(|tenant_key| tenant_key.to_string())
        .map_err(|error| SkillsApiError::BadRequest {
            code: "missing_target_tenant",
            message: error.to_string(),
        })
}

fn ensure_admin_operation_allowed(
    state: &AppState,
    operation: SkillsAdminOperation,
) -> Result<(), SkillsApiError> {
    let allowed = state
        .context
        .router_config
        .skills
        .as_ref()
        .is_some_and(|skills| skills.admin.allowed_operations.contains(&operation));
    if allowed {
        return Ok(());
    }

    Err(SkillsApiError::Forbidden {
        code: "skills_operation_not_allowed",
        message: format!("skills admin operation '{operation:?}' is not allowed"),
    })
}

fn validate_list_limit(limit: Option<u32>) -> Result<usize, SkillsApiError> {
    let limit = limit.unwrap_or(DEFAULT_SKILLS_LIST_LIMIT as u32);
    if limit == 0 || limit as usize > MAX_SKILLS_LIST_LIMIT {
        return Err(SkillsApiError::BadRequest {
            code: "invalid_limit",
            message: format!("limit must be between 1 and {MAX_SKILLS_LIST_LIMIT}"),
        });
    }
    Ok(limit as usize)
}

fn normalized_after_cursor(after: Option<&str>) -> Option<&str> {
    after.map(str::trim).filter(|value| !value.is_empty())
}

fn skills_list_start_index(
    items: &[smg_skills::SkillRecord],
    after: Option<&str>,
) -> Result<usize, SkillsApiError> {
    let Some(after) = normalized_after_cursor(after) else {
        return Ok(0);
    };
    let cursor = SkillsListCursor::decode(after)?;
    Ok(items
        .iter()
        .position(|record| skill_record_is_after_cursor(record, &cursor))
        .unwrap_or(items.len()))
}

fn skill_record_is_after_cursor(
    record: &smg_skills::SkillRecord,
    cursor: &SkillsListCursor,
) -> bool {
    record.updated_at < cursor.updated_at
        || (record.updated_at == cursor.updated_at && record.skill_id > cursor.skill_id)
}

fn start_index_from_id_cursor<T, F>(
    items: &[T],
    after: Option<&str>,
    id_of: F,
) -> Result<usize, SkillsApiError>
where
    F: Fn(&T) -> &str,
{
    if let Some(after) = normalized_after_cursor(after) {
        items
            .iter()
            .position(|item| id_of(item) == after)
            .map(|index| index + 1)
            .ok_or_else(|| invalid_after_cursor(format!("after cursor '{after}' was not found")))
    } else {
        Ok(0)
    }
}

fn paginate_from_start<T>(items: Vec<T>, start: usize, limit: usize) -> Page<T> {
    let total = items.len();
    let items = items
        .into_iter()
        .skip(start)
        .take(limit)
        .collect::<Vec<_>>();
    Page {
        has_more: total > start.saturating_add(limit),
        items,
    }
}

fn list_last_modified<I>(timestamps: I) -> chrono::DateTime<chrono::Utc>
where
    I: Iterator<Item = chrono::DateTime<chrono::Utc>>,
{
    timestamps
        .max()
        .unwrap_or_else(|| chrono::DateTime::<chrono::Utc>::from(std::time::SystemTime::UNIX_EPOCH))
}

fn build_list_etag<Q, I>(query: &Q, records: I) -> Result<String, SkillsApiError>
where
    Q: Serialize,
    I: IntoIterator<Item = (String, String)>,
{
    let mut hasher = Sha256::new();
    hasher.update(
        serde_json::to_vec(query).map_err(|error| SkillsApiError::Internal {
            code: "skills_internal_error",
            message: format!("failed to serialize skills query for ETag: {error}"),
        })?,
    );
    for (id, updated_at) in records {
        hasher.update(id.as_bytes());
        hasher.update([0]);
        hasher.update(updated_at.as_bytes());
        hasher.update([0xff]);
    }
    Ok(format!("W/\"{:x}\"", hasher.finalize()))
}

fn build_resource_etag<T: Serialize>(value: &T, weak: bool) -> Result<String, SkillsApiError> {
    let digest =
        Sha256::digest(
            serde_json::to_vec(value).map_err(|error| SkillsApiError::Internal {
                code: "skills_internal_error",
                message: format!("failed to serialize skills response payload: {error}"),
            })?,
        );
    Ok(if weak {
        format!("W/\"{digest:x}\"")
    } else {
        format!("\"{digest:x}\"")
    })
}

fn is_not_modified(
    request_headers: &HeaderMap,
    etag: &str,
    last_modified: chrono::DateTime<chrono::Utc>,
) -> bool {
    if let Some(if_none_match) = request_headers.get(header::IF_NONE_MATCH) {
        return if_none_match_matches(if_none_match, etag);
    }

    request_headers
        .get(header::IF_MODIFIED_SINCE)
        .and_then(|value| value.to_str().ok())
        .and_then(parse_http_date)
        .is_some_and(|value| last_modified <= value)
}

fn if_none_match_matches(value: &HeaderValue, etag: &str) -> bool {
    value.to_str().ok().is_some_and(|value| {
        value.split(',').map(str::trim).any(|candidate| {
            candidate == "*" || strip_weak_prefix(candidate) == strip_weak_prefix(etag)
        })
    })
}

fn strip_weak_prefix(etag: &str) -> &str {
    etag.strip_prefix("W/").unwrap_or(etag)
}

fn parse_http_date(value: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc2822(value)
        .ok()
        .map(|value| value.with_timezone(&chrono::Utc))
}

fn not_modified_response(
    etag: &str,
    last_modified: chrono::DateTime<chrono::Utc>,
) -> Result<Response, SkillsApiError> {
    Ok((
        StatusCode::NOT_MODIFIED,
        cache_headers(etag, last_modified)?,
    )
        .into_response())
}

fn with_cache_headers<T: IntoResponse>(
    etag: String,
    last_modified: chrono::DateTime<chrono::Utc>,
    body: T,
) -> Result<Response, SkillsApiError> {
    Ok((cache_headers(&etag, last_modified)?, body).into_response())
}

fn cache_headers(
    etag: &str,
    last_modified: chrono::DateTime<chrono::Utc>,
) -> Result<HeaderMap, SkillsApiError> {
    let mut headers = HeaderMap::with_capacity(2);
    headers.insert(
        header::ETAG,
        HeaderValue::from_str(etag).map_err(|error| SkillsApiError::Internal {
            code: "skills_internal_error",
            message: format!("failed to build ETag header: {error}"),
        })?,
    );
    headers.insert(
        header::LAST_MODIFIED,
        HeaderValue::from_str(
            &last_modified
                .format("%a, %d %b %Y %H:%M:%S GMT")
                .to_string(),
        )
        .map_err(|error| SkillsApiError::Internal {
            code: "skills_internal_error",
            message: format!("failed to build Last-Modified header: {error}"),
        })?,
    );
    Ok(headers)
}

async fn parse_skill_upload(
    mut multipart: Multipart,
    limits: SkillUploadLimits,
) -> Result<ParsedSkillUpload, SkillsApiError> {
    let mut tenant_id = None;
    let mut zip_upload = None;
    let mut files = Vec::new();
    let mut total_file_bytes = 0usize;

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
                let value = read_text_field_limited(
                    field,
                    SKILLS_MULTIPART_TENANT_ID_FIELD,
                    MAX_TENANT_ID_FIELD_BYTES,
                )
                .await?;
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
                let bytes =
                    read_field_bytes_limited(field, limits.max_upload_size_bytes, "zip upload")
                        .await?;
                zip_upload = Some(bytes);
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
                if files.len() >= limits.max_files_per_version {
                    return Err(SkillsApiError::BadRequest {
                        code: "invalid_skill_bundle",
                        message: format!(
                            "skill bundle contains more than {} regular files",
                            limits.max_files_per_version
                        ),
                    });
                }
                let remaining_upload_bytes = limits
                    .max_upload_size_bytes
                    .saturating_sub(total_file_bytes);
                let max_field_bytes = remaining_upload_bytes.min(limits.max_file_size_bytes);
                let contents =
                    read_field_bytes_limited(field, max_field_bytes, "uploaded skill file").await?;
                total_file_bytes =
                    total_file_bytes
                        .checked_add(contents.len())
                        .ok_or_else(|| SkillsApiError::BadRequest {
                            code: "skill_upload_too_large",
                            message: format!(
                                "skill upload exceeds maximum total size of {} bytes",
                                limits.max_upload_size_bytes
                            ),
                        })?;
                files.push(UploadedSkillFile {
                    relative_path,
                    contents,
                    media_type,
                });
            }
            _ => {
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

fn invalid_after_cursor(message: String) -> SkillsApiError {
    SkillsApiError::BadRequest {
        code: "invalid_after_cursor",
        message,
    }
}

fn unexpected_field(message: &str) -> SkillsApiError {
    SkillsApiError::BadRequest {
        code: "unexpected_field",
        message: message.to_string(),
    }
}

async fn read_text_field_limited(
    field: Field<'_>,
    field_name: &'static str,
    max_bytes: usize,
) -> Result<String, SkillsApiError> {
    let bytes = read_field_bytes_limited(field, max_bytes, field_name).await?;
    String::from_utf8(bytes).map_err(|error| SkillsApiError::BadRequest {
        code: "invalid_multipart",
        message: format!("failed to read '{field_name}' field as UTF-8: {error}"),
    })
}

async fn read_field_bytes_limited(
    mut field: Field<'_>,
    max_bytes: usize,
    field_description: &str,
) -> Result<Vec<u8>, SkillsApiError> {
    let mut bytes = Vec::new();
    while let Some(chunk) = field.chunk().await.map_err(|error| {
        invalid_multipart(format!("failed to read {field_description}: {error}"))
    })? {
        let next_len = bytes
            .len()
            .checked_add(chunk.len())
            .ok_or_else(|| upload_too_large(field_description, max_bytes))?;
        if next_len > max_bytes {
            return Err(upload_too_large(field_description, max_bytes));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

fn upload_too_large(field_description: &str, max_bytes: usize) -> SkillsApiError {
    SkillsApiError::BadRequest {
        code: "skill_upload_too_large",
        message: format!("{field_description} exceeds maximum size of {max_bytes} bytes"),
    }
}

fn build_mutation_response(
    value: SkillCreateResult,
) -> Result<SkillMutationResponse, SkillsApiError> {
    Ok(SkillMutationResponse {
        skill: skill_response_from_record(&value.skill),
        version: skill_version_response_from_record(&value.version)?,
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

fn skill_response_from_record(record: &smg_skills::SkillRecord) -> SkillResponse {
    SkillResponse {
        id: record.skill_id.clone(),
        name: record.name.clone(),
        short_description: record.short_description.clone(),
        description: record.description.clone().unwrap_or_default(),
        source: record.source.clone(),
        latest_version: record.latest_version.clone(),
        default_version: record.default_version.clone(),
        has_code_files: record.has_code_files,
        created_at: record.created_at.to_rfc3339(),
        updated_at: record.updated_at.to_rfc3339(),
    }
}

fn skill_version_ref_to_string(version: &SkillVersionRef) -> String {
    match version {
        SkillVersionRef::Latest => "latest".to_string(),
        SkillVersionRef::Integer(value) => value.to_string(),
        SkillVersionRef::Timestamp(value) => value.clone(),
    }
}

fn skill_version_response_from_record(
    record: &smg_skills::SkillVersionRecord,
) -> Result<SkillVersionResponse, SkillsApiError> {
    Ok(SkillVersionResponse {
        skill_id: record.skill_id.clone(),
        version: record.version.clone(),
        version_number: record.version_number,
        name: record.name.clone(),
        short_description: record.short_description.clone(),
        description: record.description.clone(),
        interface: serialize_optional_json(record.interface.as_ref())?,
        dependencies: serialize_optional_json(record.dependencies.as_ref())?,
        policy: serialize_optional_json(record.policy.as_ref())?,
        deprecated: record.deprecated,
        files: record
            .file_manifest
            .iter()
            .map(|entry| SkillVersionFileResponse {
                path: entry.relative_path.clone(),
                size_bytes: entry.size_bytes,
            })
            .collect(),
        created_at: record.created_at.to_rfc3339(),
    })
}

fn serialize_optional_json<T: Serialize>(
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

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;
    use smg_blob_storage::BlobStoreError;

    use super::*;

    #[tokio::test]
    async fn internal_skill_service_errors_hide_backend_details(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let response =
            SkillsApiError::from(SkillServiceError::BlobStore(BlobStoreError::Operation {
                operation: "put",
                message: "oracle dsn and filesystem path".to_string(),
            }))
            .into_response();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let bytes = to_bytes(response.into_body(), usize::MAX).await?;
        let body: Value = serde_json::from_slice(&bytes)?;
        assert_eq!(body["error"]["code"], "skills_internal_error");
        assert_eq!(
            body["error"]["message"],
            "skills request failed due to an internal error"
        );
        assert!(!body.to_string().contains("oracle dsn"));

        Ok(())
    }
}
