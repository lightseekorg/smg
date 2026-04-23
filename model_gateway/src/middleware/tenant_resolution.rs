//! Tenant resolution and request-meta insertion for serving paths.

use std::{net::SocketAddr, sync::Arc};

use axum::{
    body::Body,
    extract::{connect_info::ConnectInfo, Request, State},
    http::{header::InvalidHeaderName, HeaderMap, HeaderName, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use smg_skills::TenantAliasStore;
use tracing::error;

use crate::{
    config::{RouterConfig, TenantResolutionConfig},
    tenant::{canonical_tenant_key, DataPlaneCaller, RouteRequestMeta, TenantIdentity, TenantKey},
};

#[derive(Clone)]
pub struct TenantResolutionState {
    trust_tenant_header: bool,
    trusted_tenant_header_name: HeaderName,
    tenant_alias_store: Option<Arc<dyn TenantAliasStore>>,
}

impl TenantResolutionState {
    pub fn new(config: &RouterConfig) -> Result<Self, InvalidHeaderName> {
        Self::from_config(&config.tenant_resolution)
    }

    pub fn from_config(config: &TenantResolutionConfig) -> Result<Self, InvalidHeaderName> {
        let trusted_tenant_header_name: HeaderName = config.tenant_header_name.parse()?;

        Ok(Self {
            trust_tenant_header: config.trust_tenant_header,
            trusted_tenant_header_name,
            tenant_alias_store: None,
        })
    }

    #[must_use]
    pub fn with_tenant_alias_store(
        mut self,
        tenant_alias_store: Option<Arc<dyn TenantAliasStore>>,
    ) -> Self {
        self.tenant_alias_store = tenant_alias_store;
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RouteRequestMetaError {
    #[error("tenant alias lookup failed: {0}")]
    TenantAliasLookup(#[from] smg_skills::SkillsStoreError),
}

fn resolve_raw_tenant_key(state: &TenantResolutionState, request: &Request<Body>) -> TenantKey {
    if let Some(caller) = request.extensions().get::<DataPlaneCaller>() {
        return caller.tenant_key().clone();
    }
    if state.trust_tenant_header {
        if let Some(tenant_id) = extract_trusted_tenant_id(state, request.headers()) {
            return canonical_tenant_key(TenantIdentity::Header(Arc::from(tenant_id)));
        }
    }

    if let Some(ConnectInfo(addr)) = request.extensions().get::<ConnectInfo<SocketAddr>>() {
        return canonical_tenant_key(TenantIdentity::IpAddress(addr.ip()));
    }

    canonical_tenant_key(TenantIdentity::Anonymous)
}

async fn resolve_tenant_key(
    state: &TenantResolutionState,
    raw_tenant_key: TenantKey,
) -> Result<TenantKey, RouteRequestMetaError> {
    let Some(alias_store) = &state.tenant_alias_store else {
        return Ok(raw_tenant_key);
    };
    let Some(record) = alias_store
        .get_tenant_alias(raw_tenant_key.as_str())
        .await?
    else {
        return Ok(raw_tenant_key);
    };
    if record
        .expires_at
        .is_some_and(|expires_at| expires_at <= chrono::Utc::now())
    {
        return Ok(raw_tenant_key);
    }
    Ok(TenantKey::from(record.canonical_tenant_id))
}

fn extract_trusted_tenant_id<'a>(
    state: &TenantResolutionState,
    headers: &'a HeaderMap,
) -> Option<&'a str> {
    headers
        .get(&state.trusted_tenant_header_name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

pub fn resolve_route_request_meta<'a>(
    state: &'a TenantResolutionState,
    request: &Request<Body>,
) -> impl std::future::Future<Output = Result<RouteRequestMeta, RouteRequestMetaError>> + Send + 'a
{
    let raw_tenant_key = resolve_raw_tenant_key(state, request);
    async move {
        Ok(RouteRequestMeta::new(
            resolve_tenant_key(state, raw_tenant_key).await?,
        ))
    }
}

pub async fn route_request_meta_middleware(
    State(state): State<TenantResolutionState>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    let request_meta = match resolve_route_request_meta(&state, &request).await {
        Ok(request_meta) => request_meta,
        Err(error) => {
            error!(error = %error, "failed to resolve tenant metadata");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to resolve tenant metadata",
            )
                .into_response();
        }
    };
    request.extensions_mut().insert(request_meta);
    next.run(request).await
}

/// Backward-compatible name while the older skills task breakdown still uses
/// the original tenant-resolution terminology.
pub async fn ordinary_tenant_resolution_middleware(
    state: State<TenantResolutionState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    route_request_meta_middleware(state, request, next).await
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use axum::{
        body::Body,
        extract::connect_info::ConnectInfo,
        http::{header, HeaderValue, Request, StatusCode},
        middleware::from_fn_with_state,
        response::IntoResponse,
        routing::get,
        Router,
    };
    use smg_skills::{
        InMemorySkillStore, SkillsStoreError, SkillsStoreResult, TenantAliasRecord,
        TenantAliasStore,
    };
    use tower::ServiceExt;

    use super::*;
    use crate::{
        config::{PolicyConfig, RouterConfig, RoutingMode},
        middleware::TenantRequestMeta,
        tenant::DEFAULT_TENANT_HEADER_NAME,
    };

    #[derive(Debug)]
    struct FailingTenantAliasStore;

    #[async_trait]
    impl TenantAliasStore for FailingTenantAliasStore {
        async fn put_tenant_alias(&self, _record: TenantAliasRecord) -> SkillsStoreResult<()> {
            Err(SkillsStoreError::Storage("put not supported".to_string()))
        }

        async fn get_tenant_alias(
            &self,
            _alias_tenant_id: &str,
        ) -> SkillsStoreResult<Option<TenantAliasRecord>> {
            Err(SkillsStoreError::Storage(
                "oracle backend connection details".to_string(),
            ))
        }

        async fn delete_tenant_alias(&self, _alias_tenant_id: &str) -> SkillsStoreResult<bool> {
            Err(SkillsStoreError::Storage(
                "delete not supported".to_string(),
            ))
        }
    }

    fn resolution_state() -> TenantResolutionState {
        TenantResolutionState::new(&RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["http://worker1:8000".to_string()],
            },
            PolicyConfig::Random,
        ))
        .unwrap()
    }

    #[tokio::test]
    async fn request_meta_prefers_authenticated_data_plane_identity() {
        let state = resolution_state();
        let mut request = Request::builder().uri("/").body(Body::empty()).unwrap();
        request
            .extensions_mut()
            .insert(DataPlaneCaller::new(TenantKey::from("auth:b3c2")));
        request
            .extensions_mut()
            .insert(ConnectInfo("127.0.0.1:8080".parse::<SocketAddr>().unwrap()));

        let request_meta = resolve_route_request_meta(&state, &request).await.unwrap();
        assert_eq!(request_meta.tenant_key().as_str(), "auth:b3c2");
    }

    #[tokio::test]
    async fn request_meta_uses_trusted_header_when_enabled() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["http://worker1:8000".to_string()],
            },
            PolicyConfig::Random,
        );
        config.tenant_resolution.trust_tenant_header = true;
        let state = TenantResolutionState::new(&config).unwrap();

        let request = Request::builder()
            .uri("/")
            .header(DEFAULT_TENANT_HEADER_NAME, "team-red")
            .body(Body::empty())
            .unwrap();

        let request_meta = resolve_route_request_meta(&state, &request).await.unwrap();
        assert_eq!(request_meta.tenant_key().as_str(), "header:team-red");
    }

    #[tokio::test]
    async fn request_meta_falls_back_to_client_ip() {
        let state = resolution_state();
        let mut request = Request::builder().uri("/").body(Body::empty()).unwrap();
        request.extensions_mut().insert(ConnectInfo(
            "203.0.113.42:443".parse::<SocketAddr>().unwrap(),
        ));

        let request_meta = resolve_route_request_meta(&state, &request).await.unwrap();
        assert_eq!(request_meta.tenant_key().as_str(), "ip:203.0.113.42");
    }

    #[tokio::test]
    async fn request_meta_falls_back_to_anonymous_without_identity_sources() {
        let state = resolution_state();
        let request = Request::builder().uri("/").body(Body::empty()).unwrap();

        let request_meta = resolve_route_request_meta(&state, &request).await.unwrap();
        assert_eq!(request_meta.tenant_key().as_str(), "anonymous");
    }

    #[tokio::test]
    async fn request_meta_uses_active_tenant_alias() {
        let store = Arc::new(InMemorySkillStore::default());
        store
            .put_tenant_alias(TenantAliasRecord {
                alias_tenant_id: "auth:new-key".to_string(),
                canonical_tenant_id: "auth:old-key".to_string(),
                created_at: chrono::Utc::now(),
                expires_at: None,
            })
            .await
            .unwrap();
        let state = resolution_state().with_tenant_alias_store(Some(store));
        let mut request = Request::builder().uri("/").body(Body::empty()).unwrap();
        request
            .extensions_mut()
            .insert(DataPlaneCaller::new(TenantKey::from("auth:new-key")));

        let request_meta = resolve_route_request_meta(&state, &request).await.unwrap();
        assert_eq!(request_meta.tenant_key().as_str(), "auth:old-key");
    }

    #[tokio::test]
    async fn request_meta_ignores_missing_and_expired_tenant_aliases() {
        let store = Arc::new(InMemorySkillStore::default());
        store
            .put_tenant_alias(TenantAliasRecord {
                alias_tenant_id: "auth:expired-key".to_string(),
                canonical_tenant_id: "auth:old-key".to_string(),
                created_at: chrono::Utc::now(),
                expires_at: Some(chrono::Utc::now() - chrono::Duration::seconds(1)),
            })
            .await
            .unwrap();
        let state = resolution_state().with_tenant_alias_store(Some(store));
        let mut expired = Request::builder().uri("/").body(Body::empty()).unwrap();
        expired
            .extensions_mut()
            .insert(DataPlaneCaller::new(TenantKey::from("auth:expired-key")));
        let mut missing = Request::builder().uri("/").body(Body::empty()).unwrap();
        missing
            .extensions_mut()
            .insert(DataPlaneCaller::new(TenantKey::from("auth:missing-key")));

        let expired_meta = resolve_route_request_meta(&state, &expired).await.unwrap();
        let missing_meta = resolve_route_request_meta(&state, &missing).await.unwrap();
        assert_eq!(expired_meta.tenant_key().as_str(), "auth:expired-key");
        assert_eq!(missing_meta.tenant_key().as_str(), "auth:missing-key");
    }

    #[tokio::test]
    async fn middleware_attaches_request_meta_extension() {
        async fn handler(request: Request<Body>) -> impl IntoResponse {
            request
                .extensions()
                .get::<TenantRequestMeta>()
                .map(|meta| meta.tenant_key().to_string())
                .unwrap_or_else(|| "missing".to_string())
        }

        let app = Router::new()
            .route("/", get(handler))
            .route_layer(from_fn_with_state(
                resolution_state(),
                route_request_meta_middleware,
            ));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header(
                        header::AUTHORIZATION,
                        HeaderValue::from_static("Bearer ignored"),
                    )
                    .extension(DataPlaneCaller::new(TenantKey::from("auth:tenant-a")))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(std::str::from_utf8(&body).unwrap(), "auth:tenant-a");
    }

    #[tokio::test]
    async fn middleware_hides_tenant_alias_lookup_errors_from_clients() {
        async fn handler() -> impl IntoResponse {
            StatusCode::OK
        }

        let app = Router::new()
            .route("/", get(handler))
            .route_layer(from_fn_with_state(
                resolution_state().with_tenant_alias_store(Some(Arc::new(FailingTenantAliasStore))),
                route_request_meta_middleware,
            ));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .extension(DataPlaneCaller::new(TenantKey::from("auth:tenant-a")))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            std::str::from_utf8(&body).unwrap(),
            "failed to resolve tenant metadata"
        );
    }
}
