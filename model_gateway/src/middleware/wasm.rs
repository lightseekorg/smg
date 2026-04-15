//! WASM plugin middleware.
//!
//! Dispatches request and response through every WASM module attached at
//! the corresponding `Middleware::OnRequest` / `Middleware::OnResponse`
//! point. Streaming responses skip the OnResponse phase to avoid buffering
//! arbitrary bodies into memory.

use std::{sync::Arc, time::Duration};

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use tracing::{error, warn};

use super::request_id::{generate_request_id, RequestId};
use crate::{
    server::AppState,
    wasm::{
        module::{MiddlewareAttachPoint, WasmModuleAttachPoint},
        spec::{
            apply_modify_action_to_headers, build_wasm_headers_from_axum_headers,
            smg::gateway::middleware_types::{
                Action, Request as WasmRequest, Response as WasmResponse,
            },
        },
        types::WasmComponentInput,
    },
};

pub async fn wasm_middleware(
    State(app_state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // Check if WASM is enabled
    if !app_state.context.router_config.enable_wasm {
        return next.run(request).await;
    }

    // Get WASM manager
    let wasm_manager = match &app_state.context.wasm_manager {
        Some(manager) => manager,
        None => {
            return next.run(request).await;
        }
    };

    // Get request ID from extensions or generate one
    let request_id = request
        .extensions()
        .get::<RequestId>()
        .map(|r| r.0.clone())
        .unwrap_or_else(|| generate_request_id(request.uri().path()));

    // ===== OnRequest Phase =====
    let on_request_attach_point =
        WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnRequest);

    let modules_on_request =
        match wasm_manager.get_modules_by_attach_point(on_request_attach_point.clone()) {
            Ok(modules) => modules,
            Err(e) => {
                error!("Failed to get WASM modules for OnRequest: {}", e);
                return next.run(request).await;
            }
        };

    let response = if modules_on_request.is_empty() {
        next.run(request).await
    } else {
        // Decompose request to preserve extensions across reconstruction
        let (parts, body) = request.into_parts();
        let method = parts.method;
        let uri = parts.uri;
        let mut headers = parts.headers;
        let extensions = parts.extensions;

        let max_body_size = wasm_manager.get_max_body_size();
        let body_bytes = match axum::body::to_bytes(body, max_body_size).await {
            Ok(bytes) => bytes.to_vec(),
            Err(e) => {
                error!("Failed to read request body for WASM processing: {}", e);
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": format!("Failed to read request body: {e}")})),
                )
                    .into_response();
            }
        };

        // Process each OnRequest module
        let mut modified_body = body_bytes;

        // Pre-compute strings once before the loop to avoid repeated allocations
        let method_str = method.to_string();
        let path_str = uri.path().to_string();
        let query_str = uri.query().unwrap_or("").to_string();

        for module in modules_on_request {
            let wasm_headers = build_wasm_headers_from_axum_headers(&headers);
            let wasm_request = WasmRequest {
                method: method_str.clone(),
                path: path_str.clone(),
                query: query_str.clone(),
                headers: wasm_headers,
                body: modified_body.clone(),
                request_id: request_id.clone(),
                now_epoch_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_else(|_| Duration::from_millis(0))
                    .as_millis() as u64,
            };

            let action = match wasm_manager
                .execute_module_for_attach_point(
                    &module,
                    on_request_attach_point.clone(),
                    WasmComponentInput::MiddlewareRequest(wasm_request),
                )
                .await
            {
                Some(action) => action,
                None => continue,
            };

            match action {
                Action::Continue => {}
                Action::Reject(status) => {
                    return StatusCode::from_u16(status)
                        .unwrap_or(StatusCode::BAD_REQUEST)
                        .into_response();
                }
                Action::Modify(modify) => {
                    apply_modify_action_to_headers(&mut headers, &modify);
                    if let Some(body_bytes) = modify.body_replace {
                        modified_body = body_bytes;
                    }
                }
            }
        }

        // Reconstruct request with modifications, preserving original extensions
        let mut final_request = Request::builder()
            .method(method)
            .uri(uri)
            .body(Body::from(modified_body))
            .unwrap_or_else(|_| Request::new(Body::empty()));
        *final_request.headers_mut() = headers;
        *final_request.extensions_mut() = extensions;

        next.run(final_request).await
    };

    // ===== OnResponse Phase =====
    let on_response_attach_point =
        WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnResponse);

    let modules_on_response =
        match wasm_manager.get_modules_by_attach_point(on_response_attach_point.clone()) {
            Ok(modules) => modules,
            Err(e) => {
                error!("Failed to get WASM modules for OnResponse: {}", e);
                return response;
            }
        };
    if modules_on_response.is_empty() {
        return response;
    }

    // Skip WASM OnResponse processing for streaming responses to avoid
    // buffering the entire stream into memory (breaks SSE, causes OOM on large streams).
    let is_streaming = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("text/event-stream") || ct.contains("application/x-ndjson"))
        || response
            .headers()
            .get(header::TRANSFER_ENCODING)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|te| te.contains("chunked"));
    if is_streaming {
        warn!("Skipping WASM OnResponse for streaming response; OnResponse modules do not apply to streaming");
        return response;
    }

    // Extract response data once before processing modules
    let mut status = response.status();
    let mut headers = response.headers().clone();
    let max_body_size = wasm_manager.get_max_body_size();
    let mut body_bytes = match axum::body::to_bytes(response.into_body(), max_body_size).await {
        Ok(bytes) => bytes.to_vec(),
        Err(e) => {
            error!("Failed to read response body for WASM processing: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Failed to read response body"})),
            )
                .into_response();
        }
    };

    // Process each OnResponse module
    for module in modules_on_response {
        let wasm_headers = build_wasm_headers_from_axum_headers(&headers);
        let wasm_response = WasmResponse {
            status: status.as_u16(),
            headers: wasm_headers,
            body: body_bytes.clone(),
        };

        let action = match wasm_manager
            .execute_module_for_attach_point(
                &module,
                on_response_attach_point.clone(),
                WasmComponentInput::MiddlewareResponse(wasm_response),
            )
            .await
        {
            Some(action) => action,
            None => continue,
        };

        match action {
            Action::Continue => {}
            Action::Reject(status_code) => {
                status = StatusCode::from_u16(status_code).unwrap_or(StatusCode::BAD_REQUEST);
                let mut final_response = Response::builder()
                    .status(status)
                    .body(Body::from(body_bytes))
                    .unwrap_or_else(|_| Response::new(Body::empty()));
                *final_response.headers_mut() = headers;
                return final_response;
            }
            Action::Modify(modify) => {
                if let Some(new_status) = modify.status {
                    status = StatusCode::from_u16(new_status).unwrap_or(status);
                }
                apply_modify_action_to_headers(&mut headers, &modify);
                if let Some(new_body) = modify.body_replace {
                    body_bytes = new_body;
                }
            }
        }
    }

    // Reconstruct final response with all modifications
    let mut final_response = Response::builder()
        .status(status)
        .body(Body::from(body_bytes))
        .unwrap_or_else(|_| Response::new(Body::empty()));
    *final_response.headers_mut() = headers;
    final_response
}
