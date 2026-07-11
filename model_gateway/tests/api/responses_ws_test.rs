//! WebSocket Responses protocol tests.
//!
//! Ported from the reference stub-based suite. These tests implement a fake
//! [`WsResponsesExecutor`] and drive [`serve_responses_ws_with_config`] over an
//! in-process WebSocket, asserting on the emitted frames. No real engine / GPU
//! is involved.
//!
//! Adapted to the current API surface:
//! - `RouterTrait::route_responses_ws(req: Request<Body>, model_id: &str) -> Response`
//!   performs the WS upgrade itself (there is no `supports_responses_ws` hook and
//!   no `(headers, socket)` variant on main), so the stub router upgrades the
//!   connection and then calls `serve_responses_ws_with_config`.
//! - The `/v1/responses` server handler requires a `?model=` query parameter, so
//!   all connect URLs carry `?model=mock-model`.
//! - `ResponseOutputItem::Message` / `ResponseInputOutputItem::Message` gained a
//!   `phase` field, and `ResponseOutputItem::FunctionToolCall.id` is now
//!   `Option<String>`; the stubs supply `phase: None` / `id: Some(..)`.

// Test-only relaxations. The free helper functions here (server spawn, frame
// receive) are not `#[test]`-annotated, so clippy's `allow-*-in-tests`
// heuristic does not exempt them; panic-on-failure is the intended behavior in
// a test harness. Mirrors `tests/common/mock_openai_server.rs`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::disallowed_methods,
    clippy::allow_attributes
)]

use std::{
    collections::HashMap,
    fmt,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        FromRequestParts,
    },
    http::{HeaderMap, Request, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::{SinkExt, StreamExt};
use openai_protocol::responses::{
    ResponseContentPart, ResponseInputOutputItem, ResponseOutputItem, ResponseStatus,
    ResponsesRequest, ResponsesResponse,
};
use smg::{
    routers::{
        factory::router_ids,
        router_manager::RouterManager,
        ws_responses::{
            serve_responses_ws_with_config, CachedWsResponse, WsClientError,
            WsResponseCreateOptions, WsResponsesExecutor, WsRuntimeConfig,
        },
        RouterTrait,
    },
    worker::WorkerRegistry,
};
use tokio::{
    net::TcpListener,
    sync::{mpsc, Notify},
};
use tokio_tungstenite::connect_async;
use tower::ServiceExt;

use crate::common::test_app::{create_test_app_context, create_test_app_with_context};

#[derive(Clone)]
struct StubWsExecutor {
    gate: Option<Arc<Notify>>,
}

impl StubWsExecutor {
    fn immediate() -> Self {
        Self { gate: None }
    }

    fn gated(gate: Arc<Notify>) -> Self {
        Self { gate: Some(gate) }
    }
}

#[derive(Clone)]
struct DelayedReturnWsExecutor {
    return_delay: Duration,
}

impl DelayedReturnWsExecutor {
    fn new(return_delay: Duration) -> Self {
        Self { return_delay }
    }
}

#[async_trait]
impl WsResponsesExecutor for StubWsExecutor {
    async fn execute_response_create(
        &self,
        _headers: HeaderMap,
        request: ResponsesRequest,
        _options: WsResponseCreateOptions,
        _cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::Sender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        let model = request.model.clone();
        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": "resp_ws_test",
                "object": "response",
                "status": "in_progress",
                "model": model,
                "output": []
            }
        });
        let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));

        if let Some(gate) = &self.gate {
            gate.notified().await;
        }

        let output_text = "stub websocket output";
        let response = ResponsesResponse::builder("resp_ws_test", request.model.clone())
            .copy_from_request(&request)
            .status(ResponseStatus::Completed)
            .output(vec![ResponseOutputItem::Message {
                id: "msg_ws_test".to_string(),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: output_text.to_string(),
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
                phase: None,
            }])
            .build();

        let completed = serde_json::json!({
            "type": "response.completed",
            "response": response,
        });
        let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

        Ok(CachedWsResponse {
            response: ResponsesResponse::builder("resp_ws_test", request.model.clone())
                .copy_from_request(&request)
                .status(ResponseStatus::Completed)
                .output(vec![ResponseOutputItem::Message {
                    id: "msg_ws_test".to_string(),
                    role: "assistant".to_string(),
                    content: vec![ResponseContentPart::OutputText {
                        text: output_text.to_string(),
                        annotations: vec![],
                        logprobs: None,
                    }],
                    status: "completed".to_string(),
                    phase: None,
                }])
                .build(),
            input_items: vec![ResponseInputOutputItem::Message {
                id: "msg_user_ws_test".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "Hello websocket".to_string(),
                }],
                status: Some("completed".to_string()),
                phase: None,
            }],
        })
    }
}

#[async_trait]
impl WsResponsesExecutor for DelayedReturnWsExecutor {
    async fn execute_response_create(
        &self,
        _headers: HeaderMap,
        request: ResponsesRequest,
        _options: WsResponseCreateOptions,
        _cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::Sender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        let output_text = "delayed websocket output";
        let response = ResponsesResponse::builder("resp_ws_delayed", request.model.clone())
            .copy_from_request(&request)
            .status(ResponseStatus::Completed)
            .output(vec![ResponseOutputItem::Message {
                id: "msg_ws_delayed".to_string(),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: output_text.to_string(),
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
                phase: None,
            }])
            .build();

        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": response.id,
                "object": "response",
                "status": "in_progress",
                "model": request.model,
                "output": []
            }
        });
        let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));

        let completed = serde_json::json!({
            "type": "response.completed",
            "response": response.clone(),
        });
        let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

        tokio::time::sleep(self.return_delay).await;

        Ok(CachedWsResponse {
            response,
            input_items: vec![],
        })
    }
}

#[derive(Clone, Default)]
struct FunctionCallWsExecutor;

#[async_trait]
impl WsResponsesExecutor for FunctionCallWsExecutor {
    async fn execute_response_create(
        &self,
        _headers: HeaderMap,
        request: ResponsesRequest,
        _options: WsResponseCreateOptions,
        _cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::Sender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        let response_id = "resp_ws_tool_test";
        let item_id = "fc_ws_test";
        let call_id = "call_ws_test";
        let tool_name = "search_repo";
        let arguments = r#"{"query":"fizz_buzz"}"#;
        let model = request.model.clone();

        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "status": "in_progress",
                "model": model,
                "output": []
            }
        });
        let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));

        let output_item_added = serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "function_call",
                "call_id": call_id,
                "name": tool_name,
                "status": "in_progress",
                "arguments": ""
            }
        });
        let _ = outbound_tx.try_send(Message::Text(output_item_added.to_string().into()));

        let args_delta = serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": item_id,
            "delta": arguments
        });
        let _ = outbound_tx.try_send(Message::Text(args_delta.to_string().into()));

        let args_done = serde_json::json!({
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "item_id": item_id,
            "arguments": arguments
        });
        let _ = outbound_tx.try_send(Message::Text(args_done.to_string().into()));

        let output_item_done = serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "function_call",
                "call_id": call_id,
                "name": tool_name,
                "status": "completed",
                "arguments": arguments
            }
        });
        let _ = outbound_tx.try_send(Message::Text(output_item_done.to_string().into()));

        let response = ResponsesResponse::builder(response_id, request.model.clone())
            .copy_from_request(&request)
            .status(ResponseStatus::Completed)
            .output(vec![ResponseOutputItem::FunctionToolCall {
                id: Some(item_id.to_string()),
                call_id: call_id.to_string(),
                name: tool_name.to_string(),
                arguments: arguments.to_string(),
                output: None,
                status: "completed".to_string(),
            }])
            .build();

        let completed = serde_json::json!({
            "type": "response.completed",
            "response": response,
        });
        let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

        Ok(CachedWsResponse {
            response: ResponsesResponse::builder(response_id, request.model.clone())
                .copy_from_request(&request)
                .status(ResponseStatus::Completed)
                .output(vec![ResponseOutputItem::FunctionToolCall {
                    id: Some(item_id.to_string()),
                    call_id: call_id.to_string(),
                    name: tool_name.to_string(),
                    arguments: arguments.to_string(),
                    output: None,
                    status: "completed".to_string(),
                }])
                .build(),
            input_items: vec![ResponseInputOutputItem::Message {
                id: "msg_user_ws_tool_test".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "Call the search tool.".to_string(),
                }],
                status: Some("completed".to_string()),
                phase: None,
            }],
        })
    }
}

#[derive(Clone, Default)]
struct FailedResponseWsExecutor;

#[async_trait]
impl WsResponsesExecutor for FailedResponseWsExecutor {
    async fn execute_response_create(
        &self,
        _headers: HeaderMap,
        request: ResponsesRequest,
        _options: WsResponseCreateOptions,
        cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::Sender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        if let Some(previous_id) = request.previous_response_id.as_deref() {
            if cached_response
                .as_ref()
                .is_some_and(|cached| cached.response.id == previous_id)
            {
                return Ok(CachedWsResponse {
                    response: ResponsesResponse::builder(
                        "resp_ws_unexpected_cached_reuse",
                        request.model.clone(),
                    )
                    .copy_from_request(&request)
                    .status(ResponseStatus::Completed)
                    .output(vec![ResponseOutputItem::Message {
                        id: "msg_ws_unexpected_cached_reuse".to_string(),
                        role: "assistant".to_string(),
                        content: vec![ResponseContentPart::OutputText {
                            text: "unexpected cached continuation".to_string(),
                            annotations: vec![],
                            logprobs: None,
                        }],
                        status: "completed".to_string(),
                        phase: None,
                    }])
                    .build(),
                    input_items: vec![],
                });
            }

            return Err(WsClientError::new(
                "previous_response_not_found",
                format!(
                    "Previous response '{previous_id}' was not found in the current session or durable storage."
                ),
            )
            .with_param("previous_response_id"));
        }

        let response = ResponsesResponse::builder("resp_ws_failed", request.model.clone())
            .copy_from_request(&request)
            .status(ResponseStatus::Failed)
            .output(vec![])
            .build();
        let response_model = request.model.clone();

        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": "resp_ws_failed",
                "object": "response",
                "status": "in_progress",
                "model": response_model,
                "output": []
            }
        });
        let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));

        let completed = serde_json::json!({
            "type": "response.completed",
            "response": response.clone(),
        });
        let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

        Ok(CachedWsResponse {
            response,
            input_items: vec![],
        })
    }
}

/// Panics on the FIRST `response.create`, then succeeds on every later call.
/// Used to prove a panicking handler releases the in-flight slot: the catch in
/// the session loop routes the panic through the error path instead of wedging
/// the connection so all future turns reject with `concurrent_response_create`.
#[derive(Clone, Default)]
struct PanicOnceWsExecutor {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl WsResponsesExecutor for PanicOnceWsExecutor {
    async fn execute_response_create(
        &self,
        _headers: HeaderMap,
        request: ResponsesRequest,
        _options: WsResponseCreateOptions,
        _cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::Sender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        // Panic on the first call (returns 0); succeed on every later call.
        assert!(
            self.calls.fetch_add(1, Ordering::SeqCst) != 0,
            "simulated response.create handler panic"
        );

        let response = ResponsesResponse::builder("resp_ws_recovered", request.model.clone())
            .copy_from_request(&request)
            .status(ResponseStatus::Completed)
            .output(vec![ResponseOutputItem::Message {
                id: "msg_ws_recovered".to_string(),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: "recovered websocket output".to_string(),
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
                phase: None,
            }])
            .build();

        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": response.id,
                "object": "response",
                "status": "in_progress",
                "model": request.model,
                "output": []
            }
        });
        let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));

        let completed = serde_json::json!({
            "type": "response.completed",
            "response": response.clone(),
        });
        let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

        Ok(CachedWsResponse {
            response,
            input_items: vec![],
        })
    }
}

/// Three-call probe for the `Ok(status == Failed)` cache arm: call 0 is a
/// successful parent (cached), call 1 materializes an `Ok(Failed)` child (a
/// worker finish_reason=failed analogue, NOT an `Err`), and call 2 only succeeds
/// if the session still holds the parent in its cache. Used to prove a failed
/// child does not evict a still-valid `store:false` parent.
#[derive(Clone, Default)]
struct OkFailedChildWsExecutor {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl WsResponsesExecutor for OkFailedChildWsExecutor {
    async fn execute_response_create(
        &self,
        _headers: HeaderMap,
        request: ResponsesRequest,
        _options: WsResponseCreateOptions,
        cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::Sender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        let (response_id, status) = match self.calls.fetch_add(1, Ordering::SeqCst) {
            0 => ("resp_parent", ResponseStatus::Completed),
            1 => ("resp_failed_child", ResponseStatus::Failed),
            _ => {
                // Retry: resolve ONLY if the parent survived in the connection
                // cache (the session passes it as cached_response).
                let parent_alive = cached_response
                    .as_ref()
                    .is_some_and(|cached| cached.response.id == "resp_parent");
                if !parent_alive {
                    return Err(WsClientError::new(
                        "previous_response_not_found",
                        "cached parent was evicted by the failed child",
                    )
                    .with_param("previous_response_id"));
                }
                ("resp_retry", ResponseStatus::Completed)
            }
        };

        let response = ResponsesResponse::builder(response_id, request.model.clone())
            .copy_from_request(&request)
            .status(status.clone())
            .output(vec![])
            .build();
        let created = serde_json::json!({
            "type": "response.created",
            "response": { "id": response_id, "object": "response", "status": "in_progress", "model": request.model, "output": [] }
        });
        let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));
        let completed =
            serde_json::json!({ "type": "response.completed", "response": response.clone() });
        let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

        Ok(CachedWsResponse {
            response,
            input_items: vec![],
        })
    }
}

#[derive(Clone)]
struct StubWsRouter {
    executor: Arc<dyn WsResponsesExecutor>,
    runtime_config: WsRuntimeConfig,
}

impl fmt::Debug for StubWsRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("StubWsRouter")
    }
}

#[async_trait]
impl RouterTrait for StubWsRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Current `RouterTrait` performs the WS upgrade in `route_responses_ws`
    /// itself (there is no `(headers, socket)` variant), mirroring the real
    /// gRPC router. We split the request parts, extract the upgrade, then drive
    /// `serve_responses_ws_with_config`.
    async fn route_responses_ws(&self, req: Request<Body>, _model_id: &str) -> Response {
        let (mut parts, _body) = req.into_parts();
        let ws = match WebSocketUpgrade::from_request_parts(&mut parts, &()).await {
            Ok(ws) => ws,
            Err(e) => return e.into_response(),
        };
        let headers = parts.headers.clone();
        let executor = self.executor.clone();
        let runtime_config = self.runtime_config.clone();
        ws.on_upgrade(move |socket: WebSocket| async move {
            serve_responses_ws_with_config(socket, headers, executor, runtime_config).await;
        })
    }

    fn router_type(&self) -> &'static str {
        "stub-ws"
    }
}

/// A router that does NOT override `route_responses_ws`, so it inherits the
/// `RouterTrait` default (501 Not Implemented). Models a backend that does not
/// support the WebSocket Responses transport.
#[derive(Clone)]
struct UnsupportedWsRouter;

impl fmt::Debug for UnsupportedWsRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("UnsupportedWsRouter")
    }
}

#[async_trait]
impl RouterTrait for UnsupportedWsRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    // Intentionally no `route_responses_ws` override — uses the trait default.

    fn router_type(&self) -> &'static str {
        "unsupported-ws"
    }
}

async fn build_stub_app(executor: Arc<dyn WsResponsesExecutor>) -> axum::Router {
    build_stub_app_with_runtime_config(executor, WsRuntimeConfig::default()).await
}

async fn build_stub_app_with_runtime_config(
    executor: Arc<dyn WsResponsesExecutor>,
    runtime_config: WsRuntimeConfig,
) -> axum::Router {
    let ctx = create_test_app_context().await;
    let router = Arc::new(StubWsRouter {
        executor,
        runtime_config,
    });
    create_test_app_with_context(router, ctx)
}

async fn serve_app(app: axum::Router) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("ws://{addr}")
}

/// Build the `/v1/responses` WebSocket URL. The current server handler requires
/// a `?model=` query parameter on the upgrade request.
fn ws_endpoint(base_url: &str) -> String {
    format!("{base_url}/v1/responses?model=mock-model")
}

async fn recv_json(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> serde_json::Value {
    loop {
        let message = tokio::time::timeout(Duration::from_secs(3), socket.next())
            .await
            .expect("timed out waiting for websocket message")
            .expect("websocket stream ended")
            .expect("websocket receive failed");

        match message {
            tokio_tungstenite::tungstenite::Message::Text(text) => {
                return serde_json::from_str(text.as_ref()).expect("message should be valid JSON");
            }
            tokio_tungstenite::tungstenite::Message::Ping(_) => continue,
            tokio_tungstenite::tungstenite::Message::Pong(_) => continue,
            tokio_tungstenite::tungstenite::Message::Close(frame) => {
                panic!("unexpected websocket close frame: {frame:?}")
            }
            other => panic!("unexpected websocket message: {other:?}"),
        }
    }
}

async fn send_ws_request_and_collect(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    request: serde_json::Value,
) -> Vec<serde_json::Value> {
    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            request.to_string().into(),
        ))
        .await
        .unwrap();

    let mut events = Vec::new();
    loop {
        let event = recv_json(socket).await;
        let is_terminal = matches!(
            event["type"].as_str(),
            Some("response.completed") | Some("error")
        );
        events.push(event);
        if is_terminal {
            break;
        }
    }

    events
}

fn ws_create_request(response_fields: serde_json::Value) -> serde_json::Value {
    let serde_json::Value::Object(mut request) = response_fields else {
        panic!("response.create request fields must be a JSON object");
    };
    request.insert(
        "type".to_string(),
        serde_json::Value::String("response.create".to_string()),
    );
    serde_json::Value::Object(request)
}

fn ws_error_code(event: &serde_json::Value) -> &str {
    event
        .pointer("/error/code")
        .and_then(|value| value.as_str())
        .or_else(|| event.get("code").and_then(|value| value.as_str()))
        .unwrap_or("")
}

fn ws_error_message(event: &serde_json::Value) -> &str {
    event
        .pointer("/error/message")
        .and_then(|value| value.as_str())
        .or_else(|| event.get("message").and_then(|value| value.as_str()))
        .unwrap_or("")
}

fn ws_error_param(event: &serde_json::Value) -> Option<&str> {
    event
        .pointer("/error/param")
        .and_then(|value| value.as_str())
}

#[derive(Clone, Default)]
struct SemanticWsExecutor {
    durable_store: Arc<std::sync::Mutex<HashMap<String, CachedWsResponse>>>,
}

impl SemanticWsExecutor {
    fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl WsResponsesExecutor for SemanticWsExecutor {
    async fn execute_response_create(
        &self,
        _headers: HeaderMap,
        request: ResponsesRequest,
        options: WsResponseCreateOptions,
        cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::Sender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        if request.conversation.is_some() {
            return Err(WsClientError::new(
                "unsupported_parameter",
                "The `conversation` field is not supported in WebSocket Responses V1.",
            ));
        }

        if options.generate == Some(false) {
            let response_id = format!("resp_ws_{}", uuid::Uuid::now_v7().simple());
            let response = ResponsesResponse::builder(response_id.clone(), request.model.clone())
                .copy_from_request(&request)
                .status(ResponseStatus::Completed)
                .output(vec![])
                .build();

            let created = serde_json::json!({
                "type": "response.created",
                "response": {
                    "id": response_id.clone(),
                    "object": "response",
                    "status": "in_progress",
                    "model": request.model.clone(),
                    "output": []
                }
            });
            let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));

            let completed = serde_json::json!({
                "type": "response.completed",
                "response": response,
            });
            let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

            return Ok(CachedWsResponse {
                response: ResponsesResponse::builder(response_id, request.model.clone())
                    .copy_from_request(&request)
                    .status(ResponseStatus::Completed)
                    .output(vec![])
                    .build(),
                input_items: vec![ResponseInputOutputItem::Message {
                    id: "msg_user_ws_semantic".to_string(),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText {
                        text: "Hello websocket".to_string(),
                    }],
                    status: Some("completed".to_string()),
                    phase: None,
                }],
            });
        }

        let previous_response = if let Some(previous_id) = request.previous_response_id.as_deref() {
            if let Some(cached) = cached_response.filter(|cached| cached.response.id == previous_id)
            {
                Some(cached)
            } else {
                self.durable_store
                    .lock()
                    .unwrap()
                    .get(previous_id)
                    .cloned()
                    .ok_or_else(|| {
                        WsClientError::new(
                            "previous_response_not_found",
                            format!(
                                "Previous response '{previous_id}' was not found in the current session or durable storage."
                            ),
                        )
                        .with_param("previous_response_id")
                    })?
                    .into()
            }
        } else {
            None
        };

        let response_id = format!("resp_ws_{}", uuid::Uuid::now_v7().simple());
        let output_text = if previous_response.is_some() {
            "stub websocket continuation output"
        } else {
            "stub websocket output"
        };

        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "status": "in_progress",
                "model": request.model.clone(),
                "output": []
            }
        });
        let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));

        let response = ResponsesResponse::builder(response_id.clone(), request.model.clone())
            .copy_from_request(&request)
            .status(ResponseStatus::Completed)
            .output(vec![ResponseOutputItem::Message {
                id: "msg_ws_semantic".to_string(),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: output_text.to_string(),
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
                phase: None,
            }])
            .build();

        let completed = serde_json::json!({
            "type": "response.completed",
            "response": response,
        });
        let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

        let cached = CachedWsResponse {
            response: response.clone(),
            input_items: vec![ResponseInputOutputItem::Message {
                id: "msg_user_ws_semantic".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "Hello websocket".to_string(),
                }],
                status: Some("completed".to_string()),
                phase: None,
            }],
        };

        if request.store.unwrap_or(true) {
            self.durable_store
                .lock()
                .unwrap()
                .insert(cached.response.id.clone(), cached.clone());
        }

        Ok(cached)
    }
}

#[tokio::test]
async fn test_v1_responses_get_requires_websocket_upgrade() {
    // Current server requires a `model` query param and performs the WS upgrade
    // inside the router. A plain GET (with model, without upgrade headers) is
    // rejected by `WebSocketUpgrade::from_request_parts`: with no `Connection:
    // upgrade` header axum returns `400 Bad Request` deterministically (the
    // reference's `websocket_upgrade_required` body code does not exist on main).
    let app = build_stub_app(Arc::new(StubWsExecutor::immediate())).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/responses?model=mock-model")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "plain GET without upgrade headers must be rejected with 400, got {}",
        response.status()
    );
}

#[tokio::test]
async fn test_v1_responses_ws_unsupported_router_returns_501() {
    // A router that does not override `route_responses_ws` inherits the
    // `RouterTrait` default, which must reject the upgrade with 501 rather than
    // attempting (and failing) to serve the WebSocket transport.
    let ctx = create_test_app_context().await;
    let app = create_test_app_with_context(Arc::new(UnsupportedWsRouter), ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/responses?model=mock-model")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::NOT_IMPLEMENTED,
        "router without WS-responses support must return 501"
    );
}

#[tokio::test]
async fn test_v1_responses_ws_rejects_unknown_event_type() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            serde_json::json!({ "type": "response.delete" })
                .to_string()
                .into(),
        ))
        .await
        .unwrap();

    let event = recv_json(&mut socket).await;
    assert_eq!(event["type"], "error");
    assert_eq!(ws_error_code(&event), "unsupported_event");
}

#[tokio::test]
async fn test_v1_responses_ws_rejects_invalid_json() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            "{\"type\":\"response.create\"".into(),
        ))
        .await
        .unwrap();

    let event = recv_json(&mut socket).await;
    assert_eq!(event["type"], "error");
    assert_eq!(ws_error_code(&event), "invalid_json");
}

#[tokio::test]
async fn test_v1_responses_ws_rejects_binary_messages() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Binary(
            vec![0xde, 0xad, 0xbe, 0xef].into(),
        ))
        .await
        .unwrap();

    let event = recv_json(&mut socket).await;
    assert_eq!(event["type"], "error");
    assert_eq!(ws_error_code(&event), "unsupported_message_type");
}

#[tokio::test]
async fn test_v1_responses_ws_replies_to_ping_and_keeps_session_healthy() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    let ping_payload = vec![0x1, 0x2, 0x3, 0x4];
    socket
        .send(tokio_tungstenite::tungstenite::Message::Ping(
            ping_payload.clone().into(),
        ))
        .await
        .unwrap();

    let pong = tokio::time::timeout(Duration::from_secs(3), socket.next())
        .await
        .expect("timed out waiting for pong")
        .expect("websocket stream ended")
        .expect("websocket receive failed");

    match pong {
        tokio_tungstenite::tungstenite::Message::Pong(payload) => {
            assert_eq!(payload.as_ref(), ping_payload.as_slice());
        }
        other => panic!("expected pong after ping, got {other:?}"),
    }

    let events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Hello websocket after ping",
            "store": false
        })),
    )
    .await;

    let completed = events.last().unwrap();
    assert_eq!(completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_closes_when_session_lifetime_expires() {
    let url = serve_app(
        build_stub_app_with_runtime_config(
            Arc::new(StubWsExecutor::immediate()),
            WsRuntimeConfig {
                max_session_lifetime: Duration::from_millis(50),
            },
        )
        .await,
    )
    .await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    let error = recv_json(&mut socket).await;
    assert_eq!(error["type"], "error");
    assert_eq!(ws_error_code(&error), "websocket_connection_limit_reached");

    let close_message = tokio::time::timeout(Duration::from_secs(2), socket.next())
        .await
        .expect("timed out waiting for websocket close")
        .expect("websocket stream ended without close frame")
        .expect("websocket receive failed");

    match close_message {
        tokio_tungstenite::tungstenite::Message::Close(frame) => {
            let frame = frame.expect("expected server close frame");
            assert_eq!(
                frame.reason.to_string(),
                "Responses websocket connection limit reached (50 ms). Create a new websocket connection to continue."
            );
        }
        other => panic!("expected websocket close frame, got {other:?}"),
    }
}

#[tokio::test]
async fn test_v1_responses_ws_response_create_streams_events() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            ws_create_request(serde_json::json!({
                "model": "mock-model",
                "input": "Hello websocket",
                "store": false
            }))
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    let created = recv_json(&mut socket).await;
    let completed = recv_json(&mut socket).await;

    assert_eq!(created["type"], "response.created");
    assert_eq!(completed["type"], "response.completed");
    assert_eq!(
        completed["response"]["output"][0]["content"][0]["text"],
        "stub websocket output"
    );
}

#[tokio::test]
async fn test_v1_responses_ws_function_call_events_stream_cleanly() {
    let url = serve_app(build_stub_app(Arc::new(FunctionCallWsExecutor)).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    let events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Call the tool",
            "store": false
        })),
    )
    .await;

    let event_types: Vec<_> = events
        .iter()
        .map(|event| event["type"].as_str().unwrap_or(""))
        .collect();

    assert_eq!(
        event_types,
        vec![
            "response.created",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    assert_eq!(events[1]["item"]["type"], "function_call");
    assert_eq!(events[2]["delta"], r#"{"query":"fizz_buzz"}"#);
    assert_eq!(events[3]["arguments"], r#"{"query":"fizz_buzz"}"#);
    assert_eq!(events[4]["item"]["call_id"], "call_ws_test");
    assert_eq!(events[4]["item"]["name"], "search_repo");
    assert_eq!(
        events.last().unwrap()["response"]["output"][0]["call_id"],
        "call_ws_test"
    );
    assert_eq!(
        events.last().unwrap()["response"]["output"][0]["arguments"],
        r#"{"query":"fizz_buzz"}"#
    );
}

#[tokio::test]
async fn test_v1_responses_ws_accepts_top_level_response_create_payload() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    let events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Top level websocket request",
            "store": false,
            "tools": []
        })),
    )
    .await;

    let completed = events.last().unwrap();
    assert_eq!(events[0]["type"], "response.created");
    assert_eq!(completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_accepts_structured_input_items() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    let request = serde_json::json!({
        "type": "response.create",
        "model": "mock-model",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Say hello from a structured input."}]
        }],
        "store": false
    });

    let events = send_ws_request_and_collect(&mut socket, request).await;
    let completed = events.last().unwrap();

    assert_eq!(events[0]["type"], "response.created");
    assert_eq!(completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_rejects_second_inflight_request() {
    let gate = Arc::new(Notify::new());
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::gated(gate.clone()))).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    let request = ws_create_request(serde_json::json!({
        "model": "mock-model",
        "input": "Hello websocket",
        "store": false
    }));

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            request.to_string().into(),
        ))
        .await
        .unwrap();

    let created = recv_json(&mut socket).await;
    assert_eq!(created["type"], "response.created");

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            request.to_string().into(),
        ))
        .await
        .unwrap();

    let error = recv_json(&mut socket).await;
    assert_eq!(error["type"], "error");
    assert_eq!(ws_error_code(&error), "concurrent_response_create");

    gate.notify_waiters();
    let completed = recv_json(&mut socket).await;
    assert_eq!(completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_via_router_manager_streams_events() {
    let ctx = create_test_app_context().await;
    let manager = Arc::new(RouterManager::new(
        Arc::new(WorkerRegistry::new()),
        reqwest::Client::new(),
    ));
    manager.register_router(
        router_ids::GRPC_REGULAR,
        Arc::new(StubWsRouter {
            executor: Arc::new(StubWsExecutor::immediate()),
            runtime_config: WsRuntimeConfig::default(),
        }),
    );

    let app = create_test_app_with_context(manager as Arc<dyn RouterTrait>, ctx);
    let url = serve_app(app).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            ws_create_request(serde_json::json!({
                "model": "mock-model",
                "input": "Hello websocket",
                "store": false
            }))
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    let created = recv_json(&mut socket).await;
    let completed = recv_json(&mut socket).await;

    assert_eq!(created["type"], "response.created");
    assert_eq!(completed["type"], "response.completed");
    assert_eq!(
        completed["response"]["output"][0]["content"][0]["text"],
        "stub websocket output"
    );
}

#[tokio::test]
async fn test_v1_responses_ws_same_connection_store_false_continuation_completes() {
    let url = serve_app(build_stub_app(Arc::new(SemanticWsExecutor::new())).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    let first_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "First websocket turn",
            "store": false
        })),
    )
    .await;
    let first_completed = first_events.last().unwrap();
    assert_eq!(first_completed["type"], "response.completed");

    let response_id = first_completed["response"]["id"]
        .as_str()
        .expect("completed response should include id")
        .to_string();

    let second_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Follow up websocket turn",
            "previous_response_id": response_id,
            "store": false
        })),
    )
    .await;

    let second_completed = second_events.last().unwrap();
    assert_eq!(second_completed["type"], "response.completed");

    let second_response_id = second_completed["response"]["id"]
        .as_str()
        .expect("completed response should include id")
        .to_string();

    let third_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Third websocket turn",
            "previous_response_id": second_response_id,
            "store": false
        })),
    )
    .await;

    let third_completed = third_events.last().unwrap();
    assert_eq!(third_completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_allows_immediate_follow_up_after_completed_event() {
    let url = serve_app(
        build_stub_app(Arc::new(DelayedReturnWsExecutor::new(
            Duration::from_millis(20),
        )))
        .await,
    )
    .await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    let first_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "First delayed websocket turn",
            "store": false
        })),
    )
    .await;
    let first_completed = first_events.last().unwrap();
    assert_eq!(first_completed["type"], "response.completed");

    let second_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Immediate follow-up websocket turn",
            "store": false
        })),
    )
    .await;

    let second_completed = second_events.last().unwrap();
    assert_eq!(second_completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_store_true_continuation_survives_reconnect() {
    let executor = Arc::new(SemanticWsExecutor::new());
    let url = serve_app(build_stub_app(executor).await).await;

    let (mut first_socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();
    let first_events = send_ws_request_and_collect(
        &mut first_socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Persist this websocket turn",
            "store": true
        })),
    )
    .await;
    let first_completed = first_events.last().unwrap();
    assert_eq!(first_completed["type"], "response.completed");
    let response_id = first_completed["response"]["id"]
        .as_str()
        .expect("completed response should include id")
        .to_string();
    drop(first_socket);

    let (mut second_socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();
    let second_events = send_ws_request_and_collect(
        &mut second_socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Reconnect follow up websocket turn",
            "previous_response_id": response_id,
            "store": false
        })),
    )
    .await;

    let second_completed = second_events.last().unwrap();
    assert_eq!(second_completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_missing_previous_response_errors() {
    let url = serve_app(build_stub_app(Arc::new(SemanticWsExecutor::new())).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    let events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Missing previous response id",
            "previous_response_id": "resp_missing_ws",
            "store": false
        })),
    )
    .await;

    let error = events.last().unwrap();
    assert_eq!(error["type"], "error");
    assert_eq!(ws_error_code(error), "previous_response_not_found");
    assert_eq!(ws_error_param(error), Some("previous_response_id"));
}

#[tokio::test]
async fn test_v1_responses_ws_rejects_unsupported_parameters() {
    let url = serve_app(build_stub_app(Arc::new(SemanticWsExecutor::new())).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    let background_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Background websocket request",
            "background": true
        })),
    )
    .await;
    assert_eq!(
        background_events.last().unwrap()["type"],
        "response.completed"
    );

    let events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Conversation websocket request",
            "conversation": "conv_test_123"
        })),
    )
    .await;
    let error = events.last().unwrap();
    assert_eq!(error["type"], "error");
    assert_eq!(ws_error_code(error), "unsupported_parameter");
}

#[tokio::test]
async fn test_v1_responses_ws_accepts_generate_false_warmup() {
    let url = serve_app(build_stub_app(Arc::new(SemanticWsExecutor::new())).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    let warmup_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Warm up websocket request",
            "generate": false,
            "store": false
        })),
    )
    .await;

    let completed = warmup_events.last().unwrap();
    assert_eq!(completed["type"], "response.completed");
    assert_eq!(
        completed["response"]["output"]
            .as_array()
            .expect("warmup output should be an array")
            .len(),
        0
    );
}

/// A continuation that fails for its OWN reason (here an unsupported
/// `conversation` parameter) must NOT evict the still-valid cached parent: a
/// `store: false` parent is chainable only through the connection cache, so a
/// subsequent retry of the same `previous_response_id` must still resolve.
/// Regression for the cached-parent-eviction review (ws_responses.rs).
#[tokio::test]
async fn test_v1_responses_ws_keeps_cached_response_after_failed_continuation() {
    let url = serve_app(build_stub_app(Arc::new(SemanticWsExecutor::new())).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    let first_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "First websocket turn",
            "store": false
        })),
    )
    .await;
    let first_completed = first_events.last().unwrap();
    let response_id = first_completed["response"]["id"]
        .as_str()
        .expect("completed response should include id")
        .to_string();

    // A continuation that fails for its own reason (unsupported parameter).
    let failed_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Fail this continuation",
            "previous_response_id": response_id,
            "conversation": "conv_test_123",
            "store": false
        })),
    )
    .await;
    let failed_error = failed_events.last().unwrap();
    assert_eq!(failed_error["type"], "error");
    assert_eq!(ws_error_code(failed_error), "unsupported_parameter");

    // Retrying the same previous_response_id must still resolve — the failed
    // child did not invalidate the cached parent.
    let retry_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Retry after failed continuation",
            "previous_response_id": response_id,
            "store": false
        })),
    )
    .await;
    let retry_completed = retry_events.last().unwrap();
    assert_eq!(retry_completed["type"], "response.completed");
}

/// Companion to the eviction fix on the `Err` arm: a child that materializes an
/// `Ok` response with status=failed (a worker finish_reason=failed analogue, not
/// an `Err`) must also NOT evict the cached parent. A `store:false` parent is
/// chainable only through the connection cache, so a retry of the same
/// previous_response_id must still resolve. Regression for the `Ok(Failed)` cache
/// arm (ws_responses.rs).
#[tokio::test]
async fn test_v1_responses_ws_keeps_cached_parent_after_ok_failed_child() {
    let url = serve_app(build_stub_app(Arc::new(OkFailedChildWsExecutor::default())).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    // turn 1: successful parent (store:false) -> cached by the session.
    let first = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model", "input": "first", "store": false
        })),
    )
    .await;
    assert_eq!(first.last().unwrap()["type"], "response.completed");

    // turn 2: a child that materializes Ok(status=failed). Must NOT evict turn 1.
    let failed = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model", "input": "fail", "store": false
        })),
    )
    .await;
    let failed_last = failed.last().unwrap();
    assert_eq!(failed_last["type"], "response.completed");
    assert_eq!(failed_last["response"]["status"], "failed");

    // turn 3: retry referencing the parent -> resolves only if the parent
    // survived the failed child (executor 404s otherwise).
    let retry = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model", "input": "retry",
            "previous_response_id": "resp_parent", "store": false
        })),
    )
    .await;
    assert_eq!(
        retry.last().unwrap()["type"],
        "response.completed",
        "a failed Ok child must not evict the cached parent"
    );
}

#[tokio::test]
async fn test_v1_responses_ws_does_not_reuse_failed_cached_response() {
    let url = serve_app(build_stub_app(Arc::new(FailedResponseWsExecutor)).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    let first_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Produce a failed websocket response",
            "store": false
        })),
    )
    .await;
    let first_completed = first_events.last().unwrap();
    assert_eq!(first_completed["type"], "response.completed");
    assert_eq!(first_completed["response"]["status"], "failed");

    let retry_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Retry after failed websocket response",
            "previous_response_id": "resp_ws_failed",
            "store": false
        })),
    )
    .await;
    let retry_error = retry_events.last().unwrap();
    assert_eq!(retry_error["type"], "error");
    assert_eq!(ws_error_code(retry_error), "previous_response_not_found");
}

#[tokio::test]
async fn test_v1_responses_ws_errors_echo_event_id() {
    let url = serve_app(build_stub_app(Arc::new(SemanticWsExecutor::new())).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    let events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "event_id": "evt_ws_123",
            "model": "mock-model",
            "input": "Conversation websocket request",
            "conversation": "conv_test_123"
        })),
    )
    .await;

    let error = events.last().unwrap();
    assert_eq!(error["type"], "error");
    assert_eq!(ws_error_code(error), "unsupported_parameter");
    assert_eq!(error["event_id"], "evt_ws_123");
    assert!(!ws_error_message(error).is_empty());
}

#[tokio::test]
async fn test_v1_responses_ws_releases_slot_after_handler_panic() {
    let url = serve_app(build_stub_app(Arc::new(PanicOnceWsExecutor::default())).await).await;
    let (mut socket, _) = connect_async(ws_endpoint(&url)).await.unwrap();

    // First turn: the handler panics. The session must catch it, surface an
    // `error` event (NOT hang), and release the in-flight slot.
    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            ws_create_request(serde_json::json!({
                "model": "mock-model",
                "input": "trigger panic",
                "store": false
            }))
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    let first = recv_json(&mut socket).await;
    assert_eq!(first["type"], "error");
    assert_eq!(ws_error_code(&first), "internal_error");

    // Second turn on the SAME connection must be accepted and complete —
    // proving the panic did not wedge the slot (no `concurrent_response_create`)
    // and the connection survived.
    let second_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "after panic",
            "store": false
        })),
    )
    .await;
    let completed = second_events.last().unwrap();
    assert_eq!(completed["type"], "response.completed");
    assert_eq!(
        completed["response"]["output"][0]["content"][0]["text"],
        "recovered websocket output"
    );
}
