use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Body,
    http::{HeaderMap, Request, Response, StatusCode},
    middleware,
    response::IntoResponse,
};
use criterion::{criterion_group, criterion_main, Criterion};
use http_body_util::BodyExt;
use openai_protocol::chat::ChatCompletionRequest;
use smg::{
    app_context::AppContext,
    config::RouterConfig,
    middleware::wasm_middleware,
    routers::RouterTrait,
    server::AppState,
    wasm::module::{
        BodyPolicy, MiddlewareAttachPoint, WasmModule, WasmModuleAttachPoint, WasmModuleMeta,
    },
};
use tokio::runtime::Runtime;
use tower::{Layer, Service};
use uuid::Uuid;
use wasm_encoder::{Component, Module, TypeSection};

#[derive(Debug)]
struct MockRouter;

#[async_trait::async_trait]
impl RouterTrait for MockRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response<Body> {
        StatusCode::OK.into_response()
    }
    fn router_type(&self) -> &'static str {
        "mock"
    }
}

/// Simulates a streaming response with a 500ms delay before the final chunk.
async fn mock_next_streaming(_req: Request<Body>) -> Response<Body> {
    let (tx, rx) = tokio::sync::mpsc::channel(16);
    tokio::spawn(async move {
        let _ = tx
            .send(Ok::<_, std::io::Error>(bytes::Bytes::from("chunk 1 ")))
            .await;
        tokio::time::sleep(Duration::from_millis(500)).await;
        let _ = tx
            .send(Ok::<_, std::io::Error>(bytes::Bytes::from("chunk 2")))
            .await;
    });
    Response::new(Body::from_stream(
        tokio_stream::wrappers::ReceiverStream::new(rx),
    ))
}

/// Generates a minimal valid WASM component bytes.
fn create_dummy_wasm_bytes() -> Vec<u8> {
    let mut module = Module::new();
    module.section(&TypeSection::new());
    let mut component = Component::new();
    component.section(&wasm_encoder::ModuleSection(&module));
    component.finish()
}

fn bench_wasm_bottleneck(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = RouterConfig::builder().enable_wasm(true).build_unchecked();
    let context = rt.block_on(AppContext::from_config(config, 30)).unwrap();

    // REGISTER A DUMMY MODULE
    // This ensures the middleware doesn't skip buffering
    if let Some(wasm_manager) = context.wasm_manager.as_ref() {
        let wasm_bytes = create_dummy_wasm_bytes();
        let module_uuid = Uuid::new_v4();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let module = WasmModule {
            module_uuid,
            module_meta: WasmModuleMeta {
                name: "bottleneck_test".to_string(),
                file_path: "memory".to_string(),
                sha256_hash: [0u8; 32],
                size_bytes: wasm_bytes.len() as u64,
                created_at: now,
                last_accessed_at: now,
                access_count: 0,
                attach_points: vec![
                    WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnRequest),
                    WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnResponse),
                ],
                body_policy: BodyPolicy::HeadersOnly,
                wasm_bytes,
            },
        };
        wasm_manager
            .register_module_internal(module)
            .expect("Failed to register module");
    }

    let app_state = Arc::new(AppState {
        router: Arc::new(MockRouter),
        context: Arc::new(context),
        concurrency_queue_tx: None,
        router_manager: None,
        mesh_handler: None,
    });

    c.bench_function("wasm_middleware_buffering_bottleneck", |b| {
        b.iter(|| {
            rt.block_on(async {
                let req = Request::builder()
                    .uri("/v1/chat/completions")
                    .body(Body::empty())
                    .unwrap();
                let mut service =
                    middleware::from_fn_with_state(app_state.clone(), wasm_middleware).layer(
                        tower::service_fn(|req| async move {
                            Ok::<_, std::convert::Infallible>(mock_next_streaming(req).await)
                        }),
                    );

                let response = service.call(req).await.unwrap();
                let mut body = response.into_body();

                // Measure time to receive the FIRST frame.
                // With buffering, this will be >500ms.
                let _ = body.frame().await;
            });
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10); // Low sample size because each iteration is 500ms+
    targets = bench_wasm_bottleneck
}
criterion_main!(benches);
