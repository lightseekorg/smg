//! Mock HTTP worker endpoints — the vLLM/SGLang-compatible surface the SMG
//! gateway probes and routes to. Every response is canned; no real model.

use std::{convert::Infallible, sync::Arc};

use axum::{
    body::Bytes,
    extract::State,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures::{stream, Stream};
use serde_json::{json, Value};
use tokio::net::TcpListener;

use crate::config::Config;

/// Build the router serving the mock HTTP worker contract.
pub fn router(cfg: Arc<Config>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(chat))
        .route("/generate", post(chat))
        .route("/v1/loads", get(loads))
        .with_state(cfg)
}

/// Serve the mock HTTP worker contract on `port` until the process exits.
pub async fn serve(cfg: Arc<Config>, host: String, port: u16) {
    let listener = match TcpListener::bind((host.as_str(), port)).await {
        Ok(listener) => listener,
        Err(e) => {
            tracing::error!("http worker bind {host}:{port} failed: {e}");
            return;
        }
    };
    if let Err(e) = axum::serve(listener, router(cfg)).await {
        tracing::error!("http worker {port} stopped: {e}");
    }
}

async fn health() -> &'static str {
    "OK"
}

async fn models(State(cfg): State<Arc<Config>>) -> Response {
    Json(json!({
        "object": "list",
        "data": [{
            "id": cfg.model_id,
            "object": "model",
            "created": 0,
            "owned_by": "sglang",
            "root": cfg.model_id,
            "max_model_len": 32768,
        }],
    }))
    .into_response()
}

async fn loads() -> Response {
    Json(json!({
        "timestamp": "",
        "dp_rank_count": 1,
        "loads": [{
            "dp_rank": 0,
            "num_running_reqs": 0,
            "num_waiting_reqs": 0,
            "num_waiting_uncached_tokens": 0,
            "num_total_reqs": 0,
            "num_used_tokens": 0,
            "max_total_num_tokens": 1_000_000,
            "token_usage": 0.0,
            "gen_throughput": 0.0,
            "cache_hit_rate": 0.0,
            "utilization": 0.0,
            "max_running_requests": 0,
        }],
    }))
    .into_response()
}

async fn chat(State(cfg): State<Arc<Config>>, body: Bytes) -> Response {
    let stream_requested = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|v| v.get("stream").and_then(Value::as_bool))
        .unwrap_or(false);

    if !cfg.gen_delay.is_zero() {
        tokio::time::sleep(cfg.gen_delay).await;
    }

    if stream_requested {
        stream_chat(&cfg).into_response()
    } else {
        Json(completion(&cfg)).into_response()
    }
}

fn completion(cfg: &Config) -> Value {
    json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 0,
        "model": cfg.model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "mock"},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": cfg.output_tokens,
            "total_tokens": u64::from(cfg.output_tokens) + 1,
        },
    })
}

fn stream_chat(cfg: &Config) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mut events: Vec<Result<Event, Infallible>> = Vec::new();
    for _ in 0..cfg.output_tokens {
        let frame = json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": null}],
        });
        events.push(Ok(Event::default().data(frame.to_string())));
    }
    let final_frame = json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    });
    events.push(Ok(Event::default().data(final_frame.to_string())));
    events.push(Ok(Event::default().data("[DONE]")));
    Sse::new(stream::iter(events))
}
