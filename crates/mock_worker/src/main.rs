//! mock-worker: spin up many mock HTTP and/or gRPC inference workers in one
//! process to scale-test the SMG gateway's routing and runtime behavior.
//!
//! Workers are protocol-accurate stand-ins for vLLM/SGLang engines: HTTP
//! workers answer the probe + chat/generate surface; gRPC workers implement the
//! TokenSpeed scheduler service (the gateway tokenizes and speaks token ids).
//! All responses are canned — there is no real model.

mod config;
mod grpc;
mod http;

use std::{future::Future, pin::Pin, process::ExitCode, sync::Arc};

use futures::future::join_all;

use crate::config::Config;

type BoxFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let cfg = match Config::from_args() {
        Ok(cfg) => Arc::new(cfg),
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };

    tracing::info!(
        "mock-worker: {} http from :{}, {} grpc from :{}, model={}",
        cfg.http_count,
        cfg.http_base_port,
        cfg.grpc_count,
        cfg.grpc_base_port,
        cfg.model_id,
    );

    let mut servers: Vec<BoxFuture> = Vec::new();
    for i in 0..cfg.http_count {
        let Some(port) = cfg.http_base_port.checked_add(i) else {
            tracing::warn!("http port range overflowed u16 at offset {i}; stopping early");
            break;
        };
        servers.push(Box::pin(http::serve(cfg.clone(), cfg.host.clone(), port)));
    }
    for i in 0..cfg.grpc_count {
        let Some(port) = cfg.grpc_base_port.checked_add(i) else {
            tracing::warn!("grpc port range overflowed u16 at offset {i}; stopping early");
            break;
        };
        servers.push(Box::pin(grpc::serve(cfg.clone(), cfg.host.clone(), port)));
    }

    tracing::info!("started {} mock workers; ctrl-c to stop", servers.len());

    tokio::select! {
        _ = join_all(servers) => {}
        result = tokio::signal::ctrl_c() => {
            if let Err(e) = result {
                tracing::error!("failed to listen for ctrl-c: {e}");
            }
            tracing::info!("shutting down");
        }
    }
    ExitCode::SUCCESS
}
