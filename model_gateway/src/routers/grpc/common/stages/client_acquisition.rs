//! Client acquisition stage: Get gRPC clients from selected workers

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::PipelineStage;
use crate::{
    routers::{
        error,
        grpc::{
            client::GrpcClient,
            context::{ClientSelection, RequestContext, WorkerSelection},
        },
    },
    worker::Worker,
};

/// Client acquisition stage: Get gRPC clients from selected workers
pub(crate) struct ClientAcquisitionStage;

#[async_trait]
impl PipelineStage for ClientAcquisitionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let workers = ctx.state.workers.as_ref().ok_or_else(|| {
            error!(
                function = "ClientAcquisitionStage::execute",
                "Worker selection stage not completed"
            );
            error::internal_error(
                "worker_selection_not_completed",
                "Worker selection not completed",
            )
        })?;

        let clients = match workers {
            WorkerSelection::Single { worker } => {
                let client = get_grpc_client_from_worker(worker).await?;
                ClientSelection::Single { client }
            }
            WorkerSelection::Dual {
                prefill, decode, ..
            } => {
                let prefill_client = get_grpc_client_from_worker(prefill).await?;
                let decode_client = get_grpc_client_from_worker(decode).await?;

                ClientSelection::Dual {
                    prefill: prefill_client,
                    decode: decode_client,
                }
            }
        };

        ctx.state.clients = Some(clients);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ClientAcquisition"
    }
}

async fn get_grpc_client_from_worker(worker: &Arc<dyn Worker>) -> Result<GrpcClient, Response> {
    // Get cached client from worker (or create one if not cached yet)
    let client_arc = worker
        .get_grpc_client()
        .await
        .map_err(|e| {
            error!(
                function = "get_grpc_client_from_worker",
                error = %e,
                "Failed to get gRPC client from worker"
            );
            error::internal_error(
                "get_grpc_client_failed",
                format!("Failed to get gRPC client: {e}"),
            )
        })?
        .ok_or_else(|| {
            error!(
                function = "get_grpc_client_from_worker",
                "Selected worker not configured for gRPC"
            );
            error::internal_error(
                "worker_not_configured_for_grpc",
                "Selected worker is not configured for gRPC",
            )
        })?;

    Ok((*client_arc).clone())
}
