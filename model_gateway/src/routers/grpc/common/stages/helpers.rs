//! Common helper functions shared across stages

use std::sync::Arc;

use axum::response::Response;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::common::StringOrArray;
use rand::Rng;
use smg_grpc_client::sglang_proto::DisaggregatedParams;
use tracing::debug;

use crate::{
    routers::{
        error,
        grpc::{
            context::WorkerSelection, proto_wrapper::ProtoGenerateRequest,
            utils::resolve_mlx_stop_ids,
        },
    },
    worker::{RuntimeType, Worker, DEFAULT_BOOTSTRAP_PORT},
};

/// Inject PD bootstrap metadata for SGLang if needed.
///
/// SGLang uses DisaggregatedParams with bootstrap host/port/room.
/// vLLM uses different mechanisms: NIXL (automatic prefix matching) or
/// Mooncake (kv_transfer_params injected in request_execution stage).
pub(crate) fn maybe_inject_pd_metadata(
    request: &mut ProtoGenerateRequest,
    workers: &WorkerSelection,
) {
    if let WorkerSelection::Dual {
        prefill,
        runtime_type,
        ..
    } = workers
    {
        if *runtime_type == RuntimeType::Sglang {
            inject_sglang_bootstrap_metadata(request, prefill);
        }
    }
}

/// Inject bootstrap metadata into a SGLang gRPC request.
fn inject_sglang_bootstrap_metadata(
    request: &mut ProtoGenerateRequest,
    prefill_worker: &Arc<dyn Worker>,
) {
    let metadata = prefill_worker.metadata();
    let hostname = metadata.bootstrap_host();
    let bootstrap_port = metadata.bootstrap_port().unwrap_or(DEFAULT_BOOTSTRAP_PORT);
    let room_id = rand::rng().random_range(0..i32::MAX);

    let disagg_params = DisaggregatedParams {
        bootstrap_host: hostname.to_string(),
        bootstrap_port: bootstrap_port as i32,
        bootstrap_room: room_id,
    };

    let sglang_request = request.as_sglang_mut();
    sglang_request.disaggregated_params = Some(disagg_params);

    debug!(
        "Injected bootstrap metadata: host={}, port={}, room={}",
        hostname, bootstrap_port, room_id
    );
}

/// Convert string stop sequences to token IDs and append them to the MLX proto request.
///
/// The MLX proto only supports stop_token_ids; string stop sequences from the
/// CompletionRequest must be tokenized here before the request is dispatched.
/// No-op if the request has no string stop sequences.
#[expect(
    clippy::result_large_err,
    reason = "Response is the standard error type in the pipeline stage pattern"
)]
pub(crate) fn apply_mlx_stop_sequences(
    proto_request: &mut ProtoGenerateRequest,
    stop: Option<&StringOrArray>,
    tokenizer: Option<&dyn Tokenizer>,
) -> Result<(), Response> {
    let Some(stop) = stop else {
        return Ok(());
    };

    let token_ids = resolve_mlx_stop_ids(stop, tokenizer)?;

    if let ProtoGenerateRequest::Mlx(req) = proto_request {
        let sampling = req.sampling_params.as_mut().ok_or_else(|| {
            error::internal_error(
                "mlx_sampling_params_missing",
                "MLX GenerateRequest has no sampling_params; cannot inject stop IDs",
            )
        })?;
        sampling.stop_token_ids.extend(token_ids);
    }

    Ok(())
}
