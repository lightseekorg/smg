//! Common helper functions shared across stages

use std::sync::Arc;

use rand::Rng;
use tracing::debug;

use crate::{
    core::Worker, grpc_client::sglang_proto::DisaggregatedParams,
    routers::grpc::proto_wrapper::ProtoGenerateRequest,
};

/// Inject PD bootstrap metadata into a SGLang gRPC request.
///
/// # Panics
/// Panics if called with a non-SGLang request. Callers must check runtime type
/// before calling this function. vLLM PD uses NIXL for transparent KV transfer
/// and does not need metadata injection.
pub(crate) fn inject_bootstrap_metadata(
    request: &mut ProtoGenerateRequest,
    prefill_worker: &Arc<dyn Worker>,
) {
    let hostname = prefill_worker.bootstrap_host();
    let bootstrap_port = prefill_worker.bootstrap_port().unwrap_or(8998);

    // Generate room ID for bootstrap
    let room_id = rand::rng().random_range(0..i32::MAX);

    // Create DisaggregatedParams
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
