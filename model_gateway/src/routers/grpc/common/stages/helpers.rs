//! Common helper functions shared across stages

use std::sync::Arc;

use openai_protocol::chat::ChatMessage;
use rand::Rng;
use sha2::{Digest, Sha256};
use smg_grpc_client::sglang_proto::DisaggregatedParams;
use tracing::debug;

use crate::{
    routers::grpc::{context::WorkerSelection, proto_wrapper::ProtoGenerateRequest},
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

fn chat_message_role(msg: &ChatMessage) -> &'static str {
    match msg {
        ChatMessage::System { .. } => "system",
        ChatMessage::User { .. } => "user",
        ChatMessage::Assistant { .. } => "assistant",
        ChatMessage::Tool { .. } => "tool",
        ChatMessage::Function { .. } => "function",
        ChatMessage::Developer { .. } => "developer",
    }
}

fn chat_message_text_content(msg: &ChatMessage) -> String {
    match msg {
        ChatMessage::System { content, .. }
        | ChatMessage::User { content, .. }
        | ChatMessage::Tool { content, .. }
        | ChatMessage::Developer { content, .. } => content.to_simple_string(),
        ChatMessage::Assistant { content, .. } => content
            .as_ref()
            .map_or_else(String::new, |c| c.to_simple_string()),
        ChatMessage::Function { content, .. } => content.clone(),
    }
}

/// Compute per-message SHA-256 hashes matching TRT-LLM's `openai_server.py` format:
/// `sha256(role + "\x00" + content).hexdigest()[:12]`
pub(crate) fn compute_and_log_message_hashes(
    request_id: &str,
    messages: &[ChatMessage],
) -> Vec<(String, String)> {
    let hashes: Vec<(String, String)> = messages
        .iter()
        .map(|msg| {
            let role = chat_message_role(msg);
            let content = chat_message_text_content(msg);
            let mut hasher = Sha256::new();
            hasher.update(format!("{role}\x00{content}").as_bytes());
            let hash = format!("{:x}", hasher.finalize());
            (role.to_string(), hash[..12].to_string())
        })
        .collect();
    debug!(
        target: "smg::request",
        request_id = %request_id,
        message_hashes = ?hashes,
        "Request message hashes for session reconstruction"
    );
    hashes
}

/// Compute per-message SHA-256 hashes from InputMessage (Messages API) format.
pub(crate) fn compute_and_log_input_message_hashes(
    request_id: &str,
    messages: &[openai_protocol::messages::InputMessage],
) -> Vec<(String, String)> {
    use openai_protocol::messages::Role;
    let hashes: Vec<(String, String)> = messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            let content = match &msg.content {
                openai_protocol::messages::InputContent::String(s) => s.clone(),
                openai_protocol::messages::InputContent::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|b| {
                        if let openai_protocol::messages::InputContentBlock::Text(t) = b {
                            Some(t.text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
            };
            let mut hasher = Sha256::new();
            hasher.update(format!("{role}\x00{content}").as_bytes());
            let hash = format!("{:x}", hasher.finalize());
            (role.to_string(), hash[..12].to_string())
        })
        .collect();
    debug!(
        target: "smg::request",
        request_id = %request_id,
        message_hashes = ?hashes,
        "Request message hashes for session reconstruction"
    );
    hashes
}
