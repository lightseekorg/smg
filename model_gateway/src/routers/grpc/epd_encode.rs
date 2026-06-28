//! Backend-specific EPD encode adapters.
//!
//! Request building owns encode rendezvous planning; request execution owns
//! encode-worker dispatch. This module owns backend wire details: item assembly,
//! transport-specific tensor payloads, and the encode-worker RPC.

use anyhow::{anyhow, Result};
use rand::RngExt;
use smg_grpc_client::tokenspeed_encoder::{
    tokenspeed_encoder_proto as tokenspeed_encoder, TokenSpeedEncoderClient,
};
use uuid::Uuid;

use super::{
    client::GrpcClient,
    context::{ClientSelection, WorkerSelection},
    multimodal::{assemble_tokenspeed, MultimodalIntermediate, PrecomputedMultimodalIntermediate},
    proto_wrapper::{
        cleanup_tokenspeed_items_encoder_shm, cleanup_tokenspeed_shm_handles,
        collect_tokenspeed_multimodal_inputs_shm_handles, EncodeItemBootstrapInfo,
        TokenSpeedMultimodalData, TokenSpeedMultimodalItem,
    },
};
use crate::worker::DEFAULT_BOOTSTRAP_PORT;

pub(crate) struct EncodePlan {
    bootstrap_info: Vec<EncodeItemBootstrapInfo>,
    dispatch: EncodeDispatchPlan,
}

pub(crate) struct EncodeDispatchPlan {
    jobs: Vec<PreparedEncodeJob>,
}

pub(crate) struct PreparedEncodeJob {
    item: PreparedEncodeItem,
    endpoint: String,
    bootstrap_room: i64,
}

impl EncodePlan {
    pub(crate) fn is_empty(&self) -> bool {
        self.dispatch.is_empty()
    }

    pub(crate) fn into_parts(self) -> (Vec<EncodeItemBootstrapInfo>, EncodeDispatchPlan) {
        (self.bootstrap_info, self.dispatch)
    }
}

impl EncodeDispatchPlan {
    fn new(jobs: Vec<PreparedEncodeJob>) -> Self {
        Self { jobs }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.jobs.len()
    }

    pub(crate) fn into_jobs(self) -> Vec<PreparedEncodeJob> {
        self.jobs
    }
}

impl PreparedEncodeJob {
    pub(crate) async fn dispatch(self) -> std::result::Result<(), String> {
        self.item.dispatch(self.endpoint, self.bootstrap_room).await
    }
}

pub(crate) enum PreparedEncodeItem {
    TokenSpeed {
        item: Option<TokenSpeedMultimodalItem>,
        shm_enabled: bool,
        cleanup_on_drop: bool,
    },
}

impl PreparedEncodeItem {
    fn tokenspeed(item: TokenSpeedMultimodalItem, shm_enabled: bool) -> Self {
        Self::TokenSpeed {
            item: Some(item),
            shm_enabled,
            cleanup_on_drop: true,
        }
    }

    pub(crate) async fn dispatch(
        mut self,
        endpoint: String,
        bootstrap_room: i64,
    ) -> std::result::Result<(), String> {
        match &mut self {
            Self::TokenSpeed {
                item,
                shm_enabled,
                cleanup_on_drop,
            } => {
                let item = item
                    .take()
                    .ok_or_else(|| "encode item was already dispatched".to_string())?;
                *cleanup_on_drop = false;
                let request = tokenspeed_encoder::EncodeRequest {
                    request_id: format!("encode-{}", Uuid::now_v7()),
                    mm_inputs: Some(
                        TokenSpeedMultimodalData {
                            items: vec![item],
                            shm_enabled: *shm_enabled,
                        }
                        .into_proto(),
                    ),
                    items: vec![tokenspeed_encoder::EncodeItemAssignment { bootstrap_room }],
                };
                let shm_handles = request
                    .mm_inputs
                    .as_ref()
                    .map(collect_tokenspeed_multimodal_inputs_shm_handles)
                    .unwrap_or_default();
                let result = send_tokenspeed_encode_rpc(endpoint, request).await;
                cleanup_tokenspeed_shm_handles(&shm_handles);
                result
            }
        }
    }
}

impl Drop for PreparedEncodeItem {
    fn drop(&mut self) {
        match self {
            Self::TokenSpeed {
                item: Some(item),
                cleanup_on_drop: true,
                ..
            } => {
                cleanup_tokenspeed_items_encoder_shm(std::slice::from_ref(item), None);
            }
            _ => {}
        }
    }
}

pub(crate) fn build_plan_from_intermediate(
    intermediate: &MultimodalIntermediate,
    clients: Option<&ClientSelection>,
    workers: Option<&WorkerSelection>,
) -> Result<EncodePlan> {
    match intermediate {
        MultimodalIntermediate::Precomputed(precomputed) => {
            build_plan(precomputed, clients, workers)
        }
    }
}

fn build_plan(
    precomputed: &PrecomputedMultimodalIntermediate,
    clients: Option<&ClientSelection>,
    workers: Option<&WorkerSelection>,
) -> Result<EncodePlan> {
    let workers = workers.ok_or_else(|| anyhow!("Worker selection stage not completed"))?;
    let encode_assignments = workers
        .encode_assignments()
        .filter(|assignments| !assignments.is_empty())
        .ok_or_else(|| anyhow!("Encode planning requires EPD worker selection"))?
        .to_vec();

    let items = prepare_items(precomputed, clients, Some(workers))?;
    if items.is_empty() {
        return Ok(EncodePlan {
            bootstrap_info: Vec::new(),
            dispatch: EncodeDispatchPlan::new(Vec::new()),
        });
    }

    if encode_assignments.len() != items.len() {
        return Err(anyhow!(
            "EPD encode item/assignment count mismatch: {} items, {} assignments",
            items.len(),
            encode_assignments.len()
        ));
    }

    let mut bootstrap_info = Vec::with_capacity(items.len());
    let mut jobs = Vec::with_capacity(items.len());
    for (global_index, (item, assignment)) in items
        .into_iter()
        .zip(encode_assignments.into_iter())
        .enumerate()
    {
        if assignment.item_index != global_index {
            return Err(anyhow!(
                "EPD encode assignment order mismatch: expected item {}, got {}",
                global_index,
                assignment.item_index
            ));
        }

        // 63-bit room: no in-flight dedup, so a 2^31 space birthday-collides
        // under load (silent embedding cross-wire). See the proto field doc.
        let bootstrap_room = rand::rng().random_range(0..i64::MAX);

        bootstrap_info.push(EncodeItemBootstrapInfo {
            item_index: global_index as u32,
            bootstrap_host: assignment.worker.bootstrap_host().to_string(),
            bootstrap_port: assignment
                .worker
                .bootstrap_port()
                .unwrap_or(DEFAULT_BOOTSTRAP_PORT) as i32,
            bootstrap_room,
        });
        jobs.push(PreparedEncodeJob {
            item,
            endpoint: assignment.worker.url().to_string(),
            bootstrap_room,
        });
    }

    Ok(EncodePlan {
        bootstrap_info,
        dispatch: EncodeDispatchPlan::new(jobs),
    })
}

pub(crate) fn prepare_items(
    precomputed: &PrecomputedMultimodalIntermediate,
    clients: Option<&ClientSelection>,
    workers: Option<&WorkerSelection>,
) -> Result<Vec<PreparedEncodeItem>> {
    let clients = clients.ok_or_else(|| anyhow!("Client acquisition stage not completed"))?;
    match clients {
        ClientSelection::Disaggregated {
            prefill: GrpcClient::TokenSpeed(_),
            ..
        } => prepare_tokenspeed_items(precomputed, workers),
        ClientSelection::Disaggregated { prefill, .. } => Err(anyhow!(
            "EPD encode is not implemented for {} backend",
            backend_name(prefill)
        )),
        _ => Err(anyhow!("Encode planning requires EPD client selection")),
    }
}

fn prepare_tokenspeed_items(
    precomputed: &PrecomputedMultimodalIntermediate,
    workers: Option<&WorkerSelection>,
) -> Result<Vec<PreparedEncodeItem>> {
    let tokenspeed_mm = assemble_tokenspeed(precomputed, workers, false)?;
    let shm_enabled = tokenspeed_mm.shm_enabled;
    Ok(tokenspeed_mm
        .items
        .into_iter()
        .map(|item| PreparedEncodeItem::tokenspeed(item, shm_enabled))
        .collect())
}

fn backend_name(client: &GrpcClient) -> &'static str {
    match client {
        GrpcClient::Sglang(_) => "SGLang",
        GrpcClient::Vllm(_) => "vLLM",
        GrpcClient::Trtllm(_) => "TRT-LLM",
        GrpcClient::Mlx(_) => "MLX",
        GrpcClient::TokenSpeed(_) => "TokenSpeed",
    }
}

async fn send_tokenspeed_encode_rpc(
    endpoint: String,
    request: tokenspeed_encoder::EncodeRequest,
) -> std::result::Result<(), String> {
    let client = TokenSpeedEncoderClient::connect_cached(&endpoint)
        .await
        .map_err(|e| format!("connect to encode worker {endpoint} failed: {e}"))?;
    let response = client
        .encode(request)
        .await
        .map_err(|e| format!("encode RPC to {endpoint} failed: {}", e.message()))?;
    if !response.accepted {
        return Err(format!(
            "encode worker {endpoint} did not accept the request"
        ));
    }
    Ok(())
}
