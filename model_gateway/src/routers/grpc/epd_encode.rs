//! Backend-specific EPD encode adapters.
//!
//! Request building owns encode rendezvous planning; request execution owns
//! encode-worker dispatch. This module owns backend wire details: item assembly,
//! transport-specific tensor payloads, and the encode-worker RPC.

use anyhow::{anyhow, Result};
use rand::RngExt;
use smg_grpc_client::{
    common_proto,
    tokenspeed_encoder::{tokenspeed_encoder_proto as tokenspeed_encoder, TokenSpeedEncoderClient},
};
use uuid::Uuid;

use super::{
    client::GrpcClient,
    context::{ClientSelection, WorkerSelection},
    multimodal::{assemble_tokenspeed_for_encode, MultimodalIntermediate},
    proto_wrapper::{
        cleanup_mm_shm_handles, cleanup_tokenspeed_items_encoder_shm,
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
        shm_min_bytes: usize,
        cleanup_on_drop: bool,
    },
}

impl PreparedEncodeItem {
    fn tokenspeed(item: TokenSpeedMultimodalItem, shm_enabled: bool, shm_min_bytes: usize) -> Self {
        Self::TokenSpeed {
            item: Some(item),
            shm_enabled,
            shm_min_bytes,
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
                shm_min_bytes,
                cleanup_on_drop,
            } => {
                let mut item = item
                    .take()
                    .ok_or_else(|| "encode item was already dispatched".to_string())?;
                *cleanup_on_drop = false;
                item.encoder_input = item.encoder_input.try_export_nixl_remote(bootstrap_room);
                let request = tokenspeed_encoder::EncodeRequest {
                    request_id: format!("encode-{}", Uuid::now_v7()),
                    mm_inputs: Some(
                        TokenSpeedMultimodalData {
                            items: vec![item],
                            shm_enabled: *shm_enabled,
                            shm_min_bytes: *shm_min_bytes,
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
                let _shm_guard = TokenSpeedShmCleanupGuard(shm_handles);
                send_tokenspeed_encode_rpc(endpoint, request).await
            }
        }
    }
}

struct TokenSpeedShmCleanupGuard(Vec<common_proto::ShmHandle>);

impl Drop for TokenSpeedShmCleanupGuard {
    fn drop(&mut self) {
        cleanup_mm_shm_handles(&self.0);
    }
}

impl Drop for PreparedEncodeItem {
    fn drop(&mut self) {
        if let Self::TokenSpeed {
            item: Some(item),
            cleanup_on_drop: true,
            ..
        } = self
        {
            cleanup_tokenspeed_items_encoder_shm(std::slice::from_ref(item), None);
        }
    }
}

pub(crate) fn build_plan_from_intermediate(
    intermediate: &MultimodalIntermediate,
    clients: Option<&ClientSelection>,
    workers: Option<&WorkerSelection>,
) -> Result<EncodePlan> {
    build_plan(intermediate, clients, workers)
}

fn build_plan(
    intermediate: &MultimodalIntermediate,
    clients: Option<&ClientSelection>,
    workers: Option<&WorkerSelection>,
) -> Result<EncodePlan> {
    let workers = workers.ok_or_else(|| anyhow!("Worker selection stage not completed"))?;
    let items = prepare_items(intermediate, clients, Some(workers))?;
    if items.is_empty() {
        return Ok(EncodePlan {
            bootstrap_info: Vec::new(),
            dispatch: EncodeDispatchPlan::new(Vec::new()),
        });
    }

    plan_encode_jobs(items, workers)
}

/// Match prepared encode items to their per-item encode-worker assignments,
/// producing the encode->prefill bootstrap info and the dispatch jobs.
///
/// Validates that the EPD worker selection carries exactly one encode assignment
/// per item, in item order, then assigns each item a random bootstrap room.
/// Callers must handle the empty-`items` case before calling this (encode
/// planning requires at least one item and a matching non-empty assignment set).
fn plan_encode_jobs(
    items: Vec<PreparedEncodeItem>,
    workers: &WorkerSelection,
) -> Result<EncodePlan> {
    let encode_assignments = workers
        .encode_assignments()
        .filter(|assignments| !assignments.is_empty())
        .ok_or_else(|| anyhow!("Encode planning requires EPD worker selection"))?
        .to_vec();

    if encode_assignments.len() != items.len() {
        return Err(anyhow!(
            "EPD encode item/assignment count mismatch: {} items, {} assignments",
            items.len(),
            encode_assignments.len()
        ));
    }

    let mut bootstrap_info = Vec::with_capacity(items.len());
    let mut jobs = Vec::with_capacity(items.len());
    for (global_index, (item, assignment)) in items.into_iter().zip(encode_assignments).enumerate()
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
    intermediate: &MultimodalIntermediate,
    clients: Option<&ClientSelection>,
    workers: Option<&WorkerSelection>,
) -> Result<Vec<PreparedEncodeItem>> {
    let clients = clients.ok_or_else(|| anyhow!("Client acquisition stage not completed"))?;
    match clients {
        ClientSelection::Disaggregated {
            prefill: GrpcClient::TokenSpeed(_),
            ..
        } => prepare_tokenspeed_items(intermediate, workers),
        ClientSelection::Disaggregated { prefill, .. } => Err(anyhow!(
            "EPD encode is not implemented for {} backend",
            backend_name(prefill)
        )),
        ClientSelection::Single { .. } => {
            Err(anyhow!("Encode planning requires EPD client selection"))
        }
    }
}

fn prepare_tokenspeed_items(
    intermediate: &MultimodalIntermediate,
    workers: Option<&WorkerSelection>,
) -> Result<Vec<PreparedEncodeItem>> {
    let tokenspeed_mm = assemble_tokenspeed_for_encode(intermediate, workers)?;
    let shm_enabled = tokenspeed_mm.shm_enabled;
    let shm_min_bytes = tokenspeed_mm.shm_min_bytes;
    Ok(tokenspeed_mm
        .items
        .into_iter()
        .map(|item| PreparedEncodeItem::tokenspeed(item, shm_enabled, shm_min_bytes))
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

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use llm_multimodal::{
        FieldLayout, ImageDetail, ImageFrame, ImageSource, Modality, PlaceholderRange,
        PreprocessedEncoderInputs,
    };
    use ndarray::{ArrayD, IxDyn};

    use super::*;
    use crate::{
        routers::grpc::{
            context::{EncodeWorkerAssignment, WorkerSelection},
            proto_wrapper::{TokenSpeedModality, TokenSpeedMultimodalItem, TokenSpeedTensor},
        },
        worker::{BasicWorkerBuilder, RuntimeType, Worker, WorkerType},
    };

    /// Build a precomputed intermediate carrying `n` image items, mirroring the
    /// batched-layout fixture used in the assemble.rs tests.
    fn image_intermediate(n: usize) -> PrecomputedMultimodalIntermediate {
        // encoder_input: one row per item, 2 features each.
        let data: Vec<f32> = (0..n * 2).map(|v| v as f32).collect();
        let preprocessed = PreprocessedEncoderInputs {
            encoder_input: ArrayD::from_shape_vec(IxDyn(&[n, 2]), data).unwrap(),
            feature_token_counts: vec![1; n],
            item_sizes: vec![(1, 1); n],
            model_specific: HashMap::new(),
        };
        let images = (0..n)
            .map(|i| {
                Arc::new(ImageFrame::new(
                    image::DynamicImage::new_rgb8(1, 1),
                    bytes::Bytes::from_static(b"x"),
                    ImageDetail::Auto,
                    ImageSource::InlineBytes,
                    format!("hash-{i}"),
                ))
            })
            .collect();
        let placeholders = (0..n)
            .map(|i| PlaceholderRange {
                offset: 10 * (i + 1),
                length: 1,
            })
            .collect();
        PrecomputedMultimodalIntermediate {
            modality: Modality::Image,
            preprocessed,
            images,
            videos: vec![],
            placeholders,
            patch_offsets: None,
            placeholder_token_id: Some(151655),
            field_layouts: HashMap::from([("pixel_values".to_string(), FieldLayout::Batched)]),
            keep_on_cpu_keys: vec![],
        }
    }

    /// A synthetic prepared item with an inline (non-SHM) encoder input, so its
    /// `Drop` never touches /dev/shm.
    fn synthetic_item() -> PreparedEncodeItem {
        let item = TokenSpeedMultimodalItem {
            modality: TokenSpeedModality::Image,
            encoder_input: TokenSpeedTensor::inline(
                vec![0u8; 4],
                vec![2, 1],
                "bfloat16".to_string(),
            ),
            model_specific_tensors: HashMap::new(),
            placeholder_token_id: Some(151655),
            mm_placeholders: vec![(0, 1)],
            content_hash: vec![],
        };
        PreparedEncodeItem::tokenspeed(item, false, 0)
    }

    /// An encode worker whose URL yields `bootstrap_host` and whose spec sets a
    /// non-default `bootstrap_port`, so both are assertable in the plan output.
    fn encode_worker(host: &str, port: u16) -> Arc<dyn Worker> {
        let worker = BasicWorkerBuilder::new(format!("http://{host}:8080"))
            .worker_type(WorkerType::Encode)
            .bootstrap_port(Some(port))
            .build();
        Arc::new(worker)
    }

    fn disaggregated(assignments: Vec<EncodeWorkerAssignment>) -> WorkerSelection {
        WorkerSelection::Disaggregated {
            encode_assignments: Some(assignments),
            prefill: Arc::new(BasicWorkerBuilder::new("http://prefill:8080").build()),
            decode: Arc::new(BasicWorkerBuilder::new("http://decode:8080").build()),
            runtime_type: RuntimeType::TokenSpeed,
        }
    }

    #[test]
    fn prepare_tokenspeed_items_yields_one_item_per_image() {
        // assemble_tokenspeed splits the batched encoder input into per-item
        // tensors; prepare_tokenspeed_items wraps each as a PreparedEncodeItem.
        let precomputed = image_intermediate(3);
        let items = prepare_tokenspeed_items(&precomputed, None).unwrap();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn plan_encode_jobs_count_mismatch_errors() {
        // 1 item but 2 assignments: the count guard must fire.
        let items = vec![synthetic_item()];
        let workers = disaggregated(vec![
            EncodeWorkerAssignment {
                item_index: 0,
                worker: encode_worker("enc-a", 9001),
            },
            EncodeWorkerAssignment {
                item_index: 1,
                worker: encode_worker("enc-b", 9002),
            },
        ]);

        // Avoid unwrap_err (EncodePlan is not Debug); match the Err directly.
        let err = match plan_encode_jobs(items, &workers) {
            Ok(_) => panic!("expected count-mismatch error"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("count mismatch"), "got: {err}");
    }

    #[test]
    fn plan_encode_jobs_order_mismatch_errors() {
        // Single item whose assignment is labeled item_index=1 (should be 0):
        // the order guard must fire.
        let items = vec![synthetic_item()];
        let workers = disaggregated(vec![EncodeWorkerAssignment {
            item_index: 1,
            worker: encode_worker("enc-a", 9001),
        }]);

        // Avoid unwrap_err (EncodePlan is not Debug); match the Err directly.
        let err = match plan_encode_jobs(items, &workers) {
            Ok(_) => panic!("expected order-mismatch error"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("order mismatch"), "got: {err}");
    }

    #[test]
    fn plan_encode_jobs_happy_path_builds_bootstrap_info() {
        let items = vec![synthetic_item(), synthetic_item()];
        let workers = disaggregated(vec![
            EncodeWorkerAssignment {
                item_index: 0,
                worker: encode_worker("enc-a", 9001),
            },
            EncodeWorkerAssignment {
                item_index: 1,
                worker: encode_worker("enc-b", 9002),
            },
        ]);

        let plan = plan_encode_jobs(items, &workers).unwrap();
        let (bootstrap_info, dispatch) = plan.into_parts();

        assert_eq!(dispatch.len(), 2);
        assert_eq!(bootstrap_info.len(), 2);

        assert_eq!(bootstrap_info[0].item_index, 0);
        assert_eq!(bootstrap_info[0].bootstrap_host, "enc-a");
        assert_eq!(bootstrap_info[0].bootstrap_port, 9001);

        assert_eq!(bootstrap_info[1].item_index, 1);
        assert_eq!(bootstrap_info[1].bootstrap_host, "enc-b");
        assert_eq!(bootstrap_info[1].bootstrap_port, 9002);
    }

    #[test]
    fn plan_encode_jobs_defaults_bootstrap_port_when_unset() {
        // A worker without an explicit bootstrap_port falls back to
        // DEFAULT_BOOTSTRAP_PORT in the bootstrap info.
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://enc-c:8080")
                .worker_type(WorkerType::Encode)
                .build(),
        ) as Arc<dyn Worker>;
        let workers = disaggregated(vec![EncodeWorkerAssignment {
            item_index: 0,
            worker,
        }]);

        let plan = plan_encode_jobs(vec![synthetic_item()], &workers).unwrap();
        let (bootstrap_info, _) = plan.into_parts();
        assert_eq!(
            bootstrap_info[0].bootstrap_port,
            DEFAULT_BOOTSTRAP_PORT as i32
        );
    }
}
