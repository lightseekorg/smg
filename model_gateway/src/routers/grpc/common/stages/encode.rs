//! Encode stage: EPD fan-out of image embeddings to encode workers.
//!
//! Runs only in the EPD (`new_epd`) pipeline, after client acquisition and
//! before request building. It splits the request's preprocessed multimodal
//! payload into one piece per image (Option A: each image is encoded
//! independently and may land on a different encode worker), dispatches an
//! `Encode` RPC per image to a vision-tower-only worker, and stashes the
//! resulting per-item `EncodeItemHandshake`s for request building to inject
//! into the prefill `GenerateRequest`. The embeddings themselves travel encode
//! -> prefill over Mooncake, keyed by `bootstrap_room`; the gateway never sees
//! them.

use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use axum::response::Response;
use futures::future::join_all;
use rand::Rng;
use smg_grpc_client::{
    tokenspeed_encoder::{tokenspeed_encoder_proto as enc, TokenSpeedEncoderClient},
    tokenspeed_proto,
};
use tracing::{debug, error};
use uuid::Uuid;

use super::PipelineStage;
use crate::{
    routers::{
        error,
        grpc::{
            context::{PreparationOutput, RequestContext, WorkerSelection},
            multimodal::{assemble_tokenspeed_from_split, split_preprocessed_per_item},
        },
    },
    worker::DEFAULT_BOOTSTRAP_PORT,
};

/// Process-global cursor so single-image requests rotate across the encode pool
/// instead of pinning every request to encode_pool[0] (encode data-parallel).
/// Relaxed is fine: this is a load-balancing hint, not a synchronization point;
/// usize wrap is harmless because every use is taken `% pool_len`.
static ENCODE_RR_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Encode stage: split multimodal items per image, dispatch encode RPCs, and
/// record the per-item prefill handshakes.
pub(crate) struct EncodeStage;

#[async_trait]
impl PipelineStage for EncodeStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Only EPD (Triple) selection carries an encode pool. Anything else means
        // this stage was wired into a non-EPD pipeline by mistake; no-op safely.
        let encode_pool = match ctx.state.workers.as_ref() {
            Some(WorkerSelection::Triple { encode_pool, .. }) => encode_pool.clone(),
            _ => return Ok(None),
        };
        if encode_pool.is_empty() {
            error!(
                function = "EncodeStage::execute",
                "EPD selection has an empty encode pool"
            );
            return Err(error::internal_error(
                "epd_empty_encode_pool",
                "EPD selection has no encode workers",
            ));
        }

        // Borrow the preprocessed multimodal intermediate and split it per image.
        // Text-only requests (no multimodal) are a graceful no-op: the prefill
        // simply runs without an encode handshake.
        let (splits, im_token_id) = {
            let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
                error!(
                    function = "EncodeStage::execute",
                    "Preparation stage not completed"
                );
                error::internal_error(
                    "preparation_stage_not_completed",
                    "Preparation stage not completed",
                )
            })?;
            let intermediate = match prep {
                PreparationOutput::Chat {
                    processed_messages, ..
                }
                | PreparationOutput::Messages {
                    processed_messages, ..
                } => processed_messages.multimodal_intermediate.as_ref(),
                _ => None,
            };
            let Some(intermediate) = intermediate else {
                return Ok(None);
            };
            let splits = split_preprocessed_per_item(
                &intermediate.preprocessed,
                &intermediate.field_layouts,
            )
            .map_err(|e| {
                error!(function = "EncodeStage::execute", error = %e, "Failed to split multimodal payload per item");
                error::internal_error("mm_split_failed", format!("Failed to split multimodal payload: {e}"))
            })?;
            (splits, intermediate.im_token_id)
        };

        if splits.is_empty() {
            return Ok(None);
        }

        // One Encode RPC per image: assign a worker round-robin, mint a unique
        // rendezvous room, build the request, and record the prefill handshake.
        let pool_len = encode_pool.len();
        let mut handshakes: Vec<tokenspeed_proto::EncodeItemHandshake> =
            Vec::with_capacity(splits.len());
        let mut sends = Vec::with_capacity(splits.len());

        // Reserve a contiguous block of cursor slots for this request's images so a
        // multi-image request still spreads and consecutive requests continue where
        // the previous one stopped (cross-request balance for single-image requests).
        let num_images = splits.len();
        let rr_base = ENCODE_RR_CURSOR.fetch_add(num_images.max(1), Ordering::Relaxed);

        for (global_index, split) in splits.into_iter().enumerate() {
            let worker = encode_pool[(rr_base + global_index) % pool_len].clone();
            let bootstrap_room = rand::rng().random_range(0..i32::MAX);

            let mm_inputs = assemble_tokenspeed_from_split(split, im_token_id).into_proto();
            let request = enc::EncodeRequest {
                request_id: format!("encode-{}", Uuid::now_v7()),
                mm_inputs: Some(mm_inputs),
                // Single image per RPC, so its local index is always 0; the room
                // is the link to this image's prefill handshake below.
                items: vec![enc::EncodeItemAssignment {
                    item_index: 0,
                    bootstrap_room,
                }],
            };

            // The handshake points the prefill at THIS worker's Mooncake bootstrap
            // endpoint (host/port), not its gRPC URL. One handshake per image: when
            // prefill_tp > encode_tp the engine broadcasts this single (host,port,room)
            // to all N prefill ranks (N = prefill_tp / encode_tp, engine-derived), so
            // the gateway stays rank/N-agnostic.
            handshakes.push(tokenspeed_proto::EncodeItemHandshake {
                item_index: global_index as u32,
                bootstrap_room,
                bootstrap_port: worker.bootstrap_port().unwrap_or(DEFAULT_BOOTSTRAP_PORT) as i32,
                bootstrap_host: worker.bootstrap_host().to_string(),
            });

            let endpoint = worker.url().to_string();
            sends.push(send_encode_rpc(endpoint, request));
        }

        // Fire all encode RPCs concurrently. If ANY fails we must return an error
        // BEFORE request building / execution: a prefill dispatched without its
        // encode peer would hang forever waiting on the Mooncake room.
        for result in join_all(sends).await {
            if let Err(message) = result {
                error!(
                    function = "EncodeStage::execute",
                    error = %message,
                    "Encode RPC failed; aborting before prefill dispatch"
                );
                return Err(error::internal_error("encode_dispatch_failed", message));
            }
        }

        debug!(
            num_images = handshakes.len(),
            num_encode_workers = pool_len,
            "EPD encode dispatch complete"
        );
        ctx.state.encode_handshake = Some(handshakes);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "Encode"
    }
}

/// Connect to one encode worker and dispatch its `Encode` RPC. Returns a
/// human-readable error string on connect/RPC failure or worker rejection.
async fn send_encode_rpc(endpoint: String, request: enc::EncodeRequest) -> Result<(), String> {
    // Reuse a pooled HTTP/2 channel per endpoint instead of dialing afresh on
    // every image's RPC (the prefill/decode legs already reuse their connection).
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
