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

use super::{encode_selection, PipelineStage};
use crate::{
    routers::{
        error,
        grpc::{
            context::{PreparationOutput, RequestContext, WorkerSelection},
            multimodal::{assemble_tokenspeed, MultimodalIntermediate},
        },
    },
    worker::DEFAULT_BOOTSTRAP_PORT,
};

/// Process-global cursor for the `RoundRobin` encode routing policy so
/// single-image requests rotate across the encode pool instead of pinning every
/// request to encode_pool[0] (encode data-parallel). The default `CacheAffinity`
/// policy ignores this and routes by content hash (see `encode_selection`).
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

        // Assemble the itemized multimodal payload once (one MultimodalItem per
        // image, each carrying its own encoder_input + grid_thw), then fan the
        // items out across the encode pool below. We borrow the intermediate so
        // request building still consumes it unchanged. Text-only requests (no
        // multimodal) are a graceful no-op: the prefill runs without a handshake.
        let items = {
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
            let Some(MultimodalIntermediate::Precomputed(precomputed)) = intermediate else {
                return Ok(None);
            };
            // skip_pixel_values = false: the encode worker NEEDS the encoder_input
            // (it runs the vision tower). workers = None -> wire dtype from
            // SMG_TOKENSPEED_*_ENCODER_INPUT_DTYPE or the bf16 default.
            assemble_tokenspeed(precomputed, None, false)
                .map_err(|e| {
                    error!(function = "EncodeStage::execute", error = %e, "Failed to assemble multimodal payload for encode");
                    error::internal_error("mm_assemble_failed", format!("Failed to assemble multimodal payload: {e}"))
                })?
                .into_proto()
                .items
        };

        if items.is_empty() {
            return Ok(None);
        }

        // One Encode RPC per image: pick a worker (content-hash affinity by
        // default, or round-robin), mint a unique rendezvous room, build the
        // request, and record the prefill handshake.
        let pool_len = encode_pool.len();
        let mut handshakes: Vec<tokenspeed_proto::EncodeItemHandshake> =
            Vec::with_capacity(items.len());
        let mut sends: Vec<tokio::task::JoinHandle<Result<(), String>>> =
            Vec::with_capacity(items.len());

        let policy = encode_selection::routing_policy();

        // Stable, URL-sorted view of the pool so content-hash affinity maps the
        // same image to the same worker across requests for a fixed pool. Only
        // CacheAffinity uses it; RoundRobin routes on the raw pool order, so skip
        // the allocation + sort there.
        let order: Vec<usize> = match policy {
            encode_selection::EncodeRoutingPolicy::CacheAffinity => {
                let mut o: Vec<usize> = (0..pool_len).collect();
                o.sort_by(|&a, &b| encode_pool[a].url().cmp(encode_pool[b].url()));
                o
            }
            encode_selection::EncodeRoutingPolicy::RoundRobin => Vec::new(),
        };

        // Reserve a contiguous block of cursor slots for this request's images so a
        // multi-image request still spreads and consecutive requests continue where
        // the previous one stopped (cross-request balance for single-image requests).
        // Only the RoundRobin policy consumes this; CacheAffinity routes by hash.
        let num_images = items.len();
        let rr_base = ENCODE_RR_CURSOR.fetch_add(num_images.max(1), Ordering::Relaxed);

        for (global_index, mut item) in items.into_iter().enumerate() {
            // Select this image's encode worker.
            let worker = match policy {
                encode_selection::EncodeRoutingPolicy::RoundRobin => {
                    encode_pool[(rr_base + global_index) % pool_len].clone()
                }
                encode_selection::EncodeRoutingPolicy::CacheAffinity => {
                    let slot = encode_selection::affinity_slot(&item.content_hash, pool_len);
                    encode_pool[order[slot]].clone()
                }
            };
            // 63-bit room: no in-flight dedup, so a 2^31 space birthday-collides
            // under load (silent embedding cross-wire). See the proto field doc.
            let bootstrap_room = rand::rng().random_range(0..i64::MAX);

            // The handshake points the prefill at THIS worker's Mooncake bootstrap
            // endpoint (host/port), not its gRPC URL. One handshake per image: when
            // prefill_tp > encode_tp the engine broadcasts this single (host,port,room)
            // to all N prefill ranks (N = prefill_tp / encode_tp, engine-derived), so
            // the gateway stays rank/N-agnostic. Built on the loop thread (cheap, no
            // tensor copy) so handshake host/port/room stays paired with the worker
            // this image's RPC targets below (both read the same loop iteration).
            handshakes.push(tokenspeed_proto::EncodeItemHandshake {
                item_index: global_index as u32,
                bootstrap_room,
                bootstrap_port: worker.bootstrap_port().unwrap_or(DEFAULT_BOOTSTRAP_PORT) as i32,
                bootstrap_host: worker.bootstrap_host().to_string(),
            });

            let endpoint = worker.url().to_string();
            // Spawn the proto framing + RPC (and the RDMA export) onto the multi-thread
            // runtime so N images transmit in PARALLEL instead of one-after-another.
            // (The per-item encoder_input was already serialized by assemble_tokenspeed
            // above; only framing/transport is deferred here.)
            sends.push(tokio::spawn(async move {
                // EPD RDMA (M2): export this image's encoder_input over NIXL and swap the
                // inline payload for a remote handle so the encode worker PULLs it instead
                // of receiving ~10-50MB in this gRPC frame. Keyed by bootstrap_room (the
                // worker tags its free-notif with it). Any export failure re-attaches the
                // inline bytes -> no behaviour change. Off unless SMG_MM_PIXEL_RDMA is set.
                if crate::routers::grpc::rdma::rdma_enabled() {
                    if let Some(encoder_input) = item.encoder_input.as_mut() {
                        if let Some(tokenspeed_proto::tensor_data::Payload::Inline(bytes)) =
                            encoder_input.payload.take()
                        {
                            let nbytes = bytes.len() as u64;
                            match crate::routers::grpc::rdma::export_pixel_buffer(
                                bootstrap_room,
                                bytes,
                            ) {
                                Ok(descriptor) => {
                                    encoder_input.payload =
                                        Some(tokenspeed_proto::tensor_data::Payload::Remote(
                                            tokenspeed_proto::RemoteTensorHandle {
                                                transport: "nixl".to_string(),
                                                descriptor,
                                                nbytes,
                                            },
                                        ));
                                }
                                Err(bytes) => {
                                    encoder_input.payload = Some(
                                        tokenspeed_proto::tensor_data::Payload::Inline(bytes),
                                    );
                                }
                            }
                        }
                    }
                }
                let request = enc::EncodeRequest {
                    request_id: format!("encode-{}", Uuid::now_v7()),
                    // One MultimodalItem per Encode RPC; the room links it to this
                    // image's prefill handshake recorded above.
                    mm_inputs: Some(tokenspeed_proto::MultimodalInputs { items: vec![item] }),
                    // One assignment per image; the room links to this image's
                    // prefill handshake recorded above.
                    items: vec![enc::EncodeItemAssignment { bootstrap_room }],
                };
                send_encode_rpc(endpoint, request).await
            }));
        }

        // Record the handshakes NOW (rooms/host/port are gateway-minted above, not
        // returned by the encode RPC) so RequestBuilding + RequestExecution proceed
        // immediately and dispatch the prefill while the encode RPCs are still
        // in flight. The prefill's embedding-receiver then registers WHILE the encode
        // is transferring pixels / running the ViT, so the encode sender no longer
        // PARKS ~400ms/image waiting for a late receiver -- that park (the prefill
        // dispatched only AFTER a fully-awaited Encode stage) was the dominant EPD
        // per-image cost. Measured: ViT ~30ms, RDMA ~20ms, but enc_send_issue ->
        // enc_send_start (the park) ~358-470ms.
        debug!(
            num_images = handshakes.len(),
            num_encode_workers = pool_len,
            policy = ?policy,
            "EPD encode dispatch issued (non-blocking; prefill registers concurrently)"
        );
        ctx.state.encode_handshake = Some(handshakes);

        // Supervise the in-flight encode RPCs in the background instead of blocking
        // the stage on join_all. A failed/lost embedding is caught by the prefill's
        // embedding-receive timeout (engine-side), which aborts the request and
        // propagates the failure to the decode (so the client gets an error, not a
        // hang). Healthy workers are pre-filtered by WorkerSelection, so a mid-flight
        // RPC failure here is rare.
        tokio::spawn(async move {
            for join_res in join_all(sends).await {
                match join_res {
                    Ok(Ok(())) => {}
                    Ok(Err(message)) => {
                        error!(
                            function = "EncodeStage::execute",
                            error = %message,
                            "Encode RPC failed after prefill dispatch; embedding-receive timeout will abort the request"
                        );
                    }
                    Err(join_err) => {
                        error!(
                            function = "EncodeStage::execute",
                            error = %join_err,
                            "Encode dispatch task panicked after prefill dispatch"
                        );
                    }
                }
            }
        });
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
