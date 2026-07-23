//! Common helper functions shared across stages

use std::sync::Arc;

use rand::RngExt;
use smg_grpc_client::{
    mlx_proto,
    sglang_proto::{self, DisaggregatedParams},
    tokenspeed_proto, vllm_proto,
};
use tracing::{debug, warn};

use crate::{
    middleware::{RequestId, TenantRequestMeta},
    routers::grpc::{
        context::{RequestType, WorkerSelection},
        proto_wrapper::ProtoGenerateRequest,
    },
    worker::{
        sampling_defaults::SamplingDefaults, RuntimeType, Worker, DEFAULT_BOOTSTRAP_PORT,
        DEFAULT_SAMPLING_PARAMS_LABEL,
    },
};

#[derive(Clone, Copy, Debug, Default)]
struct SamplingDefaultsMask {
    temperature: bool,
    top_p: bool,
    top_k: bool,
    min_p: bool,
    repetition_penalty: bool,
}

impl SamplingDefaultsMask {
    fn from_request_type(request_type: &RequestType) -> Option<Self> {
        match request_type {
            RequestType::Chat(request) => Some(Self {
                temperature: request.temperature.is_none(),
                top_p: request.top_p.is_none(),
                top_k: request.top_k.is_none(),
                min_p: request.min_p.is_none(),
                repetition_penalty: request.repetition_penalty.is_none(),
            }),
            RequestType::Completion(request) => Some(Self {
                temperature: request.temperature.is_none(),
                top_p: request.top_p.is_none(),
                top_k: request.top_k.is_none(),
                min_p: request.min_p.is_none(),
                repetition_penalty: request.repetition_penalty.is_none(),
            }),
            RequestType::Generate(request) => {
                let params = request.sampling_params.as_ref();
                Some(Self {
                    temperature: params.and_then(|params| params.temperature).is_none(),
                    top_p: params.and_then(|params| params.top_p).is_none(),
                    top_k: params.and_then(|params| params.top_k).is_none(),
                    min_p: params.and_then(|params| params.min_p).is_none(),
                    repetition_penalty: params
                        .and_then(|params| params.repetition_penalty)
                        .is_none(),
                })
            }
            RequestType::Messages(request) => Some(Self {
                temperature: request.temperature.is_none(),
                top_p: request.top_p.is_none(),
                top_k: request.top_k.is_none(),
                // Messages does not expose these knobs, so model defaults are
                // the only source of request-level values for them.
                min_p: true,
                repetition_penalty: true,
            }),
            RequestType::Responses(_) | RequestType::Embedding(_) | RequestType::Classify(_) => {
                None
            }
        }
    }

    fn any(self) -> bool {
        self.temperature || self.top_p || self.top_k || self.min_p || self.repetition_penalty
    }
}

/// The middleware-assigned request id (client-sent via a configured header,
/// else generated; see `middleware/request_id.rs`), carried on the tenant
/// request metadata.
pub(crate) fn middleware_request_id(tenant_meta: Option<&TenantRequestMeta>) -> Option<&str> {
    tenant_meta
        .and_then(|meta| meta.extension::<RequestId>())
        .map(|request_id| request_id.0.as_str())
}

/// Backend request id for any endpoint.
///
/// Priority: the protocol `rid`, else the middleware request id, else a fresh
/// `{prefix}{uuid}`.
///
/// Engine ids must be unique per dispatch — PD retries re-run request
/// building while a NIXL-tagged prefill keeps the id alive until the KV lease
/// expires, and responses tool loops re-execute the pipeline per iteration —
/// so middleware-derived ids always get a per-execution suffix. A bare `rid`
/// outside PD is used exactly: its value is the caller's contract (matching
/// /generate's long-standing behavior).
pub(crate) fn resolve_request_id(
    request_type: &RequestType,
    tenant_meta: Option<&TenantRequestMeta>,
    prefix: &str,
    disaggregated: bool,
) -> String {
    if let Some(rid) = request_type.rid() {
        return if disaggregated {
            format!("{rid}-{}", uuid::Uuid::now_v7())
        } else {
            rid.to_string()
        };
    }
    match middleware_request_id(tenant_meta) {
        Some(request_id) => format!("{request_id}-{}", uuid::Uuid::now_v7()),
        None => format!("{prefix}{}", uuid::Uuid::now_v7()),
    }
}

/// Decode selected-worker sampling defaults from labels.
///
/// In PD mode the decode worker is authoritative because it produces visible
/// output tokens. The resolved request is then sent through the existing PD
/// flow unchanged.
pub(crate) fn sampling_defaults_for_request(
    workers: Option<&WorkerSelection>,
) -> Option<SamplingDefaults> {
    let worker = match workers? {
        WorkerSelection::Single { worker } => worker,
        WorkerSelection::Disaggregated { decode, .. } => decode,
    };
    let json = worker
        .metadata()
        .spec
        .labels
        .get(DEFAULT_SAMPLING_PARAMS_LABEL)?;

    match SamplingDefaults::from_json_str(json) {
        Ok(defaults) => defaults,
        Err(e) => {
            warn!(
                worker_url = %worker.url(),
                error = %e,
                "Ignoring invalid default sampling params label"
            );
            None
        }
    }
}

/// Apply model sampling defaults to a built proto request.
///
/// The proto already contains backend fallback values, so `request_type` is
/// used only as an omission mask: defaults fill fields the user did not set.
pub(crate) fn apply_sampling_defaults_to_generate_request(
    request: &mut ProtoGenerateRequest,
    request_type: &RequestType,
    workers: Option<&WorkerSelection>,
) {
    if matches!(request, ProtoGenerateRequest::Trtllm(_)) {
        return;
    }

    let Some(mask) = SamplingDefaultsMask::from_request_type(request_type) else {
        return;
    };
    if !mask.any() {
        return;
    }

    let Some(defaults) = sampling_defaults_for_request(workers) else {
        return;
    };

    match request {
        ProtoGenerateRequest::Sglang(req) => {
            let Some(params) = req.sampling_params.as_mut() else {
                warn!("Cannot apply sampling defaults to SGLang request without sampling_params");
                return;
            };
            apply_sglang_sampling_defaults(params, defaults, mask);
        }
        ProtoGenerateRequest::Vllm(req) => {
            let Some(params) = req.sampling_params.as_mut() else {
                warn!("Cannot apply sampling defaults to vLLM request without sampling_params");
                return;
            };
            apply_vllm_sampling_defaults(params, defaults, mask);
        }
        ProtoGenerateRequest::Mlx(req) => {
            let Some(params) = req.sampling_params.as_mut() else {
                warn!("Cannot apply sampling defaults to MLX request without sampling_params");
                return;
            };
            apply_mlx_sampling_defaults(params, defaults, mask);
        }
        ProtoGenerateRequest::TokenSpeed(req) => {
            let Some(params) = req.sampling_params.as_mut() else {
                warn!(
                    "Cannot apply sampling defaults to TokenSpeed request without sampling_params"
                );
                return;
            };
            apply_tokenspeed_sampling_defaults(params, defaults, mask);
        }
        ProtoGenerateRequest::Trtllm(_) => {}
    }
}

macro_rules! apply_numeric_default {
    ($params:expr, $defaults:expr, $mask:expr, $field:ident) => {
        if $mask.$field {
            if let Some(value) = $defaults.$field {
                $params.$field = value;
            }
        }
    };
}

macro_rules! apply_unsigned_top_k_default {
    ($params:expr, $defaults:expr, $mask:expr) => {
        if $mask.top_k {
            if let Some(value) = $defaults.top_k {
                $params.top_k = value.max(0) as u32;
            }
        }
    };
}

macro_rules! optional_temperature_sampling_defaults_fn {
    ($fn_name:ident, $params_ty:path) => {
        fn $fn_name(
            params: &mut $params_ty,
            defaults: SamplingDefaults,
            mask: SamplingDefaultsMask,
        ) {
            if mask.temperature {
                if let Some(value) = defaults.temperature {
                    params.temperature = Some(value);
                }
            }
            apply_numeric_default!(params, defaults, mask, top_p);
            apply_unsigned_top_k_default!(params, defaults, mask);
            apply_numeric_default!(params, defaults, mask, min_p);
            apply_numeric_default!(params, defaults, mask, repetition_penalty);
        }
    };
}

fn apply_sglang_sampling_defaults(
    params: &mut sglang_proto::SamplingParams,
    defaults: SamplingDefaults,
    mask: SamplingDefaultsMask,
) {
    apply_numeric_default!(params, defaults, mask, temperature);
    apply_numeric_default!(params, defaults, mask, top_p);
    apply_numeric_default!(params, defaults, mask, top_k);
    apply_numeric_default!(params, defaults, mask, min_p);
    apply_numeric_default!(params, defaults, mask, repetition_penalty);
}

optional_temperature_sampling_defaults_fn!(
    apply_vllm_sampling_defaults,
    vllm_proto::SamplingParams
);
optional_temperature_sampling_defaults_fn!(apply_mlx_sampling_defaults, mlx_proto::SamplingParams);

/// TokenSpeed declares every sampling scalar as `optional` so the servicer
/// can distinguish "client set 0" from "client unset". Apply defaults by
/// writing `Some(value)` rather than the bare value.
fn apply_tokenspeed_sampling_defaults(
    params: &mut tokenspeed_proto::SamplingParams,
    defaults: SamplingDefaults,
    mask: SamplingDefaultsMask,
) {
    macro_rules! apply_opt {
        ($field:ident) => {
            if mask.$field {
                if let Some(value) = defaults.$field {
                    params.$field = Some(value);
                }
            }
        };
    }
    apply_opt!(temperature);
    apply_opt!(top_p);
    apply_opt!(top_k);
    apply_opt!(min_p);
    apply_opt!(repetition_penalty);
}

/// Inject PD bootstrap metadata for SGLang if needed.
///
/// SGLang uses DisaggregatedParams with bootstrap host/port/room.
/// vLLM kv_transfer_params are handled in the request_execution stage.
pub(crate) fn maybe_inject_pd_metadata(
    request: &mut ProtoGenerateRequest,
    workers: &WorkerSelection,
) {
    if let WorkerSelection::Disaggregated {
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

/// Inject prefill->decode rendezvous params for backends that carry them in the
/// generate request.
///
/// The gateway mints one room per request and sends identical params to both the
/// prefill and decode worker (`execute_parallel_pd` clones the request after
/// this stage). Host/port name the PREFILL worker's Mooncake bootstrap server
/// (the KV data source); the decode worker discovers it there by `bootstrap_room`.
/// This KV leg is independent of any per-item encode->prefill bootstrap info.
pub(crate) fn maybe_inject_pd_rendezvous(
    request: &mut ProtoGenerateRequest,
    workers: &WorkerSelection,
) {
    // The KV bootstrap leg is identical for plain PD and EPD; EPD just layers
    // encode assignments on the disaggregated worker selection.
    let (prefill, runtime_type) = match workers {
        WorkerSelection::Disaggregated {
            prefill,
            runtime_type,
            ..
        } => (prefill, runtime_type),
        WorkerSelection::Single { .. } => return,
    };
    if *runtime_type == RuntimeType::TokenSpeed {
        let metadata = prefill.metadata();
        let hostname = metadata.bootstrap_host();
        let bootstrap_port = metadata.bootstrap_port().unwrap_or(DEFAULT_BOOTSTRAP_PORT);
        // 63-bit room: no dedup, keep the space wide so the birthday collision
        // rate stays negligible. See the proto field doc.
        let room_id = rand::rng().random_range(0..i64::MAX);

        request.set_kv_bootstrap_info(hostname.to_string(), bootstrap_port as i32, room_id);

        debug!(
            "Injected PD rendezvous: host={}, port={}, room={}",
            hostname, bootstrap_port, room_id
        );
    }
}

#[cfg(test)]
mod request_id_tests {
    use std::sync::Arc;

    use openai_protocol::chat::ChatCompletionRequest;

    use super::*;
    use crate::tenant::TenantKey;

    fn chat_request_type(rid: Option<&str>) -> RequestType {
        RequestType::Chat(Arc::new(ChatCompletionRequest {
            rid: rid.map(str::to_string),
            ..Default::default()
        }))
    }

    fn meta_with_request_id(id: &str) -> TenantRequestMeta {
        TenantRequestMeta::new(TenantKey::new("test-tenant"))
            .with_extension(RequestId(id.to_string()))
    }

    #[test]
    fn rid_is_used_exactly_outside_pd() {
        let request_type = chat_request_type(Some("client-rid"));
        let meta = meta_with_request_id("chatcmpl-mw");

        let id = resolve_request_id(&request_type, Some(&meta), "chatcmpl-", false);
        assert_eq!(id, "client-rid");
    }

    #[test]
    fn rid_gets_per_attempt_suffix_in_pd() {
        let request_type = chat_request_type(Some("client-rid"));

        let id = resolve_request_id(&request_type, None, "chatcmpl-", true);
        assert!(id.starts_with("client-rid-") && id != "client-rid");
    }

    #[test]
    fn middleware_id_is_base_with_per_execution_suffix() {
        let request_type = chat_request_type(None);
        let meta = meta_with_request_id("chatcmpl-mw");

        let first = resolve_request_id(&request_type, Some(&meta), "chatcmpl-", false);
        let second = resolve_request_id(&request_type, Some(&meta), "chatcmpl-", false);
        assert!(first.starts_with("chatcmpl-mw-"));
        assert!(second.starts_with("chatcmpl-mw-"));
        assert_ne!(first, second);
    }

    #[test]
    fn falls_back_to_prefixed_mint_without_rid_or_middleware_id() {
        let request_type = chat_request_type(None);
        let meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

        let id = resolve_request_id(&request_type, Some(&meta), "chatcmpl-", false);
        assert!(id.starts_with("chatcmpl-"));
    }
}
