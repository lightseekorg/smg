//! Common helper functions shared across stages

use std::sync::Arc;

use llm_tokenizer::traits::Tokenizer;
use rand::RngExt;
use smg_grpc_client::{
    mlx_proto,
    sglang_proto::{self, DisaggregatedParams},
    tokenspeed_proto, vllm_proto,
};
use tracing::{debug, warn};

use crate::{
    routers::grpc::{
        context::{ClientSelection, RequestType, WorkerSelection},
        epd_encode::{self, EncodePlan},
        multimodal::MultimodalIntermediate,
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

pub(crate) fn plan_epd_encode(
    intermediate: &MultimodalIntermediate,
    clients: &ClientSelection,
    workers: Option<&WorkerSelection>,
) -> anyhow::Result<Option<EncodePlan>> {
    if workers
        .and_then(WorkerSelection::encode_assignments)
        .is_none_or(|assignments| assignments.is_empty())
    {
        return Ok(None);
    }

    let plan = epd_encode::build_plan_from_intermediate(intermediate, Some(clients), workers)?;
    if plan.is_empty() {
        Ok(None)
    } else {
        Ok(Some(plan))
    }
}

/// Resolve string `stop` sequences for SGLang gRPC workers.
///
/// SMG's SGLang workers run with `skip_tokenizer_init=True` (the router owns
/// the tokenizer). Upstream SGLang's `SamplingParams.verify()` rejects string
/// `stop` sequences in that mode — it needs a tokenizer to decode generated
/// tokens back to text for matching — and returns
/// `stop=[...] is unavailable when skip_tokenizer_init=True`, which surfaces to
/// the caller as a 400. That breaks OpenAI API compatibility for any request
/// carrying the (documented, first-class) `stop` parameter (see issue #227).
///
/// The router already matches string stops itself via `StopSequenceDecoder`
/// (it detokenizes worker output and trims the stop text), so the worker never
/// needs the raw strings. This helper therefore:
///   1. Clears the string `stop` list on the SGLang request so the worker stops
///      rejecting it — this alone fixes the 400 and preserves correct output
///      because the router-side decoder still trims the text; and
///   2. As an optimization, encodes any stop string that maps to a *single*
///      token into `stop_token_ids` so the worker can still halt generation
///      early for the common case (e.g. `["."]`, `["\n"]`). The proto
///      `stop_token_ids` field is a flat list of single token IDs, so a
///      multi-token stop string cannot be represented there — pushing its
///      sub-tokens would stop generation far too eagerly (on any one of them).
///      Multi-token (and empty) stops are left entirely to the router-side
///      decoder: the worker generates until EOS/max_tokens and the router
///      truncates the text at the stop.
///
/// # Scope — call only where a trimming decoder runs in every response mode
///
/// This MUST be invoked only from the chat/completions/messages request-building
/// stages, whose streaming *and* non-streaming handlers both build a
/// `StopSequenceDecoder`. It is deliberately NOT called from:
///   - the Harmony (gpt-oss) pipeline — no decoder, and injecting a user stop
///     token can truncate before `<|return|>`/`<|call|>` (corrupts parsing); and
///   - the native `/generate` streaming path — no decoder, so a cleared stop
///     would stream untrimmed text past the stop.
/// Those paths keep returning the pre-existing 400 for SGLang string stops until
/// a decoder exists there.
///
/// # Known limitations (see issue #227 review)
///   - Multi-token stops don't halt the worker, so it runs to EOS/max_tokens:
///     extra compute/latency, and `usage.completion_tokens` reports the
///     worker's full count — tokens generated past the stop are billed even
///     though the router trims them from the visible text.
///   - `finish_reason` then reflects the worker's reason (often `length`)
///     rather than `stop` on the chat/messages paths. The completion endpoint
///     diverges internally: its streaming handler honors the router-side
///     decoder and reports `stop`, while non-streaming forwards the worker's
///     `length` — the same request can return different finish_reasons by
///     stream mode.
///   - A converted single-token stop makes the worker report `matched_stop` as a
///     numeric token id (not the stop string), so the OpenAI `matched_stop` /
///     Anthropic `stop_reason` fields differ from the vLLM path.
///   - With `ignore_eos=true` SGLang skips all `stop_token_ids` matching, so
///     the single-token conversion would be inert; it is skipped in that mode
///     and the router-side decoder alone handles the stop.
///   - On SentencePiece/BPE tokenizers a bare `"."` may encode to a
///     leading-space variant (`▁.`), so the single-token optimization may not
///     fire mid-sentence; correctness still holds via the router-side decoder.
///
/// Only SGLang is affected in the current normal gRPC deployment: the vLLM
/// servicer forces `detokenize=bool(stop)`, TokenSpeed initializes an
/// engine-side tokenizer by default, TRT-LLM tokenizes stop words server-side,
/// and the MLX proto has no string-`stop` field. Non-SGLang requests are left
/// untouched.
pub(crate) fn resolve_sglang_string_stops(
    request: &mut ProtoGenerateRequest,
    tokenizer: Option<&Arc<dyn Tokenizer>>,
) {
    let ProtoGenerateRequest::Sglang(req) = request else {
        return;
    };
    let Some(params) = req.sampling_params.as_mut() else {
        return;
    };
    if params.stop.is_empty() {
        return;
    }

    // Always drop the string stops from the SGLang request: the worker cannot
    // handle them under skip_tokenizer_init and the router-side decoder is the
    // source of truth for string-stop matching/trimming.
    let stop_strings = std::mem::take(&mut params.stop);

    // With ignore_eos=true SGLang skips ALL stop_token_ids matching
    // (Req._check_token_based_finish returns False immediately), so a converted
    // id would be inert at the worker. Skip the conversion; the router-side
    // decoder still matches and trims the string stop.
    if params.ignore_eos {
        debug!(
            "ignore_eos=true: skipping SGLang single-token stop conversion \
             (the worker ignores stop_token_ids in this mode)"
        );
        return;
    }

    // Without a tokenizer we cannot encode (not expected on the gRPC path,
    // which always resolves one to tokenize the prompt). Still safe: the
    // strings are dropped above so the worker no longer 400s.
    let Some(tokenizer) = tokenizer else {
        warn!(
            "No tokenizer available to encode SGLang stop sequences; \
             relying on router-side stop decoder only"
        );
        return;
    };

    for stop in stop_strings {
        if stop.is_empty() {
            continue;
        }
        // add_special_tokens=false: we want the literal token(s) for the stop
        // string, not a BOS/EOS-wrapped encoding.
        match tokenizer.encode(&stop, false) {
            Ok(encoding) => {
                let ids = encoding.token_ids();
                if ids.len() == 1 {
                    let id = ids[0];
                    if !params.stop_token_ids.contains(&id) {
                        params.stop_token_ids.push(id);
                    }
                } else {
                    // 0 tokens (unknown/whitespace-only) or multi-token: the
                    // router-side StopSequenceDecoder handles these.
                    debug!(
                        stop = %stop,
                        token_count = ids.len(),
                        "SGLang stop sequence not single-token; \
                         handled by router-side stop decoder"
                    );
                }
            }
            Err(e) => {
                warn!(
                    stop = %stop,
                    error = %e,
                    "Failed to encode SGLang stop sequence; \
                     relying on router-side stop decoder"
                );
            }
        }
    }
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
mod tests {
    use std::sync::Arc;

    // ---- resolve_sglang_string_stops (issue #227) ----

    use llm_tokenizer::{mock::MockTokenizer, traits::Tokenizer};
    use smg_grpc_client::{sglang_proto, vllm_proto};

    use super::{resolve_sglang_string_stops, ProtoGenerateRequest};

    fn mock_tokenizer() -> Arc<dyn Tokenizer> {
        // MockTokenizer vocab: "." => 6, "Hello" => 1, "world" => 2, ...
        // Its `encode` splits on whitespace and looks up each known word, so
        // "." => [6] (single-token) and "Hello world" => [1, 2] (multi-token).
        Arc::new(MockTokenizer::new())
    }

    fn sglang_request_with_stops(
        stop: Vec<&str>,
        stop_token_ids: Vec<u32>,
    ) -> ProtoGenerateRequest {
        ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
            sampling_params: Some(sglang_proto::SamplingParams {
                stop: stop.into_iter().map(str::to_string).collect(),
                stop_token_ids,
                ..Default::default()
            }),
            ..Default::default()
        }))
    }

    fn sglang_params(req: &ProtoGenerateRequest) -> &sglang_proto::SamplingParams {
        match req {
            ProtoGenerateRequest::Sglang(r) => r.sampling_params.as_ref().unwrap(),
            _ => panic!("expected SGLang request"),
        }
    }

    #[test]
    fn resolve_sglang_stops_single_token_becomes_stop_token_id() {
        let mut req = sglang_request_with_stops(vec!["."], vec![]);
        resolve_sglang_string_stops(&mut req, Some(&mock_tokenizer()));

        let params = sglang_params(&req);
        // String stop dropped so the worker (skip_tokenizer_init) won't 400.
        assert!(params.stop.is_empty(), "string stop should be cleared");
        // "." (token 6) forwarded as a stop token id for early worker stopping.
        assert_eq!(params.stop_token_ids, vec![6]);
    }

    #[test]
    fn resolve_sglang_stops_multi_token_relies_on_router_decoder() {
        // "Hello world" => [1, 2]: multi-token can't be a flat stop_token_id,
        // so it must NOT be forwarded (would over-eagerly stop on any subtoken).
        let mut req = sglang_request_with_stops(vec!["Hello world"], vec![]);
        resolve_sglang_string_stops(&mut req, Some(&mock_tokenizer()));

        let params = sglang_params(&req);
        assert!(params.stop.is_empty(), "string stop should be cleared");
        assert!(
            params.stop_token_ids.is_empty(),
            "multi-token stop must not be forwarded as stop_token_ids"
        );
    }

    #[test]
    fn resolve_sglang_stops_mixed_only_single_token_forwarded() {
        let mut req = sglang_request_with_stops(vec![".", "Hello world"], vec![]);
        resolve_sglang_string_stops(&mut req, Some(&mock_tokenizer()));

        let params = sglang_params(&req);
        assert!(params.stop.is_empty());
        assert_eq!(params.stop_token_ids, vec![6]);
    }

    #[test]
    fn resolve_sglang_stops_preserves_existing_stop_token_ids_and_dedups() {
        // Pre-existing stop_token_ids must be preserved; "." (6) already present
        // must not be duplicated.
        let mut req = sglang_request_with_stops(vec!["."], vec![6, 42]);
        resolve_sglang_string_stops(&mut req, Some(&mock_tokenizer()));

        let params = sglang_params(&req);
        assert!(params.stop.is_empty());
        assert_eq!(
            params.stop_token_ids,
            vec![6, 42],
            "no duplicate for existing id"
        );
    }

    #[test]
    fn resolve_sglang_stops_empty_and_unknown_strings_add_nothing() {
        // "" is skipped; "unknowntoken" encodes to [] under the mock vocab.
        let mut req = sglang_request_with_stops(vec!["", "unknowntoken"], vec![]);
        resolve_sglang_string_stops(&mut req, Some(&mock_tokenizer()));

        let params = sglang_params(&req);
        assert!(params.stop.is_empty(), "string stops still cleared");
        assert!(params.stop_token_ids.is_empty());
    }

    #[test]
    fn resolve_sglang_stops_without_tokenizer_still_clears_strings() {
        // No tokenizer: cannot encode, but the string stops must still be
        // dropped so the worker does not 400.
        let mut req = sglang_request_with_stops(vec!["."], vec![]);
        resolve_sglang_string_stops(&mut req, None);

        let params = sglang_params(&req);
        assert!(params.stop.is_empty());
        assert!(params.stop_token_ids.is_empty());
    }

    #[test]
    fn resolve_sglang_stops_noop_when_no_string_stops() {
        let mut req = sglang_request_with_stops(vec![], vec![7]);
        resolve_sglang_string_stops(&mut req, Some(&mock_tokenizer()));

        let params = sglang_params(&req);
        assert!(params.stop.is_empty());
        assert_eq!(params.stop_token_ids, vec![7], "unrelated ids untouched");
    }

    #[test]
    fn resolve_sglang_stops_leaves_non_sglang_untouched() {
        // vLLM handles string stops fine (detokenize=bool(stop)) — must not be
        // mutated by the SGLang-specific fix.
        let mut req = ProtoGenerateRequest::Vllm(Box::new(vllm_proto::GenerateRequest {
            sampling_params: Some(vllm_proto::SamplingParams {
                stop: vec![".".to_string()],
                ..Default::default()
            }),
            ..Default::default()
        }));
        resolve_sglang_string_stops(&mut req, Some(&mock_tokenizer()));

        match &req {
            ProtoGenerateRequest::Vllm(r) => {
                let params = r.sampling_params.as_ref().unwrap();
                assert_eq!(params.stop, vec![".".to_string()], "vLLM stop preserved");
                assert!(params.stop_token_ids.is_empty());
            }
            _ => panic!("expected vLLM request"),
        }
    }

    #[test]
    fn resolve_sglang_stops_two_distinct_single_tokens_both_forwarded() {
        // "." => 6 and "Hello" => 1 are both single-token under the mock vocab;
        // both must be appended (guards against a push -> overwrite regression).
        let mut req = sglang_request_with_stops(vec![".", "Hello"], vec![]);
        resolve_sglang_string_stops(&mut req, Some(&mock_tokenizer()));

        let params = sglang_params(&req);
        assert!(params.stop.is_empty());
        assert_eq!(params.stop_token_ids, vec![6, 1]);
    }

    #[test]
    fn resolve_sglang_stops_ignore_eos_skips_conversion() {
        // SGLang skips ALL stop_token_ids matching when ignore_eos=true
        // (Req._check_token_based_finish returns early), so the conversion
        // would be inert — it must be skipped. The string stop is still
        // cleared so the worker doesn't 400.
        let mut req = ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
            sampling_params: Some(sglang_proto::SamplingParams {
                stop: vec![".".to_string()],
                ignore_eos: true,
                ..Default::default()
            }),
            ..Default::default()
        }));
        resolve_sglang_string_stops(&mut req, Some(&mock_tokenizer()));

        let params = sglang_params(&req);
        assert!(params.stop.is_empty(), "string stop still cleared");
        assert!(
            params.stop_token_ids.is_empty(),
            "no conversion under ignore_eos (worker ignores stop_token_ids)"
        );
    }

    #[test]
    fn resolve_sglang_stops_missing_sampling_params_is_noop() {
        // Guard: SGLang request without sampling_params must not panic.
        let mut req = ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
            sampling_params: None,
            ..Default::default()
        }));
        resolve_sglang_string_stops(&mut req, Some(&mock_tokenizer()));

        match &req {
            ProtoGenerateRequest::Sglang(r) => assert!(r.sampling_params.is_none()),
            _ => panic!("expected SGLang request"),
        }
    }

    /// Wraps [`MockTokenizer`] but encodes `"\n"` to a single token. The shared
    /// mock strips whitespace (`split_whitespace`), which would silently route
    /// the doc's own headline example `["\n"]` through the multi/empty branch
    /// instead of the single-token path.
    struct NewlineTokenizer(MockTokenizer);

    const NEWLINE_TOKEN_ID: u32 = 99;

    impl llm_tokenizer::traits::Encoder for NewlineTokenizer {
        fn encode(
            &self,
            input: &str,
            add_special_tokens: bool,
        ) -> anyhow::Result<llm_tokenizer::traits::Encoding> {
            if input == "\n" {
                return Ok(llm_tokenizer::traits::Encoding::Plain(vec![
                    NEWLINE_TOKEN_ID,
                ]));
            }
            self.0.encode(input, add_special_tokens)
        }

        fn encode_batch(
            &self,
            inputs: &[&str],
            add_special_tokens: bool,
        ) -> anyhow::Result<Vec<llm_tokenizer::traits::Encoding>> {
            inputs
                .iter()
                .map(|i| llm_tokenizer::traits::Encoder::encode(self, i, add_special_tokens))
                .collect()
        }
    }

    impl llm_tokenizer::traits::Decoder for NewlineTokenizer {
        fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
            self.0.decode(token_ids, skip_special_tokens)
        }
    }

    impl Tokenizer for NewlineTokenizer {
        fn vocab_size(&self) -> usize {
            self.0.vocab_size()
        }
        fn get_special_tokens(&self) -> &llm_tokenizer::traits::SpecialTokens {
            self.0.get_special_tokens()
        }
        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.0.token_to_id(token)
        }
        fn id_to_token(&self, id: u32) -> Option<String> {
            self.0.id_to_token(id)
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn resolve_sglang_stops_whitespace_single_token_converted() {
        // "\n" is the doc's headline single-token example; ensure a tokenizer
        // that maps it to one token gets it forwarded as a stop_token_id.
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(NewlineTokenizer(MockTokenizer::new()));
        let mut req = sglang_request_with_stops(vec!["\n"], vec![]);
        resolve_sglang_string_stops(&mut req, Some(&tokenizer));

        let params = sglang_params(&req);
        assert!(params.stop.is_empty());
        assert_eq!(params.stop_token_ids, vec![NEWLINE_TOKEN_ID]);
    }
}
