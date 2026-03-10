//! Shared response collection logic
//!
//! This module contains common logic for collecting responses from execution results.
//! Both regular and harmony processors use these functions to avoid duplication.

use axum::response::Response;
use tracing::{error as trace_error, warn};

use crate::routers::{
    error,
    grpc::{
        context::ExecutionResult,
        proto_wrapper::{
            collect_request_stats, ProtoGenerateComplete, ProtoRequestStats, ProtoResponseVariant,
            ProtoStream,
        },
        utils::tonic_ext::TonicStatusExt,
    },
};
use crate::observability::events::UnifiedRequestStats;

pub(crate) struct CollectedResponses {
    pub completes: Vec<ProtoGenerateComplete>,
    pub request_stats: Option<UnifiedRequestStats>,
}

/// Collect and merge responses from execution result
///
/// Handles both Single and Dual (prefill-decode) execution modes.
/// For Dual mode, merges prefill input_logprobs into decode responses if requested.
///
/// # Arguments
/// * `execution_result` - The execution result containing stream(s)
/// * `merge_logprobs` - Whether to merge prefill input_logprobs (for chat with logprobs=true)
///
/// # Returns
/// Vector of GenerateComplete responses, one per index (n parameter)
pub(crate) async fn collect_responses(
    execution_result: ExecutionResult,
    merge_logprobs: bool,
    enable_request_statistics: bool,
) -> Result<CollectedResponses, Response> {
    let collected = match execution_result {
        ExecutionResult::Single { mut stream } => {
            let responses =
                collect_stream_responses(&mut stream, "Single", enable_request_statistics).await?;
            stream.mark_completed();
            responses
        }
        ExecutionResult::Dual {
            mut prefill,
            decode,
        } => {
            // Collect prefill for input_logprobs (don't mark completed yet)
            let prefill_collected = collect_stream_responses(
                &mut prefill,
                "Prefill",
                enable_request_statistics,
            )
            .await?;

            // Collect decode for actual output (don't mark completed yet)
            let mut decode_stream = *decode;
            let mut decode_collected = collect_stream_responses(
                &mut decode_stream,
                "Decode",
                enable_request_statistics,
            )
            .await?;

            // Mark both streams as completed now that both succeeded
            prefill.mark_completed();
            decode_stream.mark_completed();

            // Merge prefill input_logprobs if requested
            if merge_logprobs {
                merge_prefill_logprobs(
                    &prefill_collected.completes,
                    &mut decode_collected.completes,
                );
            }

            let request_stats = decode_collected
                .request_stats
                .or(prefill_collected.request_stats);

            CollectedResponses {
                completes: decode_collected.completes,
                request_stats,
            }
        }
        ExecutionResult::Embedding { .. } => {
            // Embeddings do not support this path (no generate complete response)
            return Err(error::internal_error(
                "invalid_execution_mode",
                "Embedding result encountered in response collection",
            ));
        }
    };

    if collected.completes.is_empty() {
        return Err(error::internal_error(
            "no_responses_from_server",
            "No responses from server",
        ));
    }

    Ok(collected)
}

/// Merge prefill input_logprobs into decode responses
///
/// Takes input_logprobs from the first prefill response and copies them
/// into all decode responses. This is used in PD mode when logprobs are requested.
/// Only works with SGLang (vLLM doesn't support PD mode).
fn merge_prefill_logprobs(
    prefill_responses: &[ProtoGenerateComplete],
    decode_responses: &mut [ProtoGenerateComplete],
) {
    // Only SGLang supports PD mode and has input_logprobs
    if let Some(ProtoGenerateComplete::Sglang(prefill_first)) = prefill_responses.first() {
        // Use ref to borrow input_logprobs instead of cloning upfront
        // This avoids one allocation when the Option is Some
        if let Some(ref prefill_input_logprobs) = prefill_first.input_logprobs {
            for response in decode_responses.iter_mut() {
                if let ProtoGenerateComplete::Sglang(decode_resp) = response {
                    decode_resp.input_logprobs = Some(prefill_input_logprobs.clone());
                }
            }
        }
    }
}

/// Collect all complete responses from a gRPC stream, discarding chunks.
async fn collect_stream_responses(
    stream: &mut ProtoStream,
    worker_name: &str,
    enable_request_statistics: bool,
) -> Result<CollectedResponses, Response> {
    let mut all_responses = Vec::new();
    let mut stream_request_stats: Vec<ProtoRequestStats> = Vec::new();

    while let Some(response) = stream.next().await {
        match response {
            Ok(gen_response) => {
                match gen_response.into_response() {
                    ProtoResponseVariant::Complete(complete) => {
                        all_responses.push(complete);
                    }
                    ProtoResponseVariant::Error(err) => {
                        // In-band error (legacy): backends should use context.abort() instead.
                        // Kept for backward compatibility during the transition.
                        warn!(function = "collect_stream_responses", worker = %worker_name, error = %err.message(), "Worker sent in-band error (legacy path, backend should use context.abort)");
                        // Don't mark as completed - let Drop send abort for error cases
                        return Err(error::internal_error(
                            "worker_generation_failed",
                            format!("{} generation failed: {}", worker_name, err.message()),
                        ));
                    }
                    ProtoResponseVariant::Chunk(_chunk) => {
                        // Streaming chunk - no action needed
                    }
                    ProtoResponseVariant::RequestStats(request_stats) => {
                        if enable_request_statistics {
                            stream_request_stats.push(request_stats);
                        }
                    }
                    ProtoResponseVariant::None => {
                        // Empty response - no action needed
                    }
                }
            }
            Err(e) => {
                trace_error!(function = "collect_stream_responses", worker = %worker_name, grpc_code = ?e.code(), error = ?e, "Worker stream error");
                // Don't mark as completed - let Drop send abort for error cases
                return Err(e.to_http_error(
                    "worker_stream_failed",
                    format!("{worker_name} stream failed: {}", e.message()),
                ));
            }
        }
    }

    let request_stats = if enable_request_statistics {
        collect_request_stats(&all_responses, &stream_request_stats)
    } else {
        None
    };

    Ok(CollectedResponses {
        completes: all_responses,
        request_stats,
    })
}
