//! Request events for observability and monitoring.
//!
//! Events use DEBUG level when OTEL is disabled, INFO when enabled.

use tracing::{debug, event, Level};

use super::otel_trace::is_otel_enabled;

/// Module path used by CustomOtelFilter to identify events for OTEL export.
#[inline]
pub const fn get_module_path() -> &'static str {
    "smg::observability::events"
}

pub trait Event {
    fn emit(&self);
}

/// Event emitted when a prefill-decode request pair is sent.
#[derive(Debug, Clone, Copy)]
pub struct RequestPDSentEvent<'a> {
    pub prefill_url: &'a str,
    pub decode_url: &'a str,
}

impl Event for RequestPDSentEvent<'_> {
    #[inline]
    fn emit(&self) {
        if is_otel_enabled() {
            event!(
                Level::INFO,
                prefill_url = %self.prefill_url,
                decode_url = %self.decode_url,
                "Sending concurrent requests"
            );
        } else {
            debug!(
                prefill_url = %self.prefill_url,
                decode_url = %self.decode_url,
                "Sending concurrent requests"
            );
        }
    }
}

/// Event emitted when a request is sent to a worker.
#[derive(Debug, Clone, Copy)]
pub struct RequestSentEvent<'a> {
    pub url: &'a str,
}

impl Event for RequestSentEvent<'_> {
    #[inline]
    fn emit(&self) {
        if is_otel_enabled() {
            event!(Level::INFO, url = %self.url, "Sending request");
        } else {
            debug!(url = %self.url, "Sending request");
        }
    }
}

/// Event emitted when concurrent requests are received.
#[derive(Debug, Clone, Copy)]
pub struct RequestReceivedEvent;

impl Event for RequestReceivedEvent {
    #[inline]
    fn emit(&self) {
        if is_otel_enabled() {
            event!(Level::INFO, "Received concurrent requests");
        } else {
            debug!("Received concurrent requests");
        }
    }
}

/// Normalized request-level stats collected from engine-specific responses.
#[derive(Debug, Clone)]
pub struct UnifiedRequestStats {
    pub engine: &'static str,
    pub error_message: Option<String>,
    pub request_received_timestamp_s: Option<f64>,
    pub first_token_generated_timestamp_s: Option<f64>,
    pub request_finished_timestamp_s: Option<f64>,
    pub response_sent_timestamp_s: Option<f64>,
    pub cache_hit_rate: Option<f64>,
    pub spec_decoding_acceptance_rate: Option<f64>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
}

impl UnifiedRequestStats {
    /// Construct and emit a [`RequestStatsEvent`] from these stats.
    pub fn emit_event(
        &self,
        request_id: &str,
        model: &str,
        router_backend: &str,
        http_status_code: Option<u16>,
        error_message: Option<&str>,
    ) {
        RequestStatsEvent {
            request_id,
            model,
            router_backend,
            http_status_code,
            error_message,
            stats: self,
        }
        .emit();
    }
}

/// Unified request-stats event emitted once per backend request.
#[derive(Debug, Clone)]
struct RequestStatsEvent<'a> {
    request_id: &'a str,
    model: &'a str,
    router_backend: &'a str,
    http_status_code: Option<u16>,
    error_message: Option<&'a str>,
    stats: &'a UnifiedRequestStats,
}

macro_rules! emit_request_stats {
    ($log_macro:ident, $event:expr, $($prefix:tt)*) => {{
        let error_message = $event.error_message.or($event.stats.error_message.as_deref());
        $log_macro!(
            $($prefix)*
            request_id = %($event.request_id),
            model = %($event.model),
            router_backend = %($event.router_backend),
            http_status_code = $event.http_status_code,
            error_message = error_message,
            engine = %($event.stats.engine),
            request_received_timestamp_s = $event.stats.request_received_timestamp_s,
            first_token_generated_timestamp_s = $event.stats.first_token_generated_timestamp_s,
            request_finished_timestamp_s = $event.stats.request_finished_timestamp_s,
            response_sent_timestamp_s = $event.stats.response_sent_timestamp_s,
            cache_hit_rate = $event.stats.cache_hit_rate,
            spec_decoding_acceptance_rate = $event.stats.spec_decoding_acceptance_rate,
            prompt_tokens = $event.stats.prompt_tokens,
            completion_tokens = $event.stats.completion_tokens,
            cached_tokens = $event.stats.cached_tokens,
            "request_stats"
        );
    }};
}

impl Event for RequestStatsEvent<'_> {
    #[inline]
    fn emit(&self) {
        if is_otel_enabled() {
            emit_request_stats!(event, self, Level::INFO,);
        } else {
            emit_request_stats!(debug, self,);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::*;

    #[test]
    fn test_event_sizes() {
        assert_eq!(size_of::<RequestReceivedEvent>(), 0);
        assert_eq!(size_of::<RequestSentEvent>(), 16);
        assert_eq!(size_of::<RequestPDSentEvent>(), 32);
    }
}
