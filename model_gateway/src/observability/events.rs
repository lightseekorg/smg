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
#[derive(Debug, Clone, Default)]
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
    /// Emit a [`RequestStatsEvent`] if stats are present, otherwise no-op.
    pub fn maybe_emit_event(
        stats: Option<Self>,
        request_id: &str,
        model: &str,
        router_backend: &str,
        http_status_code: Option<u16>,
        error_message: Option<&str>,
    ) {
        if let Some(ref s) = stats {
            RequestStatsEvent {
                request_id,
                model,
                router_backend,
                http_status_code,
                error_message,
                stats: s,
            }
            .emit();
        }
    }

    /// Emit a success [`RequestStatsEvent`] if stats are present.
    pub fn emit_for_success(
        stats: Option<Self>,
        request_id: &str,
        model: &str,
        router_backend: &str,
    ) {
        Self::maybe_emit_event(stats, request_id, model, router_backend, Some(200), None);
    }

    /// Emit an error [`RequestStatsEvent`] if stats are present, and return the
    /// error message as a `String`. Designed for use in `map_err` closures.
    pub fn emit_for_error(
        stats: Option<Self>,
        request_id: &str,
        model: &str,
        router_backend: &str,
        http_status_code: Option<u16>,
        error_message: &str,
    ) -> String {
        Self::maybe_emit_event(
            stats,
            request_id,
            model,
            router_backend,
            http_status_code,
            Some(error_message),
        );
        error_message.to_string()
    }

    /// Merge another stats sample into this one (for multi-sample aggregation).
    /// Uses min for start timestamps, max for end timestamps, sum for
    /// completion_tokens, and first seen value for the remaining fields.
    /// Skip merging if engines differ.
    pub(crate) fn merge(&mut self, other: &Self) {
        if self.engine != other.engine {
            return;
        }
        self.request_received_timestamp_s = opt_min(
            self.request_received_timestamp_s,
            other.request_received_timestamp_s,
        );
        self.first_token_generated_timestamp_s = opt_min(
            self.first_token_generated_timestamp_s,
            other.first_token_generated_timestamp_s,
        );

        self.request_finished_timestamp_s = opt_max(
            self.request_finished_timestamp_s,
            other.request_finished_timestamp_s,
        );
        self.response_sent_timestamp_s = opt_max(
            self.response_sent_timestamp_s,
            other.response_sent_timestamp_s,
        );

        if let Some(pt) = other.prompt_tokens {
            self.prompt_tokens.get_or_insert(pt);
        }
        if let Some(ct) = other.completion_tokens {
            self.completion_tokens = Some(self.completion_tokens.unwrap_or(0) + ct);
        }
        if let Some(ct) = other.cached_tokens {
            self.cached_tokens.get_or_insert(ct);
        }
        if let Some(rate) = other.cache_hit_rate {
            self.cache_hit_rate.get_or_insert(rate);
        }
        if let Some(rate) = other.spec_decoding_acceptance_rate {
            self.spec_decoding_acceptance_rate.get_or_insert(rate);
        }
    }
}

fn opt_min(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

fn opt_max(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
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
    use std::{
        mem::size_of,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };

    use tracing_subscriber::layer::SubscriberExt;

    use super::*;
    use crate::observability::metrics::metrics_labels;

    #[test]
    fn test_event_sizes() {
        assert_eq!(size_of::<RequestReceivedEvent>(), 0);
        assert_eq!(size_of::<RequestSentEvent>(), 16);
        assert_eq!(size_of::<RequestPDSentEvent>(), 32);
    }

    /// Build a dummy request stats sample for tests.
    #[expect(clippy::too_many_arguments)]
    fn dummy_request_stats(
        received: f64,
        first_token: f64,
        finished: f64,
        sent: f64,
        prompt: u64,
        completion: u64,
        cached: u64,
        hit_rate: f64,
        spec_rate: f64,
    ) -> UnifiedRequestStats {
        UnifiedRequestStats {
            engine: "sglang",
            request_received_timestamp_s: Some(received),
            first_token_generated_timestamp_s: Some(first_token),
            request_finished_timestamp_s: Some(finished),
            response_sent_timestamp_s: Some(sent),
            prompt_tokens: Some(prompt),
            completion_tokens: Some(completion),
            cached_tokens: Some(cached),
            cache_hit_rate: Some(hit_rate),
            spec_decoding_acceptance_rate: Some(spec_rate),
            ..Default::default()
        }
    }

    #[test]
    fn test_merge_different_engines_is_noop() {
        let mut a = UnifiedRequestStats {
            engine: "sglang",
            request_received_timestamp_s: Some(1.0),
            prompt_tokens: Some(10),
            completion_tokens: Some(20),
            ..Default::default()
        };
        let original = a.clone();
        let b = UnifiedRequestStats {
            engine: "vllm",
            request_received_timestamp_s: Some(0.5),
            prompt_tokens: Some(5),
            completion_tokens: Some(30),
            ..Default::default()
        };
        a.merge(&b);
        assert_eq!(
            a.request_received_timestamp_s,
            original.request_received_timestamp_s
        );
        assert_eq!(a.prompt_tokens, original.prompt_tokens);
        assert_eq!(a.completion_tokens, original.completion_tokens);
    }

    #[test]
    fn test_merge_both_fully_populated() {
        let mut a = dummy_request_stats(2.0, 3.0, 8.0, 9.0, 100, 50, 10, 0.8, 0.9);
        let b = dummy_request_stats(1.0, 4.0, 10.0, 7.0, 200, 30, 20, 0.6, 0.5);
        a.merge(&b);

        assert_eq!(a.request_received_timestamp_s, Some(1.0));
        assert_eq!(a.first_token_generated_timestamp_s, Some(3.0));
        assert_eq!(a.request_finished_timestamp_s, Some(10.0));
        assert_eq!(a.response_sent_timestamp_s, Some(9.0));
        assert_eq!(a.completion_tokens, Some(80));
        assert_eq!(a.prompt_tokens, Some(100));
        assert_eq!(a.cached_tokens, Some(10));
        assert_eq!(a.cache_hit_rate, Some(0.8));
        assert_eq!(a.spec_decoding_acceptance_rate, Some(0.9));
    }

    #[test]
    fn test_merge_with_default_is_identity() {
        let populated = dummy_request_stats(1.0, 2.0, 5.0, 6.0, 10, 20, 5, 0.5, 0.7);
        let empty = UnifiedRequestStats {
            engine: "sglang",
            ..Default::default()
        };

        let mut a = populated.clone();
        a.merge(&empty);
        assert_eq!(a.request_received_timestamp_s, Some(1.0));
        assert_eq!(a.completion_tokens, Some(20));
        assert_eq!(a.cache_hit_rate, Some(0.5));

        let mut b = empty;
        b.merge(&populated);
        assert_eq!(b.request_received_timestamp_s, Some(1.0));
        assert_eq!(b.completion_tokens, Some(20));
        assert_eq!(b.cache_hit_rate, Some(0.5));
    }

    #[test]
    fn test_merge_three_samples_sequential() {
        let mut a = dummy_request_stats(3.0, 4.0, 7.0, 8.0, 100, 10, 5, 0.8, 0.9);
        let b = dummy_request_stats(1.0, 5.0, 9.0, 6.0, 200, 20, 15, 0.6, 0.5);
        let c = dummy_request_stats(2.0, 3.5, 8.0, 10.0, 300, 30, 25, 0.4, 0.3);
        a.merge(&b);
        a.merge(&c);

        assert_eq!(a.request_received_timestamp_s, Some(1.0));
        assert_eq!(a.first_token_generated_timestamp_s, Some(3.5));
        assert_eq!(a.request_finished_timestamp_s, Some(9.0));
        assert_eq!(a.response_sent_timestamp_s, Some(10.0));
        assert_eq!(a.completion_tokens, Some(60));
        assert_eq!(a.prompt_tokens, Some(100));
        assert_eq!(a.cached_tokens, Some(5));
        assert_eq!(a.cache_hit_rate, Some(0.8));
        assert_eq!(a.spec_decoding_acceptance_rate, Some(0.9));
    }

    /// Helper to count the number of events emitted.
    struct EventCounter(Arc<AtomicUsize>);

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for EventCounter {
        fn on_event(
            &self,
            _event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn count_events(f: impl FnOnce()) -> usize {
        let count = Arc::new(AtomicUsize::new(0));
        let subscriber = tracing_subscriber::registry().with(EventCounter(count.clone()));
        tracing::subscriber::with_default(subscriber, f);
        count.load(Ordering::Relaxed)
    }

    #[test]
    fn test_maybe_emit_event_emits_nothing() {
        let n = count_events(|| {
            UnifiedRequestStats::maybe_emit_event(
                None,
                "req-id",
                "test-model",
                metrics_labels::BACKEND_REGULAR,
                Some(200),
                None,
            );
        });
        assert_eq!(n, 0);
    }

    #[test]
    fn test_maybe_emit_event_emits_one_event() {
        let stats = UnifiedRequestStats {
            engine: "sglang",
            prompt_tokens: Some(10),
            completion_tokens: Some(20),
            error_message: Some("engine error".into()),
            ..Default::default()
        };
        let n = count_events(|| {
            UnifiedRequestStats::maybe_emit_event(
                Some(stats),
                "req-id",
                "test-model",
                metrics_labels::BACKEND_REGULAR,
                Some(500),
                Some("caller error"),
            );
        });
        assert_eq!(n, 1);
    }

    #[test]
    fn test_maybe_emit_event_multiple() {
        let n = count_events(|| {
            for i in 0..5 {
                let stats = UnifiedRequestStats {
                    engine: "sglang",
                    completion_tokens: Some(i),
                    ..Default::default()
                };
                UnifiedRequestStats::maybe_emit_event(
                    Some(stats),
                    &format!("req-id-{i}"),
                    "test-model",
                    metrics_labels::BACKEND_REGULAR,
                    Some(200),
                    None,
                );
            }
        });
        assert_eq!(n, 5);
    }
}
