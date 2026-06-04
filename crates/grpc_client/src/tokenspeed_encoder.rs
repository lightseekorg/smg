//! gRPC client for the TokenSpeed EPD encode service.
//!
//! The gateway calls `encode()` to hand a vision-tower-only worker the
//! preprocessed multimodal tensors for one request; the worker runs the tower
//! and ships the embedding to the paired prefill worker over Mooncake (keyed by
//! `bootstrap_room`). The response only confirms the request was accepted — the
//! embedding transfer happens out of band.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use tonic::{transport::Channel, Request};
use tracing::{debug, warn};

use crate::{BoxedTraceInjector, NoopTraceInjector};

/// Process-global cache of connected channels, keyed by encode-worker endpoint.
///
/// The EPD encode stage dispatches one `Encode` RPC per image per request, so
/// without pooling every call paid a fresh TCP + HTTP/2 (+ TLS) handshake on the
/// request's critical path. A `tonic::Channel` is cheap to clone and multiplexes
/// concurrent RPCs over a single HTTP/2 connection, so we reuse one channel per
/// endpoint — the same connection reuse the prefill/decode legs already get from
/// their cached per-worker client.
fn channel_cache() -> &'static Mutex<HashMap<String, Channel>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Channel>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[expect(clippy::allow_attributes)]
pub mod tokenspeed_encoder_proto {
    #![allow(clippy::all, clippy::absolute_paths, unused_qualifications)]
    tonic::include_proto!("tokenspeed.grpc.encoder");
}

/// gRPC client for the TokenSpeed encode worker.
#[derive(Clone)]
pub struct TokenSpeedEncoderClient {
    client: tokenspeed_encoder_proto::token_speed_encoder_client::TokenSpeedEncoderClient<Channel>,
    trace_injector: BoxedTraceInjector,
}

impl TokenSpeedEncoderClient {
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::connect_with_trace_injector(endpoint, Arc::new(NoopTraceInjector)).await
    }

    /// Connect reusing a cached `Channel` for `endpoint` (see [`channel_cache`]).
    /// Use this on the per-request hot path; [`Self::connect`] always opens a
    /// fresh connection. The lock is never held across the connect `await`, so a
    /// rare concurrent first-connect may build two channels for the same endpoint
    /// — harmless: the first cached wins and the extra is dropped.
    pub async fn connect_cached(
        endpoint: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        {
            let cache = channel_cache()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(channel) = cache.get(endpoint) {
                return Ok(Self::from_channel(channel.clone()));
            }
        }
        debug!("Connecting to TokenSpeed encoder at {} (caching)", endpoint);
        let channel = crate::channel::connect_channel(endpoint).await?;
        {
            let mut cache = channel_cache()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            cache
                .entry(endpoint.to_string())
                .or_insert_with(|| channel.clone());
        }
        Ok(Self::from_channel(channel))
    }

    fn from_channel(channel: Channel) -> Self {
        Self {
            client:
                tokenspeed_encoder_proto::token_speed_encoder_client::TokenSpeedEncoderClient::new(
                    channel,
                ),
            trace_injector: Arc::new(NoopTraceInjector),
        }
    }

    pub async fn connect_with_trace_injector(
        endpoint: &str,
        trace_injector: BoxedTraceInjector,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Connecting to TokenSpeed encoder at {}", endpoint);
        let channel = crate::channel::connect_channel(endpoint).await?;
        let client =
            tokenspeed_encoder_proto::token_speed_encoder_client::TokenSpeedEncoderClient::new(
                channel,
            );

        Ok(Self {
            client,
            trace_injector,
        })
    }

    #[must_use]
    pub fn with_trace_injector(mut self, trace_injector: BoxedTraceInjector) -> Self {
        self.trace_injector = trace_injector;
        self
    }

    /// Trigger the vision tower on a request's multimodal inputs. Returns once
    /// the worker has accepted (enqueued) the request; the embedding then ships
    /// to the prefill peer asynchronously over Mooncake.
    pub async fn encode(
        &self,
        req: tokenspeed_encoder_proto::EncodeRequest,
    ) -> Result<tokenspeed_encoder_proto::EncodeResponse, tonic::Status> {
        let mut client = self.client.clone();
        let mut request = Request::new(req);

        if let Err(e) = self.trace_injector.inject(request.metadata_mut()) {
            warn!("Failed to inject trace context: {}", e);
        }

        let response = client.encode(request).await?;
        Ok(response.into_inner())
    }
}
