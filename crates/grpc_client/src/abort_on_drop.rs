//! Generic abort-on-drop wrapper for engine streaming responses.
//!
//! Each engine's `generate()` returns a server-streaming RPC; if the caller
//! drops the stream early (client disconnect, error, panic), the backend
//! has no way to know it should stop scheduling work. The wrapper here
//! sends an explicit `abort_request` from `Drop` so resources are
//! reclaimed even when the request never completes normally.
//!
//! Engines plug in via the `AbortOnDropClient` trait — they describe how
//! to translate `(client, request_id)` into the abort future. This keeps
//! the (large) Drop / Stream impl in one place instead of replicated
//! across `mlx_engine`, `sglang_scheduler`, `vllm_engine`, and
//! `trtllm_service`.

use std::{
    future::Future,
    marker::PhantomData,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Context, Poll},
};

use tonic::Streaming;
use tracing::{debug, warn};

/// Bridge between the generic [`AbortOnDropStream`] and an engine-specific
/// client. Implementors provide an async function that the wrapper calls
/// from `Drop` to release backend resources.
pub trait AbortOnDropClient: Clone + Send + Sync + 'static {
    /// Returned future is awaited by the spawned cleanup task — engines
    /// are free to attach the appropriate `reason` string (or omit it,
    /// for proto schemas that don't carry one).
    fn abort_for_drop(
        self,
        request_id: String,
    ) -> Pin<Box<dyn Future<Output = Result<(), tonic::Status>> + Send>>;
}

/// Smart wrapper around `tonic::Streaming<T>` that fires an abort RPC on
/// `Drop` unless [`mark_completed`](Self::mark_completed) was called.
///
/// `T` is the engine-specific stream item (typically `proto::GenerateResponse`).
/// `C` is the engine client implementing [`AbortOnDropClient`].
pub struct AbortOnDropStream<T, C: AbortOnDropClient> {
    inner: Streaming<T>,
    request_id: String,
    client: C,
    aborted: Arc<AtomicBool>,
    _marker: PhantomData<fn() -> T>,
}

impl<T, C: AbortOnDropClient> AbortOnDropStream<T, C> {
    /// Wrap a streaming response so it auto-aborts on drop.
    pub fn new(stream: Streaming<T>, request_id: String, client: C) -> Self {
        debug!("Created AbortOnDropStream for request {}", request_id);
        Self {
            inner: stream,
            request_id,
            client,
            aborted: Arc::new(AtomicBool::new(false)),
            _marker: PhantomData,
        }
    }

    /// Suppress the abort-on-drop. Call after the stream completes
    /// successfully so the backend isn't told to abort an already-finished
    /// request.
    pub fn mark_completed(&self) {
        // Release pairs with AcqRel in `Drop::drop` so the cleanup task
        // observes this write.
        self.aborted.store(true, Ordering::Release);
        debug!("Request {} marked as completed", self.request_id);
    }
}

impl<T, C: AbortOnDropClient> Drop for AbortOnDropStream<T, C> {
    fn drop(&mut self) {
        // Atomically claim the "send abort" responsibility. If
        // `mark_completed` already ran, `compare_exchange` fails and we
        // bail out; otherwise we own the cleanup.
        if self
            .aborted
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        let request_id = self.request_id.clone();
        let request_id_for_log = request_id.clone();
        let client = self.client.clone();

        #[expect(
            clippy::disallowed_methods,
            reason = "fire-and-forget abort on Drop is intentional"
        )]
        tokio::spawn(async move {
            debug!(
                "Stream dropped without completion for request {}, sending abort",
                request_id_for_log
            );
            if let Err(e) = client.abort_for_drop(request_id).await {
                warn!(
                    "Failed to send abort on drop for request {}: {}",
                    request_id_for_log, e
                );
            }
        });
    }
}

// `Streaming<T>` is `Unpin` regardless of `T`, and we never project a
// pinned reference to any field. Marking the wrapper `Unpin` lets us
// use `Pin<&mut Self>::deref_mut` to reach `inner` without needing
// `pin-project` machinery.
impl<T, C: AbortOnDropClient> Unpin for AbortOnDropStream<T, C> {}

impl<T, C: AbortOnDropClient> futures::Stream for AbortOnDropStream<T, C> {
    type Item = Result<T, tonic::Status>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-only test: any `Clone + Send + Sync + 'static` type can
    /// implement [`AbortOnDropClient`] with a single async method.
    /// `Drop` and `Stream` semantics are exercised via the engine-level
    /// integration tests that hit a real gRPC server.
    #[test]
    fn trait_is_implementable_by_simple_client() {
        #[derive(Clone)]
        struct DummyClient;

        impl AbortOnDropClient for DummyClient {
            fn abort_for_drop(
                self,
                _request_id: String,
            ) -> Pin<Box<dyn Future<Output = Result<(), tonic::Status>> + Send>> {
                Box::pin(async { Ok(()) })
            }
        }

        // `_marker` is `PhantomData<fn() -> T>`, so the struct itself is
        // `Send + Sync` regardless of `T`.
        fn assert_send_sync<X: Send + Sync>() {}
        assert_send_sync::<AbortOnDropStream<(), DummyClient>>();
    }
}
