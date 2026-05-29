//! Response-body wrapper that ties a [`SchedulerPermit`] to the response
//! stream: it marks TTFT on the first data frame and releases the slot
//! when the body finishes draining (or is dropped).

use std::{
    pin::Pin,
    task::{Context, Poll},
};

use axum::body::Body;
use bytes::Bytes;
use http_body::Frame;

use super::engine::SchedulerPermit;

/// Wraps a response [`Body`], holding the request's [`SchedulerPermit`]
/// for the lifetime of the stream.
///
/// - **TTFT.** On the first *data* frame, marks time-to-first-byte via the
///   permit's CAS. If the scheduler already won the preemption CAS
///   (`try_mark_first_byte` returns `false`), the request was preempted in
///   the narrow gap before its first byte — we end the stream
///   (`Poll::Ready(None)`) rather than emit a byte the client should never
///   see. The handler's cancel token has fired, so it is unwinding anyway.
///   Only data frames mark TTFT; trailer-only responses never do.
/// - **Slot release.** When this body drops (stream complete, client
///   disconnected, or handler unwound), the `permit` field drops with it,
///   and `SchedulerPermit::drop` releases the slot back to the scheduler
///   and unregisters the inflight handle. No explicit `Drop` is needed —
///   the field's natural drop carries the invariant.
pub struct SchedulerGuardBody {
    inner: Body,
    permit: SchedulerPermit,
    /// Latch so the TTFT CAS is attempted only once (on the first data
    /// frame); subsequent frames skip the check entirely.
    ttft_marked: bool,
}

impl SchedulerGuardBody {
    pub fn new(inner: Body, permit: SchedulerPermit) -> Self {
        Self {
            inner,
            permit,
            ttft_marked: false,
        }
    }
}

impl http_body::Body for SchedulerGuardBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        // `Body` is `UnsyncBoxBody`, which is `Unpin`; the other fields are
        // `Unpin` too, so `get_mut` + `Pin::new(&mut inner)` is sound
        // (mirrors `TokenGuardBody` in `concurrency.rs`).
        let this = self.get_mut();
        let polled = Pin::new(&mut this.inner).poll_frame(cx);

        if !this.ttft_marked {
            if let Poll::Ready(Some(Ok(frame))) = &polled {
                if frame.is_data() {
                    if this.permit.try_mark_first_byte() {
                        this.ttft_marked = true;
                    } else {
                        // Lost the TTFT race to a concurrent preemption
                        // CAS. Terminate the stream cleanly.
                        return Poll::Ready(None);
                    }
                }
            }
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use http_body_util::BodyExt;
    use smg_auth::RequestId;

    use super::*;
    use crate::middleware::scheduler::{Class, PriorityScheduler, SchedulerSettings};

    fn scheduler() -> std::sync::Arc<PriorityScheduler> {
        let settings =
            SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, None).unwrap();
        // Default reservations sum to 160 (Interactive 128 + System 32),
        // so capacity must be >= 160 for new() to accept them.
        PriorityScheduler::new(&settings, 256).unwrap()
    }

    fn permit(sched: &std::sync::Arc<PriorityScheduler>, id: &str) -> SchedulerPermit {
        sched
            .acquire_inflight(Class::Default, RequestId(id.to_string()))
            .expect("slot available")
    }

    #[tokio::test]
    async fn test_first_data_frame_marks_ttft() {
        let sched = scheduler();
        let p = permit(&sched, "req-ttft");
        let handle = std::sync::Arc::clone(p.handle());
        let mut guarded = SchedulerGuardBody::new(Body::from("hello world"), p);

        assert!(handle.is_preemptible(), "pre-TTFT before first frame");
        // Drive the first frame (SchedulerGuardBody is Unpin).
        let frame = guarded.frame().await.expect("a frame").expect("ok frame");
        assert!(frame.is_data());
        assert!(
            !handle.is_preemptible(),
            "first data frame must mark TTFT (no longer preemptible)"
        );
        // A request that has emitted its first byte must reject preemption.
        assert!(!handle.try_mark_preempted());
    }

    #[tokio::test]
    async fn test_preempted_before_first_byte_ends_stream() {
        let sched = scheduler();
        let p = permit(&sched, "req-preempt");
        let handle = std::sync::Arc::clone(p.handle());
        // Scheduler wins the preempt CAS before any byte is polled.
        assert!(handle.try_mark_preempted());

        let mut guarded = SchedulerGuardBody::new(Body::from("should not surface"), p);
        // First poll must terminate the stream rather than yield the data.
        let next = guarded.frame().await;
        assert!(
            next.is_none(),
            "preempted body must end the stream on first frame"
        );
    }

    #[tokio::test]
    async fn test_trailer_only_response_never_marks_ttft() {
        let sched = scheduler();
        let p = permit(&sched, "req-empty");
        let handle = std::sync::Arc::clone(p.handle());
        // Empty body: no data frames at all.
        let guarded = SchedulerGuardBody::new(Body::empty(), p);
        let _ = guarded.collect().await; // exhaust
        assert!(
            handle.is_preemptible(),
            "no data frame emitted → TTFT never marked"
        );
    }

    #[tokio::test]
    async fn test_drop_releases_slot() {
        let sched = scheduler();
        assert_eq!(sched.inflight_for_test(Class::Default), 0);
        let p = permit(&sched, "req-drop");
        assert_eq!(sched.inflight_for_test(Class::Default), 1);
        let guarded = SchedulerGuardBody::new(Body::from("x"), p);
        drop(guarded);
        assert_eq!(
            sched.inflight_for_test(Class::Default),
            0,
            "dropping the guarded body releases the slot"
        );
    }
}
