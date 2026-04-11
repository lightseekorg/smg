use std::sync::Arc;

use axum::body::Body;

use super::traits::Worker;

/// RAII guard for worker load management
///
/// Automatically decrements worker load when dropped. Can be attached to
/// an axum Response to tie the guard's lifetime to the response body,
/// which is essential for streaming responses where the function returns
/// immediately but the stream continues in the background.
pub struct WorkerLoadGuard {
    worker: Arc<dyn Worker>,
    routing_key: Option<String>,
}

impl WorkerLoadGuard {
    pub fn new(worker: Arc<dyn Worker>, headers: Option<&http::HeaderMap>) -> Self {
        use crate::routers::header_utils::extract_routing_key;

        worker.increment_load();

        let routing_key = extract_routing_key(headers).map(String::from);

        if let Some(ref key) = routing_key {
            worker.worker_routing_key_load().increment(key);
        }

        Self {
            worker,
            routing_key,
        }
    }
}

impl Drop for WorkerLoadGuard {
    fn drop(&mut self) {
        self.worker.decrement_load();
        if let Some(ref key) = self.routing_key {
            self.worker.worker_routing_key_load().decrement(key);
        }
    }
}

/// Body wrapper that holds an attached value.
///
/// When this body is dropped (stream ends or client disconnects),
/// the attached value is dropped automatically. This is useful for RAII guards
/// like WorkerLoadGuard that need to be tied to a response body's lifetime.
pub struct AttachedBody<T> {
    inner: Body,
    _attached: T,
}

impl<T> AttachedBody<T> {
    pub fn new(inner: Body, attached: T) -> Self {
        Self {
            inner,
            _attached: attached,
        }
    }
}

impl<T: Send + Unpin + 'static> AttachedBody<T> {
    pub fn wrap_response(
        response: axum::response::Response,
        attached: T,
    ) -> axum::response::Response {
        let (parts, body) = response.into_parts();
        axum::response::Response::from_parts(parts, Body::new(Self::new(body, attached)))
    }
}

impl<T: Send + Unpin + 'static> http_body::Body for AttachedBody<T> {
    type Data = bytes::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        std::pin::Pin::new(&mut this.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}
