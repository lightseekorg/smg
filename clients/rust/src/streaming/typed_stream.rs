use std::{
    marker::PhantomData,
    pin::Pin,
    task::{Context, Poll},
};

use futures::stream::Stream;
use serde::de::DeserializeOwned;

use crate::{streaming::SseEvent, SmgError};

/// A typed stream that deserializes SSE events into a concrete type `T`.
///
/// Implements `Stream<Item = Result<T, SmgError>>` so you can use it with
/// `StreamExt::next()` or `while let Some(chunk) = stream.next().await`.
pub struct TypedStream<T> {
    inner: Pin<Box<dyn Stream<Item = Result<SseEvent, SmgError>> + Send>>,
    _phantom: PhantomData<T>,
}

impl<T: DeserializeOwned> TypedStream<T> {
    pub(crate) fn new(
        inner: impl Stream<Item = Result<SseEvent, SmgError>> + Send + 'static,
    ) -> Self {
        Self {
            inner: Box::pin(inner),
            _phantom: PhantomData,
        }
    }
}

// INVARIANT: `TypedStream` is `Unpin` because `Pin<Box<...>>` is always `Unpin`.
impl<T> Unpin for TypedStream<T> {}

impl<T: DeserializeOwned> Stream for TypedStream<T> {
    type Item = Result<T, SmgError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(event))) => {
                let parsed = serde_json::from_str::<T>(&event.data).map_err(SmgError::from);
                Poll::Ready(Some(parsed))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}
