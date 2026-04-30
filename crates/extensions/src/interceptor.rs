use async_trait::async_trait;

use crate::context::{AfterPersistCtx, BeforeModelCtx};

/// Lifecycle hook trait for the Responses request pipeline.
///
/// Implementors are registered into an [`InterceptorRegistry`](crate::registry::InterceptorRegistry)
/// and invoked at two phases:
///
/// 1. `before_model` — after history is loaded, before the request is forwarded
///    to the model. Implementors may mutate `ctx.request` to inject context.
/// 2. `after_persist` — after persistence succeeds. Implementors typically
///    enqueue async work (memory consolidation, audit logging, etc.).
///
/// Errors are non-fatal: trait methods return `()`, and the registry catches
/// panics. A buggy interceptor cannot fail an SMG request.
///
/// ## Recommended destructure pattern
///
/// Both context types are `#[non_exhaustive]`. New fields may be added in
/// future versions. Always destructure with `..`:
///
/// ```ignore
/// async fn before_model(&self, ctx: &mut BeforeModelCtx<'_>) {
///     let BeforeModelCtx { headers, request, conversation_id, .. } = ctx;
///     // ...
/// }
/// ```
#[async_trait]
pub trait ResponsesInterceptor: Send + Sync + 'static {
    /// Stable identifier for diagnostics, metrics, and panic logging.
    fn name(&self) -> &'static str;

    /// Pre-model phase. Default: no-op.
    async fn before_model(&self, ctx: &mut BeforeModelCtx<'_>) {
        let _ = ctx;
    }

    /// Post-persist phase. Default: no-op.
    async fn after_persist(&self, ctx: &AfterPersistCtx<'_>) {
        let _ = ctx;
    }
}
