//! Generic extension trait and registry for SMG request lifecycle hooks.
//!
//! See `docs/superpowers/0001-pr1-implementation-spec.md` for the design.

pub mod context;
pub mod interceptor;
pub mod metadata;
pub mod noop;
pub mod registry;

pub use context::{AfterPersistCtx, BeforeModelCtx};
pub use interceptor::ResponsesInterceptor;
pub use metadata::{ConversationTurnInfo, RequestMetadata};
pub use noop::NoOpInterceptor;
pub use registry::{InterceptorRegistry, InterceptorRegistryBuilder};
