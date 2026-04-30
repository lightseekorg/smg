use std::{panic::AssertUnwindSafe, sync::Arc};

use futures::FutureExt;
use tracing::warn;

use crate::context::{AfterPersistCtx, BeforeModelCtx};
use crate::interceptor::ResponsesInterceptor;

#[derive(Default, Clone)]
pub struct InterceptorRegistry {
    interceptors: Arc<Vec<Arc<dyn ResponsesInterceptor>>>,
}

impl InterceptorRegistry {
    pub fn builder() -> InterceptorRegistryBuilder {
        InterceptorRegistryBuilder::new()
    }

    pub fn is_empty(&self) -> bool {
        self.interceptors.is_empty()
    }

    pub fn len(&self) -> usize {
        self.interceptors.len()
    }

    pub async fn run_before_model(&self, ctx: &mut BeforeModelCtx<'_>) {
        for i in self.interceptors.iter() {
            let result = AssertUnwindSafe(i.before_model(ctx)).catch_unwind().await;
            if result.is_err() {
                warn!(
                    interceptor = i.name(),
                    "before_model panicked; continuing"
                );
            }
        }
    }

    pub async fn run_after_persist(&self, ctx: &AfterPersistCtx<'_>) {
        for i in self.interceptors.iter() {
            let result = AssertUnwindSafe(i.after_persist(ctx)).catch_unwind().await;
            if result.is_err() {
                warn!(
                    interceptor = i.name(),
                    "after_persist panicked; continuing"
                );
            }
        }
    }
}

pub struct InterceptorRegistryBuilder {
    interceptors: Vec<Arc<dyn ResponsesInterceptor>>,
}

impl Default for InterceptorRegistryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl InterceptorRegistryBuilder {
    pub fn new() -> Self {
        Self {
            interceptors: Vec::new(),
        }
    }

    pub fn register(&mut self, i: Arc<dyn ResponsesInterceptor>) -> &mut Self {
        self.interceptors.push(i);
        self
    }

    pub fn build(self) -> InterceptorRegistry {
        InterceptorRegistry {
            interceptors: Arc::new(self.interceptors),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use async_trait::async_trait;
    use chrono::Utc;
    use smg_data_connector::ConversationItemStorage;

    use super::*;
    use crate::metadata::{ConversationTurnInfo, RequestMetadata};

    fn make_metadata() -> RequestMetadata {
        RequestMetadata {
            request_id: "req_test".into(),
            safety_identifier: None,
            tenant_id: None,
            originated_at: Utc::now(),
            storage_request_context: None,
        }
    }

    fn make_history() -> Arc<dyn ConversationItemStorage> {
        Arc::new(smg_data_connector::MemoryConversationItemStorage::new())
    }

    struct CountingInterceptor {
        before_count: Arc<AtomicUsize>,
        after_count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ResponsesInterceptor for CountingInterceptor {
        fn name(&self) -> &'static str {
            "counting"
        }
        async fn before_model(&self, _ctx: &mut BeforeModelCtx<'_>) {
            self.before_count.fetch_add(1, Ordering::SeqCst);
        }
        async fn after_persist(&self, _ctx: &AfterPersistCtx<'_>) {
            self.after_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct PanickyInterceptor;

    #[async_trait]
    impl ResponsesInterceptor for PanickyInterceptor {
        fn name(&self) -> &'static str {
            "panicky"
        }
        async fn before_model(&self, _ctx: &mut BeforeModelCtx<'_>) {
            panic!("intentional panic in before_model");
        }
        async fn after_persist(&self, _ctx: &AfterPersistCtx<'_>) {
            panic!("intentional panic in after_persist");
        }
    }

    #[tokio::test]
    async fn empty_registry_is_empty() {
        let registry = InterceptorRegistry::default();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[tokio::test]
    async fn registry_runs_both_phases_in_order() {
        let before_count = Arc::new(AtomicUsize::new(0));
        let after_count = Arc::new(AtomicUsize::new(0));

        let mut builder = InterceptorRegistry::builder();
        builder.register(Arc::new(CountingInterceptor {
            before_count: before_count.clone(),
            after_count: after_count.clone(),
        }));
        builder.register(Arc::new(CountingInterceptor {
            before_count: before_count.clone(),
            after_count: after_count.clone(),
        }));
        let registry = builder.build();

        let metadata = make_metadata();
        let history = make_history();
        let mut request = openai_protocol::responses::ResponsesRequest::default();
        let headers = axum::http::HeaderMap::new();

        let mut before = BeforeModelCtx {
            headers: &headers,
            request: &mut request,
            conversation_id: None,
            history: history.clone(),
            turn_info: ConversationTurnInfo::default(),
            request_metadata: &metadata,
        };
        registry.run_before_model(&mut before).await;

        let after = AfterPersistCtx {
            headers: &headers,
            request: &request,
            response_json: None,
            response_id: None,
            conversation_id: None,
            turn_info: ConversationTurnInfo::default(),
            persisted_item_ids: &[],
            request_metadata: &metadata,
        };
        registry.run_after_persist(&after).await;

        assert_eq!(before_count.load(Ordering::SeqCst), 2);
        assert_eq!(after_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn panic_in_before_model_does_not_propagate() {
        let mut builder = InterceptorRegistry::builder();
        builder.register(Arc::new(PanickyInterceptor));
        let registry = builder.build();

        let metadata = make_metadata();
        let history = make_history();
        let mut request = openai_protocol::responses::ResponsesRequest::default();
        let headers = axum::http::HeaderMap::new();
        let mut before = BeforeModelCtx {
            headers: &headers,
            request: &mut request,
            conversation_id: None,
            history,
            turn_info: ConversationTurnInfo::default(),
            request_metadata: &metadata,
        };

        registry.run_before_model(&mut before).await;
    }

    #[tokio::test]
    async fn panic_in_one_does_not_skip_next() {
        let after_count = Arc::new(AtomicUsize::new(0));
        let mut builder = InterceptorRegistry::builder();
        builder.register(Arc::new(PanickyInterceptor));
        builder.register(Arc::new(CountingInterceptor {
            before_count: Arc::new(AtomicUsize::new(0)),
            after_count: after_count.clone(),
        }));
        let registry = builder.build();

        let metadata = make_metadata();
        let request = openai_protocol::responses::ResponsesRequest::default();
        let headers = axum::http::HeaderMap::new();
        let after = AfterPersistCtx {
            headers: &headers,
            request: &request,
            response_json: None,
            response_id: None,
            conversation_id: None,
            turn_info: ConversationTurnInfo::default(),
            persisted_item_ids: &[],
            request_metadata: &metadata,
        };

        registry.run_after_persist(&after).await;
        assert_eq!(after_count.load(Ordering::SeqCst), 1);
    }
}
