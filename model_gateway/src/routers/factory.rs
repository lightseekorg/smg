//! Factory for creating router instances

use std::sync::Arc;

use super::{
    anthropic::AnthropicRouter,
    grpc::{pd_router::GrpcPDRouter, router::GrpcRouter},
    http::{pd_router::PDRouter, router::Router},
    openai::OpenAIRouter,
    RouterTrait,
};
use crate::{
    app_context::AppContext,
    config::{PolicyConfig, RoutingMode},
    core::ConnectionMode,
    policies::PolicyFactory,
};

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct RouterId(&'static str);

impl RouterId {
    pub const fn new(id: &'static str) -> Self {
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        self.0
    }
}

/// Static router ID constants to avoid heap allocations in hot paths
pub mod router_ids {
    use super::RouterId;

    pub const HTTP_REGULAR: RouterId = RouterId::new("http-regular");
    pub const HTTP_PD: RouterId = RouterId::new("http-pd");
    pub const HTTP_OPENAI: RouterId = RouterId::new("http-openai");
    pub const HTTP_ANTHROPIC: RouterId = RouterId::new("http-anthropic");
    pub const GRPC_REGULAR: RouterId = RouterId::new("grpc-regular");
    pub const GRPC_PD: RouterId = RouterId::new("grpc-pd");
}

/// Factory for creating router instances based on configuration
pub struct RouterFactory;

impl RouterFactory {
    /// Create a router instance from application context
    pub async fn create_router(ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
        match ctx.router_config.connection_mode {
            ConnectionMode::Grpc => match &ctx.router_config.mode {
                RoutingMode::Regular { .. } => Self::create_grpc_router(ctx).await,
                RoutingMode::PrefillDecode {
                    prefill_policy,
                    decode_policy,
                    ..
                } => {
                    Self::create_grpc_pd_router(
                        prefill_policy.as_ref(),
                        decode_policy.as_ref(),
                        &ctx.router_config.policy,
                        ctx,
                    )
                    .await
                }
                RoutingMode::OpenAI { .. } => {
                    Err("OpenAI mode requires HTTP connection_mode".to_string())
                }
                RoutingMode::Anthropic { .. } => {
                    Err("Anthropic mode requires HTTP connection_mode".to_string())
                }
            },
            ConnectionMode::Http => match &ctx.router_config.mode {
                RoutingMode::Regular { .. } => Self::create_regular_router(ctx).await,
                RoutingMode::PrefillDecode {
                    prefill_policy,
                    decode_policy,
                    ..
                } => {
                    Self::create_pd_router(
                        prefill_policy.as_ref(),
                        decode_policy.as_ref(),
                        &ctx.router_config.policy,
                        ctx,
                    )
                    .await
                }
                RoutingMode::OpenAI { .. } => Self::create_openai_router(ctx).await,
                RoutingMode::Anthropic { .. } => Self::create_anthropic_router(ctx).await,
            },
        }
    }

    /// Create a regular router
    pub async fn create_regular_router(
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        let router = Router::new(ctx).await?;

        Ok(Box::new(router))
    }

    /// Create a PD router with injected policy
    pub async fn create_pd_router(
        prefill_policy_config: Option<&PolicyConfig>,
        decode_policy_config: Option<&PolicyConfig>,
        main_policy_config: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        let prefill_policy =
            PolicyFactory::create_from_config(prefill_policy_config.unwrap_or(main_policy_config));
        let decode_policy =
            PolicyFactory::create_from_config(decode_policy_config.unwrap_or(main_policy_config));

        ctx.policy_registry.set_prefill_policy(prefill_policy);
        ctx.policy_registry.set_decode_policy(decode_policy);

        let router = PDRouter::new(ctx).await?;

        Ok(Box::new(router))
    }

    /// Create a gRPC router with injected policy
    pub async fn create_grpc_router(ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
        let router = GrpcRouter::new(ctx).await?;

        Ok(Box::new(router))
    }

    /// Create a gRPC PD router with tokenizer and worker configuration
    pub async fn create_grpc_pd_router(
        prefill_policy_config: Option<&PolicyConfig>,
        decode_policy_config: Option<&PolicyConfig>,
        main_policy_config: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        let prefill_policy =
            PolicyFactory::create_from_config(prefill_policy_config.unwrap_or(main_policy_config));
        let decode_policy =
            PolicyFactory::create_from_config(decode_policy_config.unwrap_or(main_policy_config));

        ctx.policy_registry.set_prefill_policy(prefill_policy);
        ctx.policy_registry.set_decode_policy(decode_policy);
        let router = GrpcPDRouter::new(ctx).await?;

        Ok(Box::new(router))
    }

    /// Create an OpenAI router
    ///
    /// Workers should be registered via the external worker registration workflow
    /// before using this router. The workflow discovers models from the provided
    /// endpoints and creates external workers in the registry.
    pub async fn create_openai_router(
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        let router = OpenAIRouter::new(ctx).await?;
        Ok(Box::new(router))
    }

    /// Create an Anthropic router
    ///
    /// Handles Anthropic Messages API (/v1/messages) with support for streaming,
    /// tool use, extended thinking, and other Anthropic-specific features.
    pub async fn create_anthropic_router(
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        let router = AnthropicRouter::new(ctx.clone())?;
        Ok(Box::new(router))
    }

    /// Create all routers for IGW (multi-router) mode.
    ///
    /// Returns a list of (router_id, label, creation_result) tuples.
    /// Adding a new router to IGW mode only requires adding a line here.
    pub async fn create_igw_routers(
        policy: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) -> Vec<(RouterId, &'static str, Result<Box<dyn RouterTrait>, String>)> {
        vec![
            (
                router_ids::HTTP_REGULAR,
                "HTTP Regular",
                Self::create_regular_router(ctx).await,
            ),
            (
                router_ids::GRPC_REGULAR,
                "gRPC Regular",
                Self::create_grpc_router(ctx).await,
            ),
            (
                router_ids::HTTP_PD,
                "HTTP PD",
                Self::create_pd_router(None, None, policy, ctx).await,
            ),
            (
                router_ids::GRPC_PD,
                "gRPC PD",
                Self::create_grpc_pd_router(None, None, policy, ctx).await,
            ),
            (
                router_ids::HTTP_OPENAI,
                "OpenAI",
                Self::create_openai_router(ctx).await,
            ),
            (
                router_ids::HTTP_ANTHROPIC,
                "Anthropic",
                Self::create_anthropic_router(ctx).await,
            ),
        ]
    }
}
