//! Rate limiting and concurrency control for MCP tool execution.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{McpError, McpResult, QualifiedToolName, TenantContext, TenantId};

/// Configuration for rate limits.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RateLimits {
    pub max_calls_per_minute: Option<usize>,
    pub max_calls_per_hour: Option<usize>,
    pub max_concurrent: Option<usize>,
}

impl Default for RateLimits {
    fn default() -> Self {
        Self {
            max_calls_per_minute: Some(60),
            max_calls_per_hour: Some(1000),
            max_concurrent: Some(10),
        }
    }
}

/// A sliding window of tool call timestamps.
#[derive(Debug, Default)]
struct CallWindow {
    minute_calls: Vec<Instant>,
    hour_calls: Vec<Instant>,
}

impl CallWindow {
    /// Records a new call at the current timestamp.
    fn record(&mut self) {
        let now = Instant::now();
        self.minute_calls.push(now);
        self.hour_calls.push(now);
        self.cleanup();
    }

    /// Removes expired timestamps from the windows.
    fn cleanup(&mut self) {
        let now = Instant::now();
        let minute_ago = now - Duration::from_secs(60);
        let hour_ago = now - Duration::from_secs(3600);

        self.minute_calls.retain(|&t| t > minute_ago);
        self.hour_calls.retain(|&t| t > hour_ago);
    }

    fn calls_per_minute(&self) -> usize {
        self.minute_calls.len()
    }

    fn calls_per_hour(&self) -> usize {
        self.hour_calls.len()
    }
}

/// Manages rate limits and concurrency across tenants and tools.
pub struct RateLimiter {
    tenant_calls: DashMap<TenantId, CallWindow>,
    tool_calls: DashMap<QualifiedToolName, CallWindow>,
    tenant_semaphores: DashMap<TenantId, Arc<Semaphore>>,
    defaults: RateLimits,
}

impl RateLimiter {
    pub fn new(defaults: RateLimits) -> Self {
        Self {
            tenant_calls: DashMap::new(),
            tool_calls: DashMap::new(),
            tenant_semaphores: DashMap::new(),
            defaults,
        }
    }

    /// Checks if a call is allowed under current rate limits.
    /// Priority: Tenant-specific limits > Default limits.
    pub fn check(&self, ctx: &TenantContext, tool: &QualifiedToolName) -> McpResult<()> {
        let limits = ctx.limits.unwrap_or(self.defaults);

        // Check Tenant-wide limits
        if let Some(window) = self.tenant_calls.get(&ctx.tenant_id) {
            if let Some(max) = limits.max_calls_per_minute {
                if window.calls_per_minute() >= max {
                    return Err(McpError::RateLimitExceeded(format!(
                        "Tenant '{}' minute limit reached ({})",
                        ctx.tenant_id, max
                    )));
                }
            }
            if let Some(max) = limits.max_calls_per_hour {
                if window.calls_per_hour() >= max {
                    return Err(McpError::RateLimitExceeded(format!(
                        "Tenant '{}' hour limit reached ({})",
                        ctx.tenant_id, max
                    )));
                }
            }
        }

        // Check Tool-specific limits
        if let Some(window) = self.tool_calls.get(tool) {
            if let Some(max) = limits.max_calls_per_minute {
                if window.calls_per_minute() >= max {
                    return Err(McpError::RateLimitExceeded(format!(
                        "Tool '{}' minute limit reached ({})",
                        tool, max
                    )));
                }
            }
        }

        Ok(())
    }

    /// Records a successful tool call initiation.
    pub fn record(&self, tenant_id: &TenantId, tool: &QualifiedToolName) {
        self.tenant_calls
            .entry(tenant_id.clone())
            .or_default()
            .record();
        self.tool_calls.entry(tool.clone()).or_default().record();
    }

    /// Acquires a permit for concurrent execution.
    pub async fn acquire_concurrent_permit(
        &self,
        ctx: &TenantContext,
    ) -> McpResult<OwnedSemaphorePermit> {
        let limits = ctx.limits.unwrap_or(self.defaults);
        let max_concurrent = limits.max_concurrent.unwrap_or(10);

        let semaphore = self
            .tenant_semaphores
            .entry(ctx.tenant_id.clone())
            .or_insert_with(|| Arc::new(Semaphore::new(max_concurrent)))
            .clone();

        semaphore
            .acquire_owned()
            .await
            .map_err(|_| McpError::RateLimitExceeded("Concurrency semaphore closed".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sliding_window() {
        let mut window = CallWindow::default();
        window.record();
        assert_eq!(window.calls_per_minute(), 1);
    }

    #[tokio::test]
    async fn test_rate_limiter_check() {
        let limits = RateLimits {
            max_calls_per_minute: Some(1),
            ..Default::default()
        };
        let limiter = RateLimiter::new(limits);
        let ctx = TenantContext::new("t1");
        let tool = QualifiedToolName::new("s1", "t1");

        assert!(limiter.check(&ctx, &tool).is_ok());
        limiter.record(&ctx.tenant_id, &tool);
        assert!(limiter.check(&ctx, &tool).is_err());
    }
}
