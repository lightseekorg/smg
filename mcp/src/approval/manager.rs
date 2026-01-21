//! Approval manager for MCP tool execution.
//!
//! Provides a dual-mode approval system:
//! - **Interactive mode**: Returns approval requests to the caller for user confirmation
//! - **Policy-only mode**: Auto-decides using the PolicyEngine

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use super::{
    audit::{AuditLog, DecisionResult, DecisionSource},
    policy::{PolicyDecision, PolicyEngine},
};
use crate::{
    annotations::ToolAnnotations,
    error::{ApprovalError, McpResult},
    inventory::QualifiedToolName,
    tenant::TenantContext,
};

/// Default timeout for pending approvals (5 minutes).
const DEFAULT_APPROVAL_TIMEOUT: Duration = Duration::from_secs(300);

/// Key for tracking pending approvals.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ApprovalKey {
    /// Inference request ID.
    pub request_id: String,
    /// MCP server key.
    pub server_key: String,
    /// MCP elicitation request ID.
    pub elicitation_id: String,
}

impl ApprovalKey {
    /// Create a new approval key.
    pub fn new(
        request_id: impl Into<String>,
        server_key: impl Into<String>,
        elicitation_id: impl Into<String>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            server_key: server_key.into(),
            elicitation_id: elicitation_id.into(),
        }
    }
}

impl std::fmt::Display for ApprovalKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}",
            self.request_id, self.server_key, self.elicitation_id
        )
    }
}

/// A pending approval awaiting user response.
#[derive(Debug)]
pub struct PendingApproval {
    pub key: ApprovalKey,
    pub tool_name: String,
    pub hints: ToolAnnotations,
    pub message: String,
    pub created_at: Instant,
    response_tx: oneshot::Sender<ApprovalDecision>,
}

/// User's decision on an approval request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalDecision {
    /// Approved for execution.
    Approved,
    /// Denied with optional reason.
    Denied { reason: String },
}

impl ApprovalDecision {
    /// Create an approved decision.
    pub fn approved() -> Self {
        Self::Approved
    }

    /// Create a denied decision.
    pub fn denied(reason: impl Into<String>) -> Self {
        Self::Denied {
            reason: reason.into(),
        }
    }

    /// Check if this decision allows execution.
    pub fn is_approved(&self) -> bool {
        matches!(self, ApprovalDecision::Approved)
    }
}

/// Approval mode determines how approvals are handled.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalMode {
    /// Interactive - return approval request, await user response.
    Interactive,
    /// Policy-only - auto-decide using PolicyEngine.
    #[default]
    PolicyOnly,
}

/// Parameters for requesting tool approval.
#[derive(Debug, Clone)]
pub struct ApprovalParams<'a> {
    pub request_id: &'a str,
    pub server_key: &'a str,
    pub elicitation_id: &'a str,
    pub tool_name: &'a str,
    pub hints: &'a ToolAnnotations,
    pub message: &'a str,
    pub tenant_ctx: &'a TenantContext,
}

/// Approval request to be sent to the client (matches OpenAI format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpApprovalRequest {
    /// MCP server key.
    pub server_key: String,
    /// Tool name.
    pub tool_name: String,
    /// Human-readable message explaining the request.
    pub message: String,
    /// Elicitation ID for correlation.
    pub elicitation_id: String,
}

/// Approval response from the client (matches OpenAI format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpApprovalResponse {
    /// Whether the user approved.
    pub approve: bool,
    /// Optional reason for denial.
    pub reason: Option<String>,
}

/// Result of handling an approval request.
#[derive(Debug)]
pub enum ApprovalOutcome {
    /// Immediate decision (policy-only mode or cached).
    Decided(PolicyDecision),
    /// Pending user approval (interactive mode).
    Pending {
        /// The approval key for resolving later.
        key: ApprovalKey,
        /// Receiver to await the user's decision.
        rx: oneshot::Receiver<ApprovalDecision>,
        /// The approval request to send to the client.
        approval_request: McpApprovalRequest,
    },
}

/// Coordinates approval flow - supports both interactive and policy-only modes.
pub struct ApprovalManager {
    /// Policy engine for auto-decisions.
    policy_engine: Arc<PolicyEngine>,

    /// Pending interactive approvals awaiting user response.
    pending: DashMap<ApprovalKey, PendingApproval>,

    /// Audit log for all decisions.
    audit_log: Arc<AuditLog>,

    /// Timeout for pending approvals.
    approval_timeout: Duration,
}

impl ApprovalManager {
    /// Create a new approval manager with the given policy engine.
    pub fn new(policy_engine: Arc<PolicyEngine>, audit_log: Arc<AuditLog>) -> Self {
        Self {
            policy_engine,
            pending: DashMap::new(),
            audit_log,
            approval_timeout: DEFAULT_APPROVAL_TIMEOUT,
        }
    }

    /// Set the approval timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.approval_timeout = timeout;
        self
    }

    /// Handle an approval request based on the mode.
    pub async fn handle_approval(
        &self,
        mode: ApprovalMode,
        params: ApprovalParams<'_>,
    ) -> McpResult<ApprovalOutcome> {
        match mode {
            ApprovalMode::Interactive => self.request_interactive(&params).await,
            ApprovalMode::PolicyOnly => {
                let decision = self.policy_engine.evaluate(
                    params.server_key,
                    params.tool_name,
                    params.hints,
                    params.tenant_ctx,
                    params.request_id,
                );
                Ok(ApprovalOutcome::Decided(decision))
            }
        }
    }

    async fn request_interactive(&self, params: &ApprovalParams<'_>) -> McpResult<ApprovalOutcome> {
        let key = ApprovalKey::new(params.request_id, params.server_key, params.elicitation_id);

        if self.pending.contains_key(&key) {
            return Err(ApprovalError::AlreadyPending(key.to_string()).into());
        }

        let (tx, rx) = oneshot::channel();

        let pending = PendingApproval {
            key: key.clone(),
            tool_name: params.tool_name.to_string(),
            hints: params.hints.clone(),
            message: params.message.to_string(),
            created_at: Instant::now(),
            response_tx: tx,
        };

        self.audit_log.record_decision(
            &QualifiedToolName::new(params.server_key, params.tool_name),
            &params.tenant_ctx.tenant_id,
            params.request_id,
            DecisionResult::Pending,
            DecisionSource::UserInteractive,
        );

        self.pending.insert(key.clone(), pending);

        let approval_request = McpApprovalRequest {
            server_key: params.server_key.to_string(),
            tool_name: params.tool_name.to_string(),
            message: params.message.to_string(),
            elicitation_id: params.elicitation_id.to_string(),
        };

        Ok(ApprovalOutcome::Pending {
            key,
            rx,
            approval_request,
        })
    }

    /// Resolve a pending approval with the user's response.
    pub async fn resolve(
        &self,
        request_id: &str,
        server_key: &str,
        elicitation_id: &str,
        approved: bool,
        reason: Option<String>,
        tenant_ctx: &TenantContext,
    ) -> McpResult<()> {
        let key = ApprovalKey::new(request_id, server_key, elicitation_id);

        let (_, pending) = self
            .pending
            .remove(&key)
            .ok_or_else(|| ApprovalError::NotFound(key.to_string()))?;

        let decision = if approved {
            ApprovalDecision::Approved
        } else {
            ApprovalDecision::Denied {
                reason: reason.unwrap_or_else(|| "User denied".to_string()),
            }
        };

        // Log the decision
        let result = if approved {
            DecisionResult::Approved
        } else {
            DecisionResult::Denied {
                reason: match &decision {
                    ApprovalDecision::Denied { reason } => reason.clone(),
                    _ => "User denied".to_string(),
                },
            }
        };

        self.audit_log.record_decision(
            &QualifiedToolName::new(server_key, &pending.tool_name),
            &tenant_ctx.tenant_id,
            request_id,
            result,
            DecisionSource::UserInteractive,
        );

        // Send response to waiting handler
        pending
            .response_tx
            .send(decision)
            .map_err(|_| ApprovalError::ChannelClosed)?;

        Ok(())
    }

    /// Check if there's a pending approval for the given key.
    pub fn has_pending(&self, key: &ApprovalKey) -> bool {
        self.pending.contains_key(key)
    }

    /// Get the count of pending approvals.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Clean up expired pending approvals.
    pub fn evict_expired(&self) {
        let now = Instant::now();
        let timeout = self.approval_timeout;

        self.pending
            .retain(|_, pending| now.duration_since(pending.created_at) < timeout);
    }

    /// Get the policy engine.
    pub fn policy_engine(&self) -> &Arc<PolicyEngine> {
        &self.policy_engine
    }

    /// Get the audit log.
    pub fn audit_log(&self) -> &Arc<AuditLog> {
        &self.audit_log
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_manager() -> ApprovalManager {
        let audit_log = Arc::new(AuditLog::new());
        let policy_engine = Arc::new(PolicyEngine::new(audit_log.clone()));
        ApprovalManager::new(policy_engine, audit_log)
    }

    #[tokio::test]
    async fn test_policy_only_mode() {
        let manager = test_manager();
        let tenant = TenantContext::new("test");
        let hints = ToolAnnotations::new().with_read_only(true);

        let params = ApprovalParams {
            request_id: "req-1",
            server_key: "server",
            elicitation_id: "elicit-1",
            tool_name: "read_tool",
            hints: &hints,
            message: "Allow read?",
            tenant_ctx: &tenant,
        };

        let outcome = manager
            .handle_approval(ApprovalMode::PolicyOnly, params)
            .await
            .unwrap();

        match outcome {
            ApprovalOutcome::Decided(decision) => assert!(decision.is_allowed()),
            ApprovalOutcome::Pending { .. } => panic!("Expected decided outcome"),
        }
    }

    #[tokio::test]
    async fn test_interactive_mode_pending() {
        let manager = test_manager();
        let tenant = TenantContext::new("test");
        let hints = ToolAnnotations::new().with_destructive(true);

        let params = ApprovalParams {
            request_id: "req-1",
            server_key: "server",
            elicitation_id: "elicit-1",
            tool_name: "delete_tool",
            hints: &hints,
            message: "Allow delete?",
            tenant_ctx: &tenant,
        };

        let outcome = manager
            .handle_approval(ApprovalMode::Interactive, params)
            .await
            .unwrap();

        match outcome {
            ApprovalOutcome::Pending {
                key,
                approval_request,
                ..
            } => {
                assert_eq!(key.request_id, "req-1");
                assert_eq!(approval_request.tool_name, "delete_tool");
                assert!(manager.has_pending(&key));
            }
            ApprovalOutcome::Decided(_) => panic!("Expected pending outcome"),
        }
    }

    #[tokio::test]
    async fn test_interactive_resolve() {
        let manager = test_manager();
        let tenant = TenantContext::new("test");
        let hints = ToolAnnotations::new();

        let params = ApprovalParams {
            request_id: "req-1",
            server_key: "server",
            elicitation_id: "elicit-1",
            tool_name: "tool",
            hints: &hints,
            message: "Allow?",
            tenant_ctx: &tenant,
        };

        let outcome = manager
            .handle_approval(ApprovalMode::Interactive, params)
            .await
            .unwrap();

        let rx = match outcome {
            ApprovalOutcome::Pending { rx, .. } => rx,
            _ => panic!("Expected pending"),
        };

        manager
            .resolve("req-1", "server", "elicit-1", true, None, &tenant)
            .await
            .unwrap();

        let decision = rx.await.unwrap();
        assert!(decision.is_approved());
        assert_eq!(manager.pending_count(), 0);
    }

    #[tokio::test]
    async fn test_resolve_not_found() {
        let manager = test_manager();
        let tenant = TenantContext::new("test");

        let result = manager
            .resolve("nonexistent", "server", "elicit", true, None, &tenant)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_duplicate_pending() {
        let manager = test_manager();
        let tenant = TenantContext::new("test");
        let hints = ToolAnnotations::new();

        let params = ApprovalParams {
            request_id: "req-1",
            server_key: "server",
            elicitation_id: "elicit-1",
            tool_name: "tool",
            hints: &hints,
            message: "Allow?",
            tenant_ctx: &tenant,
        };

        manager
            .handle_approval(ApprovalMode::Interactive, params)
            .await
            .unwrap();

        let params2 = ApprovalParams {
            request_id: "req-1",
            server_key: "server",
            elicitation_id: "elicit-1",
            tool_name: "tool",
            hints: &hints,
            message: "Allow?",
            tenant_ctx: &tenant,
        };

        let result = manager
            .handle_approval(ApprovalMode::Interactive, params2)
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn test_evict_expired() {
        let manager = test_manager().with_timeout(Duration::from_millis(1));
        manager.evict_expired();
        assert_eq!(manager.pending_count(), 0);
    }
}
