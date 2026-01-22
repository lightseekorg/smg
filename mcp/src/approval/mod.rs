//! Approval system for MCP tool execution.

pub mod audit;
pub mod manager;
pub mod policy;

pub use audit::{AuditEntry, AuditLog, DecisionResult, DecisionSource};
pub use manager::{
    ApprovalDecision, ApprovalKey, ApprovalManager, ApprovalMode, ApprovalOutcome, ApprovalParams,
    McpApprovalRequest, McpApprovalResponse,
};
pub use policy::{
    PolicyDecision, PolicyEngine, PolicyRule, RuleCondition, RulePattern, ServerPolicy, TrustLevel,
};
