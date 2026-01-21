//! Policy engine for MCP tool approval decisions.

use std::{collections::HashMap, sync::Arc};

use regex::Regex;
use serde::{Deserialize, Serialize};

use super::audit::{AuditLog, DecisionResult, DecisionSource};
use crate::{
    annotations::{AnnotationType, ToolAnnotations},
    inventory::QualifiedToolName,
    tenant::TenantContext,
};

/// Result of a policy evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PolicyDecision {
    #[default]
    Allow,
    Deny,
    DenyWithReason(String),
}

impl PolicyDecision {
    pub fn is_allowed(&self) -> bool {
        matches!(self, PolicyDecision::Allow)
    }

    pub fn denial_reason(&self) -> Option<&str> {
        match self {
            PolicyDecision::DenyWithReason(reason) => Some(reason),
            PolicyDecision::Deny => Some("Policy denied"),
            PolicyDecision::Allow => None,
        }
    }
}

/// Trust level for an MCP server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TrustLevel {
    Trusted,
    #[default]
    Standard,
    Untrusted,
    Sandboxed,
}

/// Server-level policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerPolicy {
    pub default: PolicyDecision,
    pub trust_level: TrustLevel,
}

impl Default for ServerPolicy {
    fn default() -> Self {
        Self {
            default: PolicyDecision::Allow,
            trust_level: TrustLevel::Standard,
        }
    }
}

/// Tool-level policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPolicy {
    pub decision: PolicyDecision,
}

/// Pattern for matching in policy rules.
#[derive(Debug, Clone)]
pub enum RulePattern {
    Server(Regex),
    Tool(Regex),
    Qualified(Regex),
    Any,
}

impl RulePattern {
    pub fn matches(&self, server_key: &str, tool_name: &str) -> bool {
        match self {
            RulePattern::Server(re) => re.is_match(server_key),
            RulePattern::Tool(re) => re.is_match(tool_name),
            RulePattern::Qualified(re) => {
                let qualified = format!("{}/{}", server_key, tool_name);
                re.is_match(&qualified)
            }
            RulePattern::Any => true,
        }
    }
}

/// Condition for policy rules.
#[derive(Debug, Clone)]
pub enum RuleCondition {
    Always,
    HasAnnotation(AnnotationType),
    LacksAnnotation(AnnotationType),
}

impl RuleCondition {
    pub fn evaluate(&self, hints: &ToolAnnotations) -> bool {
        match self {
            RuleCondition::Always => true,
            RuleCondition::HasAnnotation(ann_type) => ann_type.matches(hints),
            RuleCondition::LacksAnnotation(ann_type) => !ann_type.matches(hints),
        }
    }
}

/// A policy rule combining pattern, condition, and decision.
#[derive(Debug, Clone)]
pub struct PolicyRule {
    pub name: String,
    pub pattern: RulePattern,
    pub condition: RuleCondition,
    pub decision: PolicyDecision,
}

impl PolicyRule {
    pub fn new(
        name: impl Into<String>,
        pattern: RulePattern,
        condition: RuleCondition,
        decision: PolicyDecision,
    ) -> Self {
        Self {
            name: name.into(),
            pattern,
            condition,
            decision,
        }
    }

    pub fn evaluate(
        &self,
        server_key: &str,
        tool_name: &str,
        hints: &ToolAnnotations,
    ) -> Option<PolicyDecision> {
        if self.pattern.matches(server_key, tool_name) && self.condition.evaluate(hints) {
            Some(self.decision.clone())
        } else {
            None
        }
    }
}

/// Policy engine configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PolicyConfig {
    #[serde(default = "default_policy")]
    pub default_policy: PolicyDecision,
    #[serde(default)]
    pub server_policies: HashMap<String, ServerPolicy>,
}

fn default_policy() -> PolicyDecision {
    PolicyDecision::Allow
}

/// Policy-based approval handler for MCP tools.
pub struct PolicyEngine {
    default_policy: PolicyDecision,
    server_policies: HashMap<String, ServerPolicy>,
    tool_policies: HashMap<QualifiedToolName, ToolPolicy>,
    rules: Vec<PolicyRule>,
    audit_log: Arc<AuditLog>,
}

impl PolicyEngine {
    pub fn new(audit_log: Arc<AuditLog>) -> Self {
        Self {
            default_policy: PolicyDecision::Allow,
            server_policies: HashMap::new(),
            tool_policies: HashMap::new(),
            rules: Vec::new(),
            audit_log,
        }
    }

    pub fn from_config(config: &PolicyConfig, audit_log: Arc<AuditLog>) -> Self {
        Self {
            default_policy: config.default_policy.clone(),
            server_policies: config.server_policies.clone(),
            tool_policies: HashMap::new(),
            rules: Vec::new(),
            audit_log,
        }
    }

    pub fn with_default_policy(mut self, policy: PolicyDecision) -> Self {
        self.default_policy = policy;
        self
    }

    pub fn with_server_policy(
        mut self,
        server_key: impl Into<String>,
        policy: ServerPolicy,
    ) -> Self {
        self.server_policies.insert(server_key.into(), policy);
        self
    }

    pub fn with_tool_policy(
        mut self,
        qualified_name: QualifiedToolName,
        policy: ToolPolicy,
    ) -> Self {
        self.tool_policies.insert(qualified_name, policy);
        self
    }

    pub fn with_rule(mut self, rule: PolicyRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Evaluate a tool execution request and return a decision.
    pub fn evaluate(
        &self,
        server_key: &str,
        tool_name: &str,
        hints: &ToolAnnotations,
        tenant_ctx: &TenantContext,
        request_id: &str,
    ) -> PolicyDecision {
        let qualified = QualifiedToolName::new(server_key, tool_name);

        // 1. Check explicit tool policy
        if let Some(policy) = self.tool_policies.get(&qualified) {
            self.log_decision(
                &qualified,
                tenant_ctx,
                request_id,
                &policy.decision,
                DecisionSource::ExplicitToolPolicy,
            );
            return policy.decision.clone();
        }

        // 2. Check server policy + trust level
        if let Some(server_policy) = self.server_policies.get(server_key) {
            let decision =
                self.evaluate_with_trust(&server_policy.trust_level, hints, &server_policy.default);
            if matches!(server_policy.trust_level, TrustLevel::Trusted)
                || !matches!(decision, PolicyDecision::Allow)
            {
                self.log_decision(
                    &qualified,
                    tenant_ctx,
                    request_id,
                    &decision,
                    DecisionSource::ServerPolicy,
                );
                return decision;
            }
        }

        // 3. Evaluate pattern-based rules in order
        for rule in &self.rules {
            if let Some(decision) = rule.evaluate(server_key, tool_name, hints) {
                self.log_decision(
                    &qualified,
                    tenant_ctx,
                    request_id,
                    &decision,
                    DecisionSource::RuleMatch,
                );
                return decision;
            }
        }

        // 4. Apply annotation-based defaults
        let decision = self.annotation_based_decision(hints);
        self.log_decision(
            &qualified,
            tenant_ctx,
            request_id,
            &decision,
            DecisionSource::AnnotationDefault,
        );
        decision
    }

    fn evaluate_with_trust(
        &self,
        trust_level: &TrustLevel,
        hints: &ToolAnnotations,
        server_default: &PolicyDecision,
    ) -> PolicyDecision {
        match trust_level {
            TrustLevel::Trusted => PolicyDecision::Allow,
            TrustLevel::Standard => server_default.clone(),
            TrustLevel::Untrusted => {
                if hints.destructive && !hints.read_only {
                    PolicyDecision::DenyWithReason(
                        "Untrusted server: destructive operation denied".into(),
                    )
                } else {
                    server_default.clone()
                }
            }
            TrustLevel::Sandboxed => {
                if hints.open_world {
                    PolicyDecision::DenyWithReason(
                        "Sandboxed server: external access denied".into(),
                    )
                } else if hints.read_only {
                    PolicyDecision::Allow
                } else {
                    PolicyDecision::DenyWithReason(
                        "Sandboxed server: write operations denied".into(),
                    )
                }
            }
        }
    }

    fn annotation_based_decision(&self, hints: &ToolAnnotations) -> PolicyDecision {
        if hints.read_only {
            PolicyDecision::Allow
        } else if hints.destructive {
            PolicyDecision::DenyWithReason("Destructive operation requires explicit policy".into())
        } else {
            self.default_policy.clone()
        }
    }

    fn log_decision(
        &self,
        qualified: &QualifiedToolName,
        tenant_ctx: &TenantContext,
        request_id: &str,
        decision: &PolicyDecision,
        source: DecisionSource,
    ) {
        let result = match decision {
            PolicyDecision::Allow => DecisionResult::Approved,
            PolicyDecision::Deny => DecisionResult::Denied {
                reason: "Policy denied".to_string(),
            },
            PolicyDecision::DenyWithReason(reason) => DecisionResult::Denied {
                reason: reason.clone(),
            },
        };
        self.audit_log.record_decision(
            qualified,
            &tenant_ctx.tenant_id,
            request_id,
            result,
            source,
        );
    }

    pub fn audit_log(&self) -> &Arc<AuditLog> {
        &self.audit_log
    }
}

impl Default for PolicyEngine {
    fn default() -> Self {
        let audit_log = Arc::new(AuditLog::new());
        Self::new(audit_log).with_rule(PolicyRule::new(
            "allow_read_only",
            RulePattern::Any,
            RuleCondition::HasAnnotation(AnnotationType::ReadOnly),
            PolicyDecision::Allow,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_engine() -> PolicyEngine {
        PolicyEngine::new(Arc::new(AuditLog::new()))
    }

    #[test]
    fn test_default_policy() {
        let engine = test_engine().with_default_policy(PolicyDecision::Allow);
        let tenant = TenantContext::new("test");
        let hints = ToolAnnotations::new();

        let decision = engine.evaluate("server", "tool", &hints, &tenant, "req-1");
        assert!(decision.is_allowed());
    }

    #[test]
    fn test_tool_policy() {
        let engine = test_engine().with_tool_policy(
            QualifiedToolName::new("server", "dangerous_tool"),
            ToolPolicy {
                decision: PolicyDecision::Deny,
            },
        );

        let tenant = TenantContext::new("test");
        let hints = ToolAnnotations::new();

        let decision = engine.evaluate("server", "dangerous_tool", &hints, &tenant, "req-1");
        assert!(!decision.is_allowed());
    }

    #[test]
    fn test_server_trust_level() {
        let engine = test_engine().with_server_policy(
            "trusted_server",
            ServerPolicy {
                default: PolicyDecision::Allow,
                trust_level: TrustLevel::Trusted,
            },
        );

        let tenant = TenantContext::new("test");
        let hints = ToolAnnotations::new().with_destructive(true);

        let decision = engine.evaluate("trusted_server", "tool", &hints, &tenant, "req-1");
        assert!(decision.is_allowed());
    }

    #[test]
    fn test_untrusted_server() {
        let engine = test_engine().with_server_policy(
            "untrusted",
            ServerPolicy {
                default: PolicyDecision::Allow,
                trust_level: TrustLevel::Untrusted,
            },
        );

        let tenant = TenantContext::new("test");
        let destructive = ToolAnnotations::new().with_destructive(true);
        let read_only = ToolAnnotations::new().with_read_only(true);

        let decision = engine.evaluate("untrusted", "tool", &destructive, &tenant, "req-1");
        assert!(!decision.is_allowed());

        let decision = engine.evaluate("untrusted", "tool", &read_only, &tenant, "req-2");
        assert!(decision.is_allowed());
    }

    #[test]
    fn test_sandboxed_server() {
        let engine = test_engine().with_server_policy(
            "sandbox",
            ServerPolicy {
                default: PolicyDecision::Allow,
                trust_level: TrustLevel::Sandboxed,
            },
        );

        let tenant = TenantContext::new("test");
        let open_world = ToolAnnotations::new().with_open_world(true);
        let read_only = ToolAnnotations::new().with_read_only(true);

        let decision = engine.evaluate("sandbox", "tool", &open_world, &tenant, "req-1");
        assert!(!decision.is_allowed());

        let decision = engine.evaluate("sandbox", "tool", &read_only, &tenant, "req-2");
        assert!(decision.is_allowed());
    }

    #[test]
    fn test_pattern_rule() {
        let engine = test_engine()
            .with_default_policy(PolicyDecision::Allow)
            .with_rule(PolicyRule::new(
                "block_delete",
                RulePattern::Tool(Regex::new("^delete_").unwrap()),
                RuleCondition::Always,
                PolicyDecision::Deny,
            ));

        let tenant = TenantContext::new("test");
        let hints = ToolAnnotations::new();

        let decision = engine.evaluate("server", "delete_user", &hints, &tenant, "req-1");
        assert!(!decision.is_allowed());

        let decision = engine.evaluate("server", "get_user", &hints, &tenant, "req-2");
        assert!(decision.is_allowed());
    }

    #[test]
    fn test_annotation_based_decision() {
        let engine = test_engine().with_default_policy(PolicyDecision::Deny);
        let tenant = TenantContext::new("test");

        let read_only = ToolAnnotations::new().with_read_only(true);
        let decision = engine.evaluate("server", "tool", &read_only, &tenant, "req-1");
        assert!(decision.is_allowed());
    }
}
