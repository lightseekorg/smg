//! Audit logging for MCP approval decisions.

use std::{collections::VecDeque, sync::RwLock};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{inventory::QualifiedToolName, tenant::TenantId};

const DEFAULT_MAX_ENTRIES: usize = 10000;

/// Source of an approval decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionSource {
    UserInteractive,
    PolicyEngine,
    ExplicitToolPolicy,
    ServerPolicy,
    RuleMatch,
    AnnotationDefault,
    GlobalDefault,
    Timeout,
}

/// Result of an approval decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionResult {
    Approved,
    Denied { reason: String },
    Pending,
    TimedOut,
}

impl DecisionResult {
    pub fn is_approved(&self) -> bool {
        matches!(self, DecisionResult::Approved)
    }

    pub fn is_final(&self) -> bool {
        !matches!(self, DecisionResult::Pending)
    }
}

/// A single audit log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub tenant_id: TenantId,
    pub request_id: String,
    pub server_key: String,
    pub tool_name: String,
    pub result: DecisionResult,
    pub source: DecisionSource,
}

impl AuditEntry {
    pub fn new(
        tenant_id: TenantId,
        request_id: String,
        server_key: String,
        tool_name: String,
        result: DecisionResult,
        source: DecisionSource,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            tenant_id,
            request_id,
            server_key,
            tool_name,
            result,
            source,
        }
    }

    pub fn qualified_name(&self) -> QualifiedToolName {
        QualifiedToolName::new(&self.server_key, &self.tool_name)
    }
}

/// Thread-safe audit log for approval decisions.
#[derive(Debug)]
pub struct AuditLog {
    entries: RwLock<VecDeque<AuditEntry>>,
    max_entries: usize,
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

impl AuditLog {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_ENTRIES)
    }

    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            entries: RwLock::new(VecDeque::with_capacity(max_entries)),
            max_entries,
        }
    }

    pub fn record(&self, entry: AuditEntry) {
        let mut entries = self.entries.write().unwrap();
        if entries.len() >= self.max_entries {
            entries.pop_front();
        }
        entries.push_back(entry);
    }

    pub fn record_decision(
        &self,
        qualified_name: &QualifiedToolName,
        tenant_id: &TenantId,
        request_id: &str,
        result: DecisionResult,
        source: DecisionSource,
    ) {
        self.record(AuditEntry::new(
            tenant_id.clone(),
            request_id.to_string(),
            qualified_name.server_key.clone(),
            qualified_name.tool_name.clone(),
            result,
            source,
        ));
    }

    pub fn recent(&self, limit: usize) -> Vec<AuditEntry> {
        let entries = self.entries.read().unwrap();
        entries.iter().rev().take(limit).cloned().collect()
    }

    pub fn for_tenant(&self, tenant_id: &TenantId, limit: usize) -> Vec<AuditEntry> {
        let entries = self.entries.read().unwrap();
        entries
            .iter()
            .rev()
            .filter(|e| &e.tenant_id == tenant_id)
            .take(limit)
            .cloned()
            .collect()
    }

    pub fn for_request(&self, request_id: &str) -> Vec<AuditEntry> {
        let entries = self.entries.read().unwrap();
        entries
            .iter()
            .filter(|e| e.request_id == request_id)
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.entries.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().unwrap().is_empty()
    }

    pub fn clear(&self) {
        self.entries.write().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_log_record() {
        let log = AuditLog::new();
        let tenant = TenantId::new("test");

        log.record_decision(
            &QualifiedToolName::new("server", "tool"),
            &tenant,
            "req-1",
            DecisionResult::Approved,
            DecisionSource::PolicyEngine,
        );

        assert_eq!(log.len(), 1);
        let recent = log.recent(10);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].tool_name, "tool");
    }

    #[test]
    fn test_audit_log_max_entries() {
        let log = AuditLog::with_capacity(5);
        let tenant = TenantId::new("test");

        for i in 0..10 {
            log.record_decision(
                &QualifiedToolName::new("server", format!("tool-{}", i)),
                &tenant,
                &format!("req-{}", i),
                DecisionResult::Approved,
                DecisionSource::PolicyEngine,
            );
        }

        assert_eq!(log.len(), 5);
        let recent = log.recent(10);
        assert_eq!(recent[0].tool_name, "tool-9");
        assert_eq!(recent[4].tool_name, "tool-5");
    }

    #[test]
    fn test_filter_by_tenant() {
        let log = AuditLog::new();
        let tenant1 = TenantId::new("tenant1");
        let tenant2 = TenantId::new("tenant2");
        let name = QualifiedToolName::new("server", "tool");

        log.record_decision(
            &name,
            &tenant1,
            "r1",
            DecisionResult::Approved,
            DecisionSource::PolicyEngine,
        );
        log.record_decision(
            &name,
            &tenant2,
            "r2",
            DecisionResult::Approved,
            DecisionSource::PolicyEngine,
        );
        log.record_decision(
            &name,
            &tenant1,
            "r3",
            DecisionResult::Approved,
            DecisionSource::PolicyEngine,
        );

        let t1_entries = log.for_tenant(&tenant1, 10);
        assert_eq!(t1_entries.len(), 2);
    }
}
