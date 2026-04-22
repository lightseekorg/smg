use chrono::{DateTime, Duration, Utc};
use serde_json::json;
use smg_data_connector::{
    ConversationId, ConversationMemoryStatus, ConversationMemoryType, NewConversationMemory,
    ResponseId,
};

use super::{MemoryExecutionContext, MemoryPolicyMode};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnqueueSkipReason {
    SubjectId,
    EmbeddingModel,
    ExtractionModel,
}

#[derive(Debug, Clone)]
pub struct EnqueueInputs {
    pub now: DateTime<Utc>,
    pub memory_execution_context: MemoryExecutionContext,
    pub conversation_id: ConversationId,
    pub response_id: Option<ResponseId>,
    pub user_text: Option<String>,
    pub assistant_text: Option<String>,
}

#[derive(Debug, Clone)]
pub struct EnqueuePlan {
    pub rows: Vec<NewConversationMemory>,
}

pub fn build_enqueue_plan(inputs: EnqueueInputs) -> Result<Option<EnqueuePlan>, EnqueueSkipReason> {
    if !inputs.memory_execution_context.store_ltm.active() {
        return Ok(None);
    }

    let subject_id = required_value(
        inputs.memory_execution_context.subject_id.clone(),
        EnqueueSkipReason::SubjectId,
    )?;
    let embedding_model = required_value(
        inputs.memory_execution_context.embedding_model.clone(),
        EnqueueSkipReason::EmbeddingModel,
    )?;
    let extraction_model = required_value(
        inputs.memory_execution_context.extraction_model.clone(),
        EnqueueSkipReason::ExtractionModel,
    )?;

    let memory_config = json!({
        "policy": policy_mode_label(inputs.memory_execution_context.policy_mode),
        "subject_id": subject_id,
        "embedding_model_id": embedding_model,
        "extraction_model_id": extraction_model,
    })
    .to_string();

    let mut rows = vec![NewConversationMemory {
        conversation_id: inputs.conversation_id.clone(),
        conversation_version: None,
        response_id: inputs.response_id.clone(),
        memory_type: ConversationMemoryType::Ltm,
        status: ConversationMemoryStatus::Ready,
        attempt: 0,
        owner_id: None,
        next_run_at: inputs.now + Duration::seconds(30),
        lease_until: None,
        content: None,
        memory_config: Some(memory_config.clone()),
        scope_id: None,
        error_msg: None,
    }];

    let user_text = normalize_text(inputs.user_text);
    let assistant_text = normalize_text(inputs.assistant_text);

    if user_text.is_some() || assistant_text.is_some() {
        rows.push(NewConversationMemory {
            conversation_id: inputs.conversation_id,
            conversation_version: None,
            response_id: inputs.response_id,
            memory_type: ConversationMemoryType::OnDemand,
            status: ConversationMemoryStatus::Ready,
            attempt: 0,
            owner_id: None,
            next_run_at: inputs.now,
            lease_until: None,
            content: Some(
                json!({
                    "user_text": user_text,
                    "assistant_text": assistant_text,
                })
                .to_string(),
            ),
            memory_config: Some(memory_config),
            scope_id: None,
            error_msg: None,
        });
    }

    Ok(Some(EnqueuePlan { rows }))
}

fn required_value(
    value: Option<String>,
    reason: EnqueueSkipReason,
) -> Result<String, EnqueueSkipReason> {
    value
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .ok_or(reason)
}

fn normalize_text(value: Option<String>) -> Option<String> {
    value
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn policy_mode_label(mode: MemoryPolicyMode) -> &'static str {
    match mode {
        MemoryPolicyMode::StoreOnly => "store_only",
        MemoryPolicyMode::StoreAndRecall => "store_and_recall",
        MemoryPolicyMode::RecallOnly => "recall_only",
        MemoryPolicyMode::ExplicitNone => "none",
        MemoryPolicyMode::Unspecified => "unspecified",
        MemoryPolicyMode::Unrecognized => "unrecognized",
    }
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, Utc};
    use smg_data_connector::{
        ConversationId, ConversationMemoryStatus, ConversationMemoryType, ResponseId,
    };

    use super::{build_enqueue_plan, EnqueueInputs};
    use crate::memory::{MemoryExecutionContext, MemoryExecutionState, MemoryPolicyMode};

    fn active_store_ctx() -> MemoryExecutionContext {
        MemoryExecutionContext {
            store_ltm: MemoryExecutionState::Active,
            recall: MemoryExecutionState::Active,
            policy_mode: MemoryPolicyMode::StoreAndRecall,
            subject_id: Some("subject_123".to_string()),
            embedding_model: Some("text-embedding-3-small".to_string()),
            extraction_model: Some("gpt-4.1-mini".to_string()),
        }
    }

    #[test]
    fn responses_path_builds_ltm_and_ondemand_rows() {
        let now = Utc::now();
        let plan = build_enqueue_plan(EnqueueInputs {
            now,
            memory_execution_context: active_store_ctx(),
            conversation_id: ConversationId::from("conv_123"),
            response_id: Some(ResponseId::from("resp_123")),
            user_text: Some("hello".to_string()),
            assistant_text: Some("world".to_string()),
        })
        .expect("planner should succeed")
        .expect("enqueue should be active");

        assert_eq!(plan.rows.len(), 2);
        assert_eq!(plan.rows[0].memory_type, ConversationMemoryType::Ltm);
        assert_eq!(plan.rows[0].status, ConversationMemoryStatus::Ready);
        assert_eq!(plan.rows[0].next_run_at, now + Duration::seconds(30));
        assert_eq!(plan.rows[1].memory_type, ConversationMemoryType::OnDemand);
        assert_eq!(plan.rows[1].status, ConversationMemoryStatus::Ready);
        assert_eq!(plan.rows[1].next_run_at, now);
        assert_eq!(plan.rows[0].conversation_version, None);
        assert_eq!(plan.rows[1].conversation_version, None);
    }
}
