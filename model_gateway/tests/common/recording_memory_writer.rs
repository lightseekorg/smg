use std::sync::Arc;

use async_trait::async_trait;
use smg_data_connector::{
    ConversationMemoryId, ConversationMemoryResult, ConversationMemoryWriter, NewConversationMemory,
};
use tokio::sync::Mutex;
use uuid::Uuid;

#[derive(Clone, Default)]
pub struct RecordingConversationMemoryWriter {
    rows: Arc<Mutex<Vec<NewConversationMemory>>>,
}

impl RecordingConversationMemoryWriter {
    pub async fn snapshot(&self) -> Vec<NewConversationMemory> {
        self.rows.lock().await.clone()
    }
}

#[async_trait]
impl ConversationMemoryWriter for RecordingConversationMemoryWriter {
    async fn create_memory(
        &self,
        input: NewConversationMemory,
    ) -> ConversationMemoryResult<ConversationMemoryId> {
        self.rows.lock().await.push(input);
        Ok(ConversationMemoryId(format!("mem_{}", Uuid::now_v7())))
    }
}
