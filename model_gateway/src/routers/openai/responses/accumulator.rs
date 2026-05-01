//! Streaming response accumulator for persisting responses.

use openai_protocol::event_types::{OutputItemEvent, ResponseEvent};
use serde_json::Value;
use tracing::warn;

use super::common::{extract_output_index, get_event_type, parse_sse_block};

/// Helper that parses SSE frames from the OpenAI responses stream and
/// accumulates enough information to persist the final response locally.
pub(crate) struct StreamingResponseAccumulator {
    /// The initial `response.created` payload (if emitted).
    initial_response: Option<Value>,
    /// The final `response.completed` / `response.incomplete` payload (if emitted).
    completed_response: Option<Value>,
    /// Collected output items keyed by the upstream output index, used when
    /// a final response payload is absent and we need to synthesize one.
    output_items: Vec<(usize, Value)>,
    /// Captured error payload (if the upstream stream fails midway).
    encountered_error: Option<Value>,
}

impl StreamingResponseAccumulator {
    pub fn new() -> Self {
        Self {
            initial_response: None,
            completed_response: None,
            output_items: Vec::new(),
            encountered_error: None,
        }
    }

    /// Feed the accumulator with the next SSE chunk.
    pub fn ingest_block(&mut self, block: &str) {
        if block.trim().is_empty() {
            return;
        }
        self.process_block(block);
    }

    /// Consume the accumulator and produce the best-effort final response value.
    pub fn original_response_id(&self) -> Option<&str> {
        self.initial_response
            .as_ref()
            .and_then(|response| response.get("id"))
            .and_then(|id| id.as_str())
    }

    pub fn snapshot_final_response(&self) -> Option<Value> {
        if let Some(resp) = &self.completed_response {
            return Some(resp.clone());
        }
        self.build_fallback_response_snapshot()
    }

    fn build_fallback_response_snapshot(&self) -> Option<Value> {
        let mut response = self.initial_response.clone()?;

        if let Some(obj) = response.as_object_mut() {
            obj.insert("status".to_string(), Value::String("completed".to_string()));

            let mut output_items = self.output_items.clone();
            output_items.sort_by_key(|(index, _)| *index);
            let outputs: Vec<Value> = output_items.into_iter().map(|(_, item)| item).collect();
            obj.insert("output".to_string(), Value::Array(outputs));
        }

        Some(response)
    }

    fn process_block(&mut self, block: &str) {
        let trimmed = block.trim();
        if trimmed.is_empty() {
            return;
        }

        let (event_name, data) = parse_sse_block(trimmed);
        if data.is_empty() {
            return;
        }

        self.handle_event(event_name, &data);
    }

    fn handle_event(&mut self, event_name: Option<&str>, data_payload: &str) {
        let parsed: Value = match serde_json::from_str(data_payload) {
            Ok(value) => value,
            Err(err) => {
                warn!("Failed to parse streaming event JSON: {}", err);
                return;
            }
        };

        match get_event_type(event_name, &parsed) {
            ResponseEvent::CREATED if self.initial_response.is_none() => {
                if let Some(response) = parsed.get("response") {
                    self.initial_response = Some(response.clone());
                }
            }
            ResponseEvent::COMPLETED | ResponseEvent::INCOMPLETE => {
                if let Some(response) = parsed.get("response") {
                    self.completed_response = Some(response.clone());
                }
            }
            OutputItemEvent::DONE => {
                if let (Some(index), Some(item)) =
                    (extract_output_index(&parsed), parsed.get("item"))
                {
                    self.output_items.push((index, item.clone()));
                }
            }
            "response.error" => {
                self.encountered_error = Some(parsed);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn captures_response_incomplete_as_final_response() {
        let mut accumulator = StreamingResponseAccumulator::new();
        let payload = json!({
            "type": ResponseEvent::INCOMPLETE,
            "response": {
                "id": "resp_1",
                "status": "incomplete",
                "incomplete_details": { "reason": "max_output_tokens" },
                "output": []
            }
        });

        accumulator.ingest_block(&format!(
            "event: {}\ndata: {}\n\n",
            ResponseEvent::INCOMPLETE,
            payload
        ));

        let response = accumulator
            .snapshot_final_response()
            .expect("final response snapshot");
        assert_eq!(response["status"], "incomplete");
        assert_eq!(
            response["incomplete_details"]["reason"],
            "max_output_tokens"
        );
    }
}
