//! Protocol message types for the `/ws/metrics` endpoint.

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Topics that clients can subscribe to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Topic {
    Workers,
    Loads,
    Cluster,
    Mesh,
    RateLimits,
    Models,
    Metrics,
}

impl Topic {
    pub const ALL: &[Topic] = &[
        Topic::Workers,
        Topic::Loads,
        Topic::Cluster,
        Topic::Mesh,
        Topic::RateLimits,
        Topic::Models,
        Topic::Metrics,
    ];
}

/// Client -> server message types.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    Subscribe {
        #[serde(default = "default_topics", deserialize_with = "deserialize_topics")]
        topics: HashSet<Topic>,
    },
}

fn default_topics() -> HashSet<Topic> {
    Topic::ALL.iter().copied().collect()
}

/// Deserialize topics, treating an empty set the same as absent (= all topics).
fn deserialize_topics<'de, D>(deserializer: D) -> Result<HashSet<Topic>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let topics = HashSet::<Topic>::deserialize(deserializer)?;
    if topics.is_empty() {
        Ok(default_topics())
    } else {
        Ok(topics)
    }
}

/// Server -> client: full state snapshot for a topic.
#[derive(Debug, Serialize)]
pub struct SnapshotMessage {
    #[serde(rename = "type")]
    msg_type: &'static str,
    pub topic: Topic,
    pub timestamp: DateTime<Utc>,
    pub data: serde_json::Value,
}

impl SnapshotMessage {
    pub fn new(topic: Topic, data: serde_json::Value) -> Self {
        Self {
            msg_type: "snapshot",
            topic,
            timestamp: Utc::now(),
            data,
        }
    }
}

/// Server -> client: significant state change event.
#[derive(Debug, Serialize)]
pub struct EventMessage {
    #[serde(rename = "type")]
    msg_type: &'static str,
    pub topic: Topic,
    pub timestamp: DateTime<Utc>,
    pub event: String,
    pub data: serde_json::Value,
}

impl EventMessage {
    pub fn new(topic: Topic, event: String, data: serde_json::Value) -> Self {
        Self {
            msg_type: "event",
            topic,
            timestamp: Utc::now(),
            event,
            data,
        }
    }
}

/// Envelope for all server-to-client messages.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ServerMessage {
    Snapshot(SnapshotMessage),
    Event(EventMessage),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topic_all_contains_seven_topics() {
        assert_eq!(Topic::ALL.len(), 7);
    }

    #[test]
    fn snapshot_message_serializes_correctly() {
        let msg = ServerMessage::Snapshot(SnapshotMessage::new(
            Topic::Workers,
            serde_json::json!({"total": 5}),
        ));
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "snapshot");
        assert_eq!(parsed["topic"], "workers");
        assert_eq!(parsed["data"]["total"], 5);
        assert!(parsed["timestamp"].is_string());
    }

    #[test]
    fn event_message_serializes_correctly() {
        let msg = ServerMessage::Event(EventMessage::new(
            Topic::Workers,
            "worker_health_changed".to_string(),
            serde_json::json!({"worker_id": "w1", "healthy": false}),
        ));
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "event");
        assert_eq!(parsed["topic"], "workers");
        assert_eq!(parsed["event"], "worker_health_changed");
    }

    #[test]
    fn subscribe_message_deserializes() {
        let json = r#"{"type":"subscribe","topics":["workers","metrics"]}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClientMessage::Subscribe { topics } => {
                assert_eq!(topics.len(), 2);
                assert!(topics.contains(&Topic::Workers));
                assert!(topics.contains(&Topic::Metrics));
            }
        }
    }

    #[test]
    fn subscribe_all_topics_deserializes() {
        let json = r#"{"type":"subscribe","topics":["workers","loads","cluster","mesh","rate_limits","models","metrics"]}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClientMessage::Subscribe { topics } => assert_eq!(topics.len(), 7),
        }
    }

    #[test]
    fn subscribe_without_topics_defaults_to_all() {
        let json = r#"{"type":"subscribe"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClientMessage::Subscribe { topics } => assert_eq!(topics.len(), 7),
        }
    }

    #[test]
    fn subscribe_deduplicates_topics() {
        let json = r#"{"type":"subscribe","topics":["workers","workers","metrics"]}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClientMessage::Subscribe { topics } => assert_eq!(topics.len(), 2),
        }
    }

    #[test]
    fn subscribe_empty_topics_defaults_to_all() {
        let json = r#"{"type":"subscribe","topics":[]}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClientMessage::Subscribe { topics } => assert_eq!(topics.len(), 7),
        }
    }

    #[test]
    fn invalid_message_type_rejected() {
        let json = r#"{"type":"ping","topics":["workers"]}"#;
        let result = serde_json::from_str::<ClientMessage>(json);
        assert!(result.is_err());
    }

    #[test]
    fn topic_serializes_snake_case() {
        let json = serde_json::to_string(&Topic::RateLimits).unwrap();
        assert_eq!(json, r#""rate_limits""#);
    }
}
